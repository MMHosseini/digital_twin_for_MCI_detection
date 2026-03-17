from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import librosa
import os
import json
from pathlib import Path
import subprocess
from tqdm import tqdm
import math
from control_panel import ControlPanel
from speechbrain.pretrained import EncoderClassifier
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
import torchaudio
import torch

controller = ControlPanel()

class AudioHandler:
    def extract_audio_from_video(self):
        def extract_chunks_start_and_end(transcript):
            key_start_end_tuple_list = []
            transcript_keys = transcript.keys()
            for key in transcript_keys:
                start = transcript[key]['start']
                end = transcript[key]['end']
                key_start_end_tuple_list.append((key, (start, end)))
            return key_start_end_tuple_list

        def extract_audio_chunks(video_dir, audio_dir, chunks_list, sample_rate=16000):
            for key, (start, end) in chunks_list:
                out_file = os.path.join(audio_dir, key + '.wav')
                duration = str(end - start)
                cmd = [
                    "ffmpeg", "-y",
                    "-hide_banner", "-loglevel", "error",
                    "-ss", str(start),
                    "-i", str(video_dir),
                    "-t", duration,
                    "-vn",  # remove video
                    "-ac", "1",  # mono
                    "-ar", str(sample_rate),
                    "-c:a", "pcm_s16le",  # WAV PCM
                    str(out_file)
                ]
                try:
                    subprocess.run(cmd, check=True)
                except Exception as e:
                    print("I could not save", out_file)

        subsets = [controller.train_data_path, controller.validation_data_path]
        for subset in subsets:
            participants = os.listdir(subset)
            for participant in participants:
                current_path = os.path.join(subset, participant)
                videos_path = [str(p) for p in Path(current_path).glob("*.mp4")]

                for video_path in tqdm(videos_path):
                    video_path = Path(video_path)
                    transcript_path = Path(str(video_path).replace('.mp4', '.json'))
                    with open(transcript_path) as f0:
                        data = json.load(f0)

                    moderator_transcript = data['moderator_transcript']
                    participant_transcript = data['participant_transcript']
                    moderator_chunks = extract_chunks_start_and_end(moderator_transcript)
                    participant_chunks = extract_chunks_start_and_end(participant_transcript)
                    audio_save_dir = os.path.join(video_path.parent, video_path.stem)
                    os.makedirs(audio_save_dir, exist_ok=True)
                    extract_audio_chunks(video_path, audio_save_dir, moderator_chunks, sample_rate=16000)
                    extract_audio_chunks(video_path, audio_save_dir, participant_chunks, sample_rate=16000)


    def extract_audio_prosody(self):
        def _stats(x: np.ndarray) -> Dict[str, float]:
            x = np.asarray(x, dtype=np.float32)
            if x.size == 0:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
            return {
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "p25": float(np.percentile(x, 25)),
                "p50": float(np.percentile(x, 50)),
                "p75": float(np.percentile(x, 75)),
            }

        def extract_sentence_features(wav_path, cfg):
            wav_path = Path(wav_path)
            y, sr = librosa.load(wav_path, sr=cfg['sr'], mono=True)
            dur = len(y) / sr
            # --- Energy (RMS)
            rms = librosa.feature.rms(y=y, frame_length=cfg['frame_length'], hop_length=cfg['hop_length'])[0]
            # --- Pitch (F0) via librosa.pyin (robust, a bit slower)
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
                frame_length=cfg['frame_length'],
                hop_length=cfg['hop_length'],
            )
            f0_voiced = f0[~np.isnan(f0)]
            voiced_ratio = float(np.mean(~np.isnan(f0))) if f0 is not None else 0.0
            # --- MFCCs
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg['n_mfcc'], hop_length=cfg['hop_length'])
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            # --- Spectral features
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=cfg['hop_length'])[0]
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=cfg['hop_length'])[0]
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=cfg['hop_length'])[0]
            flatness = librosa.feature.spectral_flatness(y=y, hop_length=cfg['hop_length'])[0]
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=cfg['frame_length'], hop_length=cfg['hop_length'])[0]
            # --- Pause stats (silence regions)
            # librosa.effects.split returns non-silent intervals
            intervals = librosa.effects.split(y, top_db=cfg['top_db'], frame_length=cfg['frame_length'], hop_length=cfg['hop_length'])
            # Convert to silence durations between non-silent intervals
            pauses = []
            last_end = 0
            for start_i, end_i in intervals:
                if start_i > last_end:
                    pauses.append((start_i - last_end) / sr)
                last_end = end_i
            if last_end < len(y):
                pauses.append((len(y) - last_end) / sr)
            pauses = np.array(pauses, dtype=np.float32)
            feats: Dict[str, Any] = {
                "duration_sec": float(dur),
                "rms": _stats(rms),
                "f0_hz": _stats(f0_voiced),
                "voiced_ratio": voiced_ratio,
                "spectral_centroid": _stats(centroid),
                "spectral_bandwidth": _stats(bandwidth),
                "spectral_rolloff": _stats(rolloff),
                "spectral_flatness": _stats(flatness),
                "zcr": _stats(zcr),

                "pause_count": int(pauses.size),
                "pause_total_sec": float(np.sum(pauses)) if pauses.size else 0.0,
                "pause": _stats(pauses),
            }
            # Flatten MFCC stats into fixed keys
            for i in range(cfg['n_mfcc']):
                feats[f"mfcc{i + 1}_mean"] = float(mfcc_mean[i])
                feats[f"mfcc{i + 1}_std"] = float(mfcc_std[i])
            return feats

        def extract_values(d):
            result = []
            for v in d.values():
                if isinstance(v, dict):
                    result.extend(extract_values(v))  # recursive
                else:
                    result.append(v)
            return result

        audio_feature_config = {'sr': 16000, 'n_mfcc': 13, 'frame_length': 1024, 'hop_length': 256, 'top_db': 30}
        subsets = [controller.train_data_path, controller.validation_data_path]
        for subset in subsets:
            participants = os.listdir(subset)
            for participant in participants:
                current_path = os.path.join(subset, participant)
                transcripts_path = [str(p) for p in Path(current_path).glob("*.json")]
                general_info_path = os.path.join(current_path, 'general_info.json')
                transcripts_path.remove(str(general_info_path))

                for transcript_path in tqdm(transcripts_path):
                    try:
                        with open(transcript_path, "r", encoding="utf-8") as f1:
                            data = json.load(f1)
                    except Exception as e:
                        print('>>>>>>>>>>>>>>>>>>>>>>', transcript_path)
                        continue

                    change_happened = False
                    sources = ['moderator_transcript', 'participant_transcript']
                    for source in sources:
                        transcript = data[source]
                        transcript_keys = list(transcript.keys())
                        for key in transcript_keys:
                            wav_path = os.path.join(str(transcript_path).replace('.json', ''), key + '.wav')
                            if os.path.isfile(wav_path) is not True:
                                continue
                            if 'prosody_audio' in data[source][key]:
                                continue
                            change_happened = True
                            try:
                                features = extract_sentence_features(wav_path, audio_feature_config)
                                features = extract_values(features)
                                data[source][key]['prosody_audio'] = features
                            except Exception as e:
                                data[source][key]['prosody_audio'] = [0.0]*86
                                print('Could not extract prosody for source', source, 'key', key, 'in', transcript_path)

                    if change_happened:
                        with open(transcript_path, "w") as f2:
                            json.dump(data, f2, indent=4)

    def extract_audio_xvector(self):
        xvector_classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models")

        subsets = [controller.train_data_path, controller.validation_data_path]
        for subset in subsets:
            participants = os.listdir(subset)
            for participant in participants:
                current_path = os.path.join(subset, participant)
                transcripts_path = [str(p) for p in Path(current_path).glob("*.json")]
                general_info_path = os.path.join(current_path, 'general_info.json')
                transcripts_path.remove(str(general_info_path))

                for transcript_path in tqdm(transcripts_path):
                    try:
                        with open(transcript_path, "r", encoding="utf-8") as f1:
                            data = json.load(f1)
                    except Exception as e:
                        print('>>>>>>>>>>>>>>>>>>>>>>', transcript_path)
                        continue

                    change_happened = False
                    sources = ['moderator_transcript', 'participant_transcript']
                    for source in sources:
                        transcript = data[source]
                        transcript_keys = list(transcript.keys())
                        for key in transcript_keys:
                            wav_path = os.path.join(str(transcript_path).replace('.json', ''), key + '.wav')
                            if os.path.isfile(wav_path) is not True:
                                continue
                            if 'xvector_audio' in data[source][key]:
                                continue
                            change_happened = True
                            try:
                                signal, sample_rate = torchaudio.load(wav_path)
                                xvector_embeddings = xvector_classifier.encode_batch(signal)
                                xvector_embeddings = xvector_embeddings[0][0][:]
                                data[source][key]['xvector_audio'] = xvector_embeddings.tolist()
                            except Exception as e:
                                data[source][key]['xvector_audio'] = [0.0]*512
                                print('Could not extract xvector for source', source, 'key', key, 'in', transcript_path)

                    if change_happened:
                        with open(transcript_path, "w") as f2:
                            json.dump(data, f2, indent=4)

    def extract_audio_wavlm(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "microsoft/wavlm-base-plus-sv"
        _WAVLM_FE = AutoFeatureExtractor.from_pretrained(model_name)
        # IMPORTANT: use the correct head for "-sv" checkpoints
        _WAVLM_MODEL = AutoModelForAudioXVector.from_pretrained(model_name).to(device)
        _WAVLM_MODEL.eval()

        def extract_wavlm_sliding_mean(wav, sr, max_window_s=10.0, min_window_s=0.4, max_windows=None):
            hop_s = 0.75 * max_window_s
            max_win = int(max_window_s * sr)
            min_len = int(min_window_s * sr)
            hop = int(hop_s * sr)
            n = wav.numel()

            def forward_chunk(chunk_1d: torch.Tensor) -> np.ndarray:
                inputs = _WAVLM_FE(chunk_1d.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = _WAVLM_MODEL(**inputs)
                    # Fixed-size speaker embedding
                    emb = out.embeddings.squeeze(0).detach().float().cpu().numpy()
                emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
                return emb

            # If shorter than a window: do one pass
            if n <= max_win:
                if n <= min_len: # if it is too short pad it
                    pad_len = min_len - wav.numel()
                    wav = torch.nn.functional.pad(wav, (math.floor(pad_len/2), math.ceil(pad_len/2)))
                return forward_chunk(wav).astype(np.float32)

            # Sliding windows
            embs = []
            for i, start in enumerate(range(0, n - max_win + 1, hop)):
                if max_windows is not None and i >= max_windows:
                    break
                chunk = wav[start:start + max_win]
                embs.append(forward_chunk(chunk))

            # Average over all window embeddings (what you asked for)
            emb_mean = np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)
            return emb_mean

        subsets = [controller.train_data_path, controller.validation_data_path]
        for subset in subsets:
            participants = os.listdir(subset)
            for participant in participants:
                current_path = os.path.join(subset, participant)
                transcripts_path = [str(p) for p in Path(current_path).glob("*.json")]
                general_info_path = os.path.join(current_path, "general_info.json")
                if str(general_info_path) in transcripts_path:
                    transcripts_path.remove(str(general_info_path))
                for transcript_path in tqdm(transcripts_path):
                    try:
                        with open(transcript_path, "r", encoding="utf-8") as f1:
                            data = json.load(f1)
                    except Exception:
                        print(">>>>>>>>>>>>>>>>>>>>>>", transcript_path)
                        continue

                    change_happened = False
                    sources = ["moderator_transcript", "participant_transcript"]
                    for source in sources:
                        if source not in data:
                            continue
                        transcript = data[source]
                        transcript_keys = list(transcript.keys())

                        for key in transcript_keys:
                            wav_path = os.path.join(str(transcript_path).replace(".json", ""), key + ".wav")
                            if not os.path.isfile(wav_path):
                                continue
                            if "wavlm_audio" in data[source][key] and np.isnan(data[source][key]['wavlm_audio']).sum()==0:
                                continue

                            change_happened = True
                            # load wav
                            wav, sr = torchaudio.load(wav_path)  # [C, N]

                            # mono
                            wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav.squeeze(0)

                            # resample to 16k
                            if sr != 16000:
                                wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
                                sr = 16000

                            # extract embedding
                            try:
                                wavlm_embeddings = extract_wavlm_sliding_mean(wav, sr, max_window_s=10.0, min_window_s=0.4)
                                data[source][key]["wavlm_audio"] = wavlm_embeddings.tolist()
                            except Exception as e:
                                print("Could not extract wavlm for source", source, "key", key, "in", transcript_path)
                                print(e)

                    if change_happened:
                        with open(transcript_path, "w", encoding="utf-8") as f2:
                            json.dump(data, f2, indent=4)



