import json
import os
import re
import random

import numpy as np
from openai import OpenAI
from typing import List
from pathlib import Path
from sentence_transformers import SentenceTransformer
from control_panel import ControlPanel

controller = ControlPanel()

class TextHandler:
    def extract_text_embeddings(self):
        def extract_sub_sentences(text):
            pattern = re.compile(r"\s*(.*?)\s*((?:<[^>]+>\s*)+|$)")
            results = []
            for m in pattern.finditer(text):
                chunk = m.group(1).strip()
                tags = m.group(2) or ""
                if not chunk:
                    continue
                tempo = re.search(r"<TEMPO_(\w+)>", tags)
                pause = re.search(r"<PAUSE_(\w+)>", tags)
                results.append({"text": chunk, "tempo": tempo.group(1) if tempo else None, "pause": pause.group(1) if pause else None})
            return results

        model = SentenceTransformer("all-mpnet-base-v2")
        subsets = [controller.train_data_path, controller.validation_data_path]
        for subset in subsets:
            participants = os.listdir(subset)
            for participant in participants:
                current_path = os.path.join(subset, participant)
                transcripts_path = [str(p) for p in Path(current_path).glob("*.json")]
                general_info_path = os.path.join(current_path, 'general_info.json')
                transcripts_path.remove(str(general_info_path))

                for transcript_path in transcripts_path:
                    with open(transcript_path, "r", encoding="utf-8") as f1:
                        data = json.load(f1)

                    participant_transcript = data['participant_transcript']
                    participant_transcript_keys = list(participant_transcript.keys())
                    for participant_key in participant_transcript_keys:
                        response = participant_transcript[participant_key]['text']
                        sentences = extract_sub_sentences(response)

                        sentence_list = []
                        answer = ''
                        for sentence in sentences:
                            text = sentence['text']
                            sentence_list.append(text)
                            answer = answer + text + ' '
                        answer = answer[:-1]
                        embeddings = model.encode(sentence_list)
                        embedding_mean = np.mean(embeddings, axis=0)
                        embedding_max = np.max(embeddings, axis=0)
                        embedding_all = model.encode(answer)
                        # feature_vector = np.concatenate([embedding_mean, embedding_max, embedding_all])
                        feature_vector = np.stack([embedding_mean, embedding_max, embedding_all], axis=1)
                        feature_vector = feature_vector.tolist()

                        data['participant_transcript'][participant_key]['embedding_text'] = feature_vector
                    with open(transcript_path, 'w') as f2:
                        json.dump(data, f2, indent=4)



