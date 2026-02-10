import json
import os
import re

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import nltk
from collections import Counter
import spacy
import math
from openai import OpenAI
from lexicalrichness import LexicalRichness
from control_panel import ControlPanel

controller = ControlPanel()

class TextHandler:
    def correct_transcripts_order(self):
        def regenerate_transcript(transcripts, transcript_keys, upgrade_or_downgrade):
            new_transcript = dict()
            value = 1 if upgrade_or_downgrade == 'upgrade' else -1 if upgrade_or_downgrade == 'downgrade' else 0
            for transcript_key in transcript_keys:
                new_transcript_key = str(int(transcript_key)+value)
                new_transcript[new_transcript_key] = transcripts[transcript_key]
            return new_transcript

        train_set = controller.train_data_path
        validation_set = controller.validation_data_path
        subsets = [train_set, validation_set]

        for subset in subsets:
            participants = os.listdir(subset)
            for participant in participants:
                current_path = os.path.join(subset, participant)
                transcripts_path = [str(p) for p in Path(current_path).glob("*.json")]
                general_info_path = os.path.join(current_path, 'general_info.json')
                transcripts_path.remove(str(general_info_path))

                for transcript_path in transcripts_path:
                    with open(transcript_path, "r", encoding="utf-8") as f2:
                        data = json.load(f2)

                    participant_transcripts = data['participant_transcript']
                    moderator_transcripts = data['moderator_transcript']
                    participant_transcript_keys = list(participant_transcripts.keys())
                    moderator_transcript_keys = list(moderator_transcripts.keys())

                    if int(moderator_transcript_keys[-1]) > int(participant_transcript_keys[-1]):  # moderator cannot finish chat
                        del moderator_transcripts[moderator_transcript_keys[-1]]
                        del moderator_transcript_keys[-1]

                    if participant_transcript_keys[0] == '0': # participant cannot start chat
                        del participant_transcripts[participant_transcript_keys[0]]
                        del participant_transcript_keys[0]
                        participant_transcripts = regenerate_transcript(participant_transcripts, participant_transcript_keys,
                                                                            upgrade_or_downgrade='downgrade')
                        moderator_transcripts = regenerate_transcript(moderator_transcripts, moderator_transcript_keys,
                                                                            upgrade_or_downgrade='downgrade')

                    data['participant_transcript'] = participant_transcripts
                    data['moderator_transcript'] = moderator_transcripts

                    with open(transcript_path, 'w') as f3:
                        json.dump(data, f3, indent=4)

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

    def extract_text_stylometrics(self):
        def lexical_richness_calculator(text):
            lex = LexicalRichness(text.lower())
            N = lex.words  # <-- this is an int (token count) in your version
            ttr = lex.ttr if N > 0 else 0.0
            mtld = math.log10(lex.mtld() + 1) if N >= 10 else 0.0
            hdd = lex.hdd(draws=min(42, N - 1)) if N >= 2 else 0.0
            return {"ttr": ttr, "mtld": mtld, "hdd": hdd}

        def pronoun_ratios_calculator(text, nlp_model, include_raw_counts=True):
            pronoun_groups = {
                "first_singular": {"i", "me", "my", "mine", "myself"},
                "first_plural": {"we", "us", "our", "ours", "ourselves"},
                "second": {"you", "your", "yours", "yourself", "yourselves"},
                "third_singular": {"he", "him", "his", "himself", "she", "her", "hers", "herself",
                                   "they", "them", "their", "theirs", "themself", "themselves"},
                "third_plural": {"they", "them", "their", "theirs", "themselves"},
                "demonstrative": {"this", "that", "these", "those"},
                "indefinite": {"someone", "somebody", "something", "anyone", "anybody", "anything", "everyone",
                               "everybody", "everything", "noone", "nobody", "nothing", "none"}
            }
            doc = nlp_model(text)
            total_nouns = sum(1 for tok in doc if tok.pos_ in {"NOUN", "PROPN"} and tok.is_alpha)
            pron_tokens = []
            for tok in doc:
                if tok.pos_ == "PRON" and tok.is_alpha:
                    pron_tokens.append(tok.text.lower())

            total_pronouns = len(pron_tokens)
            group_counts = {g: 0 for g in pronoun_groups.keys()}
            for p in pron_tokens:
                for g, vocab in pronoun_groups.items():
                    if p in vocab:
                        group_counts[g] += 1

            ratios = {f"pron_ratio_{g}": (c / max(1, total_pronouns)) for g, c in group_counts.items()}
            pron_to_noun = total_pronouns / max(1, total_nouns)
            out = {**ratios, "total_pronouns": total_pronouns, "total_nouns": total_nouns, "pronoun_to_noun_ratio": pron_to_noun}
            if include_raw_counts:
                out.update({f"pron_count_{g}": c for g, c in group_counts.items()})
            return out

        def pos_distribution_calculator(text, nlp_model):
            ALL_POS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
                       "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
            doc = nlp_model(text)
            pos_counts = Counter(tok.pos_ for tok in doc if tok.is_alpha)
            total = sum(pos_counts.values())
            return {
                f"pos_{pos}": (pos_counts.get(pos, 0) / total if total > 0 else 0.0)
                for pos in ALL_POS
            }
        def function_content_ratio_calculator(text, nlp_model):
            CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
            FUNCTION_POS = {"PRON", "DET", "ADP", "AUX", "CCONJ", "SCONJ"}
            doc = nlp_model(text)
            content = sum(1 for tok in doc if tok.pos_ in CONTENT_POS)
            function = sum(1 for tok in doc if tok.pos_ in FUNCTION_POS)
            return {
                "content_ratio": content / max(1, content + function),
                "function_ratio": function / max(1, content + function)
            }

        def disfluency_features_calculator(text):
            FILLERS = {"uh", "um", "uhm", "er", "erm", "hmm", "mm", "mmm", "ah", "eh", "oh", "uh-huh", "mm-hmm"}
            tokens = text.lower().split()
            filler_count = sum(1 for t in tokens if t in FILLERS)
            repetitions = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1])

            return {
                "filler_ratio": filler_count / max(1, len(tokens)),
                "repetition_ratio": repetitions / max(1, len(tokens))
            }

        def sentence_complexity_calculator(text, nlp_model):
            doc = nlp_model(text)
            sentences = list(doc.sents)
            sent_lengths = [len([t for t in sent if t.is_alpha]) for sent in sentences]
            depths = [max(len(list(tok.ancestors)) for tok in sent) for sent in sentences]

            return {
                "avg_sentence_length": float(math.log1p(np.mean(sent_lengths))) if sent_lengths else 0,
                "avg_clause_depth": float(math.log1p(np.mean(depths))) if depths else 0
            }

        def tempo_pause_features_calculator(text):
            """
            Extract prosody/fluency features from a single question's transcript string.
            Handles <TEMPO_X> with optional following <PAUSE_Y>.
            """
            TEMPO_LEVELS = ["SLOW", "MED", "FAST"]
            PAUSE_LEVELS = ["NONE", "SHORT", "MED", "LONG", "VLONG"]

            TEMPO_SCORE = {"SLOW": 0, "MED": 1, "FAST": 2}
            PAUSE_SCORE = {"NONE": 0, "SHORT": 1, "MED": 2, "LONG": 3, "VLONG": 4}

            # Matches patterns like: <TEMPO_MED><PAUSE_SHORT> or <TEMPO_FAST> (pause optional)
            TP_PATTERN = re.compile(r"<TEMPO_(SLOW|MED|FAST)>(?:<PAUSE_(NONE|SHORT|MED|LONG|VLONG)>)?")

            def _safe_mean(xs):
                return sum(xs) / len(xs) if xs else 0.0

            def _safe_std(xs):
                if len(xs) < 2:
                    return 0.0
                mu = _safe_mean(xs)
                return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))

            def _switch_rate(seq):
                if len(seq) < 2:
                    return 0.0
                switches = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
                return switches / (len(seq) - 1)

            def _runs(seq):
                """Return run lengths for consecutive identical items."""
                if not seq:
                    return []
                run_lengths = []
                cur = 1
                for i in range(1, len(seq)):
                    if seq[i] == seq[i - 1]:
                        cur += 1
                    else:
                        run_lengths.append(cur)
                        cur = 1
                run_lengths.append(cur)
                return run_lengths

            def _transition_probs(seq, levels):
                """Flattened transition probs over levels x levels."""
                out = {}
                if len(seq) < 2:
                    # all zeros
                    for a in levels:
                        for b in levels:
                            out[f"trans_{a}_to_{b}"] = 0.0
                    return out

                counts = Counter()
                total = 0
                for i in range(1, len(seq)):
                    counts[(seq[i - 1], seq[i])] += 1
                    total += 1

                for a in levels:
                    for b in levels:
                        out[f"trans_{a}_to_{b}"] = counts[(a, b)] / total
                return out

            matches = TP_PATTERN.findall(text)
            tempos = [t for (t, p) in matches if t]
            pauses = [p for (t, p) in matches if p]  # pause is optional; last tempo may have no pause

            # Distributions
            tempo_counts = Counter(tempos)
            pause_counts = Counter(pauses)

            n_segments = len(tempos)
            n_pauses = len(pauses)

            feats = {
                "n_segments": n_segments,
                "n_pauses": n_pauses,
            }

            # Ratios (normalize by totals)
            for lvl in TEMPO_LEVELS:
                feats[f"tempo_ratio_{lvl}"] = tempo_counts[lvl] / max(1, n_segments)
            for lvl in PAUSE_LEVELS:
                feats[f"pause_ratio_{lvl}"] = pause_counts[lvl] / max(1, n_pauses)

            # Numeric summaries
            tempo_vals = [TEMPO_SCORE[t] for t in tempos]
            pause_vals = [PAUSE_SCORE[p] for p in pauses]

            feats.update({
                "tempo_mean": _safe_mean(tempo_vals),
                "tempo_std": _safe_std(tempo_vals),
                "pause_mean": _safe_mean(pause_vals),
                "pause_std": _safe_std(pause_vals),
                "tempo_max": max(tempo_vals) if tempo_vals else 0.0,
                "pause_max": max(pause_vals) if pause_vals else 0.0,
                "tempo_switch_rate": _switch_rate(tempos),
                "pause_switch_rate": _switch_rate(pauses),
            })

            # Run-length dynamics
            tempo_runs = _runs(tempos)
            pause_runs = _runs(pauses)
            feats.update({
                "tempo_run_mean": _safe_mean(tempo_runs),
                "tempo_run_max": max(tempo_runs) if tempo_runs else 0.0,
                "pause_run_mean": _safe_mean(pause_runs),
                "pause_run_max": max(pause_runs) if pause_runs else 0.0,
            })

            # Transition probabilities (compact but powerful)
            feats.update({f"tempo_{k}": v for k, v in _transition_probs(tempos, TEMPO_LEVELS).items()})
            feats.update({f"pause_{k}": v for k, v in _transition_probs(pauses, PAUSE_LEVELS).items()})

            return feats

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
                        sentence_with_tempo_pause = participant_transcript[participant_key]['text']
                        sentence = re.sub(r"<(?:TEMPO|PAUSE)_[A-Z]+>", "", sentence_with_tempo_pause)

                        feature_vector = []
                        nlp = spacy.load("en_core_web_sm", disable=["ner"])
                        
                        lexical_richness = lexical_richness_calculator(sentence)
                        feature_vector.append(lexical_richness['ttr'])
                        feature_vector.append(lexical_richness['mtld'])
                        feature_vector.append(lexical_richness['hdd'])

                        pronoun_ratios = pronoun_ratios_calculator(sentence, nlp)
                        feature_vector.append(pronoun_ratios['pron_ratio_first_singular'])
                        feature_vector.append(pronoun_ratios['pron_ratio_first_plural'])
                        feature_vector.append(pronoun_ratios['pron_ratio_second'])
                        feature_vector.append(pronoun_ratios['pron_ratio_third_singular'])
                        feature_vector.append(pronoun_ratios['pron_ratio_third_plural'])
                        feature_vector.append(pronoun_ratios['pron_ratio_demonstrative'])
                        feature_vector.append(pronoun_ratios['pron_ratio_indefinite'])
                        feature_vector.append(pronoun_ratios['pronoun_to_noun_ratio'])

                        pos_distribution = pos_distribution_calculator(sentence, nlp)
                        feature_vector.append(pos_distribution['pos_ADJ'])
                        feature_vector.append(pos_distribution['pos_ADP'])
                        feature_vector.append(pos_distribution['pos_ADV'])
                        feature_vector.append(pos_distribution['pos_AUX'])
                        feature_vector.append(pos_distribution['pos_CCONJ'])
                        feature_vector.append(pos_distribution['pos_DET'])
                        feature_vector.append(pos_distribution['pos_INTJ'])
                        feature_vector.append(pos_distribution['pos_NOUN'])
                        feature_vector.append(pos_distribution['pos_NUM'])
                        feature_vector.append(pos_distribution['pos_PART'])
                        feature_vector.append(pos_distribution['pos_PRON'])
                        feature_vector.append(pos_distribution['pos_PROPN'])
                        feature_vector.append(pos_distribution['pos_PUNCT'])
                        feature_vector.append(pos_distribution['pos_SCONJ'])
                        feature_vector.append(pos_distribution['pos_SYM'])
                        feature_vector.append(pos_distribution['pos_VERB'])
                        feature_vector.append(pos_distribution['pos_X'])
                        feature_vector.append(pos_distribution['pos_SPACE'])

                        function_content_ratio = function_content_ratio_calculator(sentence, nlp)
                        feature_vector.append(function_content_ratio['function_ratio'])
                        feature_vector.append(function_content_ratio['content_ratio'])

                        disfluency_features = disfluency_features_calculator(sentence)
                        feature_vector.append(disfluency_features['filler_ratio'])
                        feature_vector.append(disfluency_features['repetition_ratio'])

                        sentence_complexity = sentence_complexity_calculator(sentence, nlp)
                        feature_vector.append(sentence_complexity['avg_sentence_length'])
                        feature_vector.append(sentence_complexity['avg_clause_depth'])

                        tempo_pause_features = tempo_pause_features_calculator(sentence_with_tempo_pause)
                        feature_vector.append(tempo_pause_features['n_segments'])
                        feature_vector.append(tempo_pause_features['n_pauses'])
                        feature_vector.append(tempo_pause_features['tempo_ratio_SLOW'])
                        feature_vector.append(tempo_pause_features['tempo_ratio_MED'])
                        feature_vector.append(tempo_pause_features['tempo_ratio_FAST'])
                        feature_vector.append(tempo_pause_features['pause_ratio_NONE'])
                        feature_vector.append(tempo_pause_features['pause_ratio_SHORT'])
                        feature_vector.append(tempo_pause_features['pause_ratio_MED'])
                        feature_vector.append(tempo_pause_features['pause_ratio_LONG'])
                        feature_vector.append(tempo_pause_features['pause_ratio_VLONG'])
                        feature_vector.append(tempo_pause_features['tempo_mean'])
                        feature_vector.append(tempo_pause_features['tempo_std'])
                        feature_vector.append(tempo_pause_features['pause_mean'])
                        feature_vector.append(tempo_pause_features['pause_std'])
                        feature_vector.append(tempo_pause_features['tempo_switch_rate'])
                        feature_vector.append(tempo_pause_features['pause_switch_rate'])
                        feature_vector.append(tempo_pause_features['tempo_run_mean'])
                        feature_vector.append(tempo_pause_features['tempo_run_max'])
                        feature_vector.append(tempo_pause_features['pause_run_mean'])
                        feature_vector.append(tempo_pause_features['pause_run_max'])
                        feature_vector.append(tempo_pause_features['tempo_trans_SLOW_to_SLOW'])
                        feature_vector.append(tempo_pause_features['tempo_trans_SLOW_to_MED'])
                        feature_vector.append(tempo_pause_features['tempo_trans_SLOW_to_FAST'])
                        feature_vector.append(tempo_pause_features['tempo_trans_MED_to_SLOW'])
                        feature_vector.append(tempo_pause_features['tempo_trans_MED_to_MED'])
                        feature_vector.append(tempo_pause_features['tempo_trans_MED_to_FAST'])
                        feature_vector.append(tempo_pause_features['tempo_trans_FAST_to_SLOW'])
                        feature_vector.append(tempo_pause_features['tempo_trans_FAST_to_MED'])
                        feature_vector.append(tempo_pause_features['tempo_trans_FAST_to_FAST'])
                        feature_vector.append(tempo_pause_features['pause_trans_NONE_to_NONE'])
                        feature_vector.append(tempo_pause_features['pause_trans_NONE_to_SHORT'])
                        feature_vector.append(tempo_pause_features['pause_trans_NONE_to_MED'])
                        feature_vector.append(tempo_pause_features['pause_trans_NONE_to_LONG'])
                        feature_vector.append(tempo_pause_features['pause_trans_NONE_to_VLONG'])
                        feature_vector.append(tempo_pause_features['pause_trans_SHORT_to_NONE'])
                        feature_vector.append(tempo_pause_features['pause_trans_SHORT_to_SHORT'])
                        feature_vector.append(tempo_pause_features['pause_trans_SHORT_to_MED'])
                        feature_vector.append(tempo_pause_features['pause_trans_SHORT_to_LONG'])
                        feature_vector.append(tempo_pause_features['pause_trans_SHORT_to_VLONG'])
                        feature_vector.append(tempo_pause_features['pause_trans_MED_to_NONE'])
                        feature_vector.append(tempo_pause_features['pause_trans_MED_to_SHORT'])
                        feature_vector.append(tempo_pause_features['pause_trans_MED_to_MED'])
                        feature_vector.append(tempo_pause_features['pause_trans_MED_to_LONG'])
                        feature_vector.append(tempo_pause_features['pause_trans_MED_to_VLONG'])
                        feature_vector.append(tempo_pause_features['pause_trans_LONG_to_NONE'])
                        feature_vector.append(tempo_pause_features['pause_trans_LONG_to_SHORT'])
                        feature_vector.append(tempo_pause_features['pause_trans_LONG_to_MED'])
                        feature_vector.append(tempo_pause_features['pause_trans_LONG_to_LONG'])
                        feature_vector.append(tempo_pause_features['pause_trans_LONG_to_VLONG'])
                        feature_vector.append(tempo_pause_features['pause_trans_VLONG_to_NONE'])
                        feature_vector.append(tempo_pause_features['pause_trans_VLONG_to_SHORT'])
                        feature_vector.append(tempo_pause_features['pause_trans_VLONG_to_MED'])
                        feature_vector.append(tempo_pause_features['pause_trans_VLONG_to_LONG'])
                        feature_vector.append(tempo_pause_features['pause_trans_VLONG_to_VLONG'])

                        there_is_non_expected_value = any(not isinstance(x, (int, float)) for x in feature_vector)
                        if there_is_non_expected_value:
                            print('text_stylometrics error: non expected value for participant_key', participant_key, 'in', transcript_path)
                        else:
                            data['participant_transcript'][participant_key]['stylometric_text'] = feature_vector
                    with open(transcript_path, 'w') as f2:
                        json.dump(data, f2, indent=4)


