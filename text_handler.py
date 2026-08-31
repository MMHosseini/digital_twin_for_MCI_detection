import json
import os
import re
from multiprocessing.dummy import current_process
from datetime import datetime
from textwrap import indent

import torch
from sklearn.metrics.cluster import contingency_matrix
from transformers import AutoTokenizer, AutoModel
import numpy as np
from pathlib import Path
import nltk
from tqdm import tqdm
from collections import Counter
import spacy
import math
from openai import OpenAI
import numpy as np
import json
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
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

                for transcript_path in tqdm(transcripts_path):
                    change_happened = False
                    with open(transcript_path, "r", encoding="utf-8") as f1:
                        data = json.load(f1)

                    sources = ['moderator_transcript', 'participant_transcript', 'simulated_transcript_raw_gpt', 'simulated_transcript_sft',
                               'simulated_transcript_sft_rag', 'simulated_transcript_dpo', 'simulated_transcript_dpo_rag']
                    for source in sources:
                        if source not in data:
                            continue

                        transcript = data[source]
                        transcript_keys = list(transcript.keys())
                        for key in transcript_keys:
                            if 'embedding_text' in transcript[key]:
                                continue

                            change_happened = True
                            full_sentence_with_markers = transcript[key]['text']
                            sentences = extract_sub_sentences(full_sentence_with_markers)

                            sentence_list = []
                            full_sentence = ''
                            for sentence in sentences:
                                text = sentence['text']
                                sentence_list.append(text)
                                full_sentence = full_sentence + text + ' '
                            full_sentence = full_sentence[:-1]
                            embeddings = model.encode(sentence_list)
                            embedding_mean = np.mean(embeddings, axis=0)
                            embedding_max = np.max(embeddings, axis=0)
                            embedding_all = model.encode(full_sentence)
                            feature_vector = np.stack([embedding_mean, embedding_max, embedding_all], axis=1)
                            feature_vector = feature_vector.tolist()
                            data[source][key]['embedding_text'] = feature_vector

                    if change_happened:
                        with open(transcript_path, 'w') as f2:
                            json.dump(data, f2, indent=4)

    def extract_text_sentiments(self):
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

        def get_sentiment_vector(model, tokenizer, text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            outputs = model(**inputs)
            sentiment_vector = outputs.last_hidden_state[:, 0, :]
            return sentiment_vector.detach().cpu().numpy().T

        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        subsets = [controller.train_data_path, controller.validation_data_path]
        for subset in subsets:
            participants = os.listdir(subset)
            for participant in participants:
                current_path = os.path.join(subset, participant)
                transcripts_path = [str(p) for p in Path(current_path).glob("*.json")]
                general_info_path = os.path.join(current_path, 'general_info.json')
                transcripts_path.remove(str(general_info_path))

                for transcript_path in tqdm(transcripts_path):
                    change_happened = False
                    with open(transcript_path, "r", encoding="utf-8") as f1:
                        data = json.load(f1)

                    sources = ['moderator_transcript', 'participant_transcript', 'simulated_transcript_raw_gpt', 'simulated_transcript_sft',
                               'simulated_transcript_sft_rag', 'simulated_transcript_dpo', 'simulated_transcript_dpo_rag']
                    for source in sources:
                        if source not in data:
                            continue

                        transcript = data[source]
                        transcript_keys = list(transcript.keys())
                        for key in transcript_keys:
                            if 'sentiment_text' in transcript[key]:
                                continue

                            change_happened = True
                            full_sentence_with_markers = transcript[key]['text']
                            sentences = extract_sub_sentences(full_sentence_with_markers)

                            sentence_list = []
                            full_sentence = ''
                            for sentence in sentences:
                                text = sentence['text']
                                sentence_list.append(text)
                                full_sentence = full_sentence + text + ' '
                            full_sentence = full_sentence[:-1]

                            feature_vectors = []
                            for text in sentence_list:
                                sentiment_vector = get_sentiment_vector(model, tokenizer, text)
                                feature_vectors.append(sentiment_vector)

                            sentiment_mean = np.mean(np.array(feature_vectors), axis=0).squeeze()
                            sentiment_std = np.std(np.array(feature_vectors), axis=0).squeeze()
                            sentiment_all = get_sentiment_vector(model, tokenizer, full_sentence).squeeze()

                            feature_vector = np.stack([sentiment_mean, sentiment_std, sentiment_all], axis=1)
                            feature_vector = feature_vector.tolist()
                            data[source][key]['sentiment_text'] = feature_vector

                    if change_happened:
                        with open(transcript_path, 'w') as f2:
                            json.dump(data, f2, indent=4)

    def extract_title_descriptor(self):
        subsets = [controller.train_data_path, controller.validation_data_path]
        topics_list = []
        transcript_path_list = []
        for subset in subsets:
            participants = os.listdir(subset)
            participants.sort()
            for participant in participants:
                current_path = os.path.join(subset, participant)
                transcripts_path = [str(p) for p in Path(current_path).glob("*.json")]
                general_info_path = os.path.join(current_path, 'general_info.json')
                transcripts_path.remove(str(general_info_path))
                for transcript_path in tqdm(transcripts_path):
                    with open(transcript_path, "r", encoding="utf-8") as f1:
                        data = json.load(f1)
                    topics_list.append(data['topic'])
                    transcript_path_list.append(transcript_path)
        topics = list(set(s.strip().lower() for s in topics_list))

        # model = SentenceTransformer("all-mpnet-base-v2") # 768
        model = SentenceTransformer("all-MiniLM-L6-v2") # 384
        embeddings = model.encode(topics, convert_to_numpy=True)  # (N, d)

        # Standardize (important before PCA)
        scaler = StandardScaler()
        X = scaler.fit_transform(embeddings)

        # Fit full PCA
        max_components = min(X.shape[0], X.shape[1])
        pca_full = PCA(n_components=max_components, svd_solver="full", random_state=42)
        X_pca_full = pca_full.fit_transform(X)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        target = 0.90
        valid = np.where(cumvar <= target)[0]
        k = int(valid[-1] + 1) if len(valid) > 0 else 1
        energy_preserved = float(cumvar[k - 1])
        print(f"Selected k: {k}")
        print(f"Energy preserved: {energy_preserved:.4f} ({energy_preserved * 100:.2f}%)")

        # Keep first k components
        topic_vectors = X_pca_full[:, :k]

        # Optional normalization (recommended)
        topic_vectors = normalize(topic_vectors, norm="l2")
        print("Final descriptor shape:", topic_vectors.shape)

        # Save scaler + PCA (IMPORTANT for future topics)
        # joblib.dump(scaler, "topic_scaler.pkl")
        # joblib.dump(pca_full, "topic_pca_full.pkl")

        # Save descriptors
        for i, topic in enumerate(tqdm(topics)):
            indices = [j for j, v in enumerate(topics_list) if v.strip().lower() == topic]
            for index in indices:
                transcript_path = transcript_path_list[index]
                with open(transcript_path, 'r') as f2:
                    data = json.load(f2)
                data['topic_descriptor'] = topic_vectors[i, :].tolist()
                with open(transcript_path, 'w') as f3:
                    json.dump(data, f3, indent=4)

    def prepare_rag_data(self):
        def read_participant_data(transcripts_path):
            order_of_interviews = []
            dates = []
            questions = []
            question_vectors = []
            answers= []
            answer_vectors = []
            jsons_path = []
            keys = []


            for transcript_path in transcripts_path:
                with open(transcript_path, "r", encoding="utf-8") as f1:
                    data = json.load(f1)

                order_of_interview = data['order_of_interview']
                date = datetime.strptime(data['date'], "%Y-%m-%d").date()
                moderator_transcript = data['moderator_transcript']
                participant_transcript = data['participant_transcript']
                moderator_transcript_keys = moderator_transcript.keys()

                for moderator_key in moderator_transcript_keys:
                    participant_key = str(int(moderator_key) + 1)
                    question = moderator_transcript[moderator_key]['text']
                    question_vector = np.array(moderator_transcript[moderator_key]['embedding_text'])[:, :].reshape(-1, order="F")
                    answer = participant_transcript[participant_key]['text']
                    answer_vector = np.array(participant_transcript[participant_key]['embedding_text'])[:, :].reshape(-1, order="F")

                    order_of_interviews.append(order_of_interview)
                    dates.append(date)
                    questions.append(question)
                    question_vectors.append(question_vector)
                    answers.append(answer)
                    answer_vectors.append(answer_vector)
                    jsons_path.append(transcript_path)
                    keys.append(participant_key)

            dates = np.array(dates)
            questions = np.array(questions)
            question_vectors = np.array(question_vectors).T
            answers = np.array(answers)
            jsons_path = np.array(jsons_path)
            keys = np.array(keys)

            data = {"dates": dates, "questions": questions, "question_vectors": question_vectors,
                    "answers": answers, 'jsons_path': jsons_path, 'keys': keys}
            return data

        def find_top_k_similar(source, targets, k=20):
            dists = np.linalg.norm(targets - source[:, None], axis=0)  # (100,)
            top_indices = np.argpartition(dists, k - 1)[:k]
            top_indices = top_indices[np.argsort(dists[top_indices])]
            return top_indices

        def determine_closest_dates(target_date, dates):
            target_date = np.array(target_date, dtype="datetime64[D]")
            dates = np.array(dates, dtype="datetime64[D]")
            diffs = np.abs(dates - target_date)
            order = np.argsort(diffs)
            return order.tolist()

        train_path = controller.train_data_path
        validation_path = controller.validation_data_path
        participants = os.listdir(train_path)

        for participant in participants:
            current_train_path = os.path.join(train_path, participant)
            train_transcripts_path = [str(p) for p in Path(current_train_path).glob("*.json")]
            train_general_info_path = os.path.join(current_train_path, 'general_info.json')
            train_transcripts_path.remove(str(train_general_info_path))
            train_data = read_participant_data(train_transcripts_path)

            current_validation_path = os.path.join(validation_path, participant)
            validation_transcripts_path = [str(p) for p in Path(current_validation_path).glob("*.json")]
            validation_general_info_path = os.path.join(current_validation_path, 'general_info.json')
            validation_transcripts_path.remove(str(validation_general_info_path))
            validation_data = read_participant_data(validation_transcripts_path)

            train_questions = train_data['questions']
            train_answers = train_data['answers']
            train_question_vectors = train_data['question_vectors']
            train_dates = train_data['dates']

            validation_question_vectors = validation_data['question_vectors']
            validation_dates = validation_data['dates']
            validation_jsons_path = validation_data['jsons_path']
            validation_participant_keys = validation_data['keys']

            selected_questions_list = []
            selected_answers_list = []
            for i in range(validation_question_vectors.shape[-1]):
                validation_question_vector = validation_question_vectors[:, i]
                validation_date = validation_dates[i]

                similar_indices = find_top_k_similar(validation_question_vector, train_question_vectors, k=20)
                selected_dates = train_dates[similar_indices]
                order_of_dates = determine_closest_dates(validation_date, selected_dates)
                selected_indices = similar_indices[order_of_dates[0:5]]

                selected_questions = list(train_questions[selected_indices])
                selected_answers = list(train_answers[selected_indices])

                selected_questions_list.append(selected_questions)
                selected_answers_list.append(selected_answers)

            validation_jsons_path = list(validation_jsons_path)
            validation_participant_keys = list(validation_participant_keys)

            while len(validation_jsons_path) > 0:
                current_json_path = validation_jsons_path[0]
                indices = [m for m, v in enumerate(validation_jsons_path) if v == current_json_path]

                with open(current_json_path, 'r') as f3:
                    data = json.load(f3)

                for index in indices:
                    key = str(validation_participant_keys[index])
                    questions = [str(x) for x in selected_questions_list[index]]
                    answers = [str(x) for x in selected_answers_list[index]]
                    data['participant_transcript'][key]['rag_samples'] = {'questions': questions, 'answers': answers}

                with open(current_json_path, 'w') as f4:
                    json.dump(data, f4, indent=4)

                for idx in sorted(indices, reverse=True):
                    del validation_jsons_path[idx]
                    del validation_participant_keys[idx]















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

                    sources = ['moderator_transcript', 'participant_transcript', 'simulated_transcript_raw_gpt', 'simulated_transcript_sft',
                               'simulated_transcript_sft_rag', 'simulated_transcript_dpo', 'simulated_transcript_dpo_rag']
                    for source in sources:
                        if source not in data:
                            continue
                        
                        transcript = data[source]
                        transcript_keys = list(transcript.keys())
                        for key in transcript_keys:
                            if 'stylometric_text' in transcript[key]:
                                continue
                            
                            sentence_with_tempo_pause = transcript[key]['text']
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
                                print('text_stylometrics error: non expected value for source',
                                      source, 'participant_key', key, 'in', transcript_path)
                            else:
                                data[source][key]['stylometric_text'] = feature_vector
    
                    with open(transcript_path, 'w') as f2:
                        json.dump(data, f2, indent=4)

