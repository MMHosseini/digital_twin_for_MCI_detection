import os
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
from typing import Any, Dict, Tuple, Union, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from control_panel import ControlPanel

controller = ControlPanel()

class Evaluator:
    def text_based_author_identification(self):
        feature_types = ['sentiment_text', 'embedding_text', 'all_text']
        response_sources = ['participant_transcript', 'simulated_transcript_raw_gpt', 'simulated_transcript_sft', 'simulated_transcript_dpo']
        os.makedirs(controller.result_path, exist_ok=True)
        save_path = os.path.join(controller.result_path, 'text_based_author_identification.txt')

        feat_dim_list = [0, 1, 2, range(0,2)]
        for feature_type in feature_types:
            for feat_dim in feat_dim_list:
                X_train, y_train = self.load_data(controller.train_data_path, feature_type, response_source='participant_transcript', feature_dim=feat_dim)
                model, le = self.train_linear_svm_l2(X_train, y_train, C=1.0)

                for response_source in response_sources:
                    print('Evaluating', feature_type, 'feature on', response_source, 'data')
                    X_val, y_val = self.load_data(controller.validation_data_path, feature_type, response_source=response_source, feature_dim=feat_dim)
                    res = self.evaluate_linear_svm_l2(model, le, X_val, y_val)

                    val1 = res["macro_f1"]
                    val2 = res["labels"]
                    val3 = res["confusion_matrix"]
                    val4 = res["classification_report"]

                    print("Feature:", feature_type)
                    print("Dimension", feat_dim)
                    print("Response source:", response_source)
                    print("Macro-F1:", val1)
                    print("Labels:", val2)
                    print("Confusion matrix:\n", val3)
                    print("Classification report:\n", val4)
                    print("============================================\n")
                    print("============================================\n")

                    with open(save_path, "a") as f:
                        f.write(f"Response source: {response_source}\n")
                        f.write(f"Feature: {feature_type}\n")
                        f.write(f"Feature Dimension: {feat_dim}\n")
                        f.write(f"Macro-F1: {val1}\n")
                        f.write(f"Labels: {val2}\n")
                        f.write(f"Confusion matrix:\n {val3}\n")
                        f.write(f"Classification report:\n {val4}\n")
                        f.write("============================================\n")
                        f.write("============================================\n")

    def audio_based_author_identification(self):
        feature_types = ['prosody_audio', 'wavlm_audio', 'all_audio']
        # response_sources = ['participant_transcript', 'simulated_transcript_raw_gpt', 'simulated_transcript_sft']
        response_sources = ['participant_transcript', 'simulated_transcript_sft']
        os.makedirs(controller.result_path, exist_ok=True)
        save_path = os.path.join(controller.result_path, 'audio_based_author_identification.txt')

        for feature_type in feature_types:
            X_train, y_train = self.load_data(controller.train_data_path, feature_type, response_source='participant_transcript', feature_dim=None)
            model, le = self.train_linear_svm_l2(X_train, y_train, C=1.0)

            for response_source in response_sources:
                print('Evaluating', feature_type, 'feature on', response_source, 'data')
                X_val, y_val = self.load_data(controller.validation_data_path, feature_type, response_source=response_source, feature_dim=None)
                res = self.evaluate_linear_svm_l2(model, le, X_val, y_val)

                val1 = res["macro_f1"]
                val2 = res["labels"]
                val3 = res["confusion_matrix"]
                val4 = res["classification_report"]

                print("Feature:", feature_type)
                print("Response source:", response_source)
                print("Macro-F1:", val1)
                print("Labels:", val2)
                print("Confusion matrix:\n", val3)
                print("Classification report:\n", val4)
                print("============================================\n")
                print("============================================\n")

                with open(save_path, "a") as f:
                    f.write(f"Response source: {response_source}\n")
                    f.write(f"Feature: {feature_type}\n")
                    f.write(f"Macro-F1: {val1}\n")
                    f.write(f"Labels: {val2}\n")
                    f.write(f"Confusion matrix:\n {val3}\n")
                    f.write(f"Classification report:\n {val4}\n")
                    f.write("============================================\n")
                    f.write("============================================\n")

    def train_linear_svm_l2(self, X_train, y_train, C=1.0):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)

        model = Pipeline([
            # ("pca", PCA(n_components=0.95, random_state=42)),
            ("l2norm", Normalizer(norm="l2")),
            ("svm", LinearSVC(C=C, class_weight="balanced", random_state=42)),
        ])

        model.fit(X_train, y_train_enc)
        return model, le

    def evaluate_linear_svm_l2(self, model, le, X_val, y_val):
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)

        y_val_enc = le.transform(y_val)
        y_pred_enc = model.predict(X_val)

        macro_f1 = f1_score(y_val_enc, y_pred_enc, average="macro")
        labels_enc = np.arange(len(le.classes_))
        cm = confusion_matrix(y_val_enc, y_pred_enc, labels=labels_enc)

        report = classification_report(y_val_enc, y_pred_enc, labels=labels_enc, target_names=[str(c) for c in le.classes_], digits=4)
        y_pred_labels = le.inverse_transform(y_pred_enc)

        info = {
            "macro_f1": float(macro_f1),
            "confusion_matrix": cm,
            "labels": le.classes_,
            "classification_report": report,
            "y_pred_labels": y_pred_labels,
        }
        return info

    def load_data(self, path, feature_type, response_source='participant_transcript', feature_dim=None):
        X = []
        y = []
        participants = sorted(os.listdir(path))
        for participant in participants:
            label = participants.index(participant)

            current_path = os.path.join(path, participant)
            transcripts_path = [str(p) for p in Path(current_path).glob("*.json")]
            general_info_path = os.path.join(current_path, 'general_info.json')
            transcripts_path.remove(str(general_info_path))
            for transcript_path in tqdm(transcripts_path):
                with open(transcript_path, 'r') as f:
                    data = json.load(f)
                transcript = data[response_source]
                key_list = list(transcript.keys())
                for key in key_list:
                    try:
                        feature = None
                        if feature_type == 'embedding_text':
                            embedding = np.array(transcript[key]['embedding_text'])[:, feature_dim]
                            embedding = embedding.reshape(-1, order="F")
                            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                            feature = embedding
                        elif feature_type == 'sentiment_text':
                            sentiment = np.array(transcript[key]['sentiment_text'])[:, feature_dim]
                            sentiment = sentiment.reshape(-1, order="F")
                            sentiment = sentiment / (np.linalg.norm(sentiment) + 1e-8)
                            feature = sentiment
                        elif feature_type == 'prosody_audio':
                            prosody = np.array(transcript[key]['prosody_audio'])
                            prosody = prosody / (np.linalg.norm(prosody) + 1e-8)
                            feature = prosody
                        elif feature_type == 'xvector_audio':
                            xvector = np.array(transcript[key]['xvector_audio'])
                            xvector = xvector / (np.linalg.norm(xvector) + 1e-8)
                            feature = xvector
                        elif feature_type == 'wavlm_audio':
                            wavlm = np.array(transcript[key]['wavlm_audio'])
                            wavlm = wavlm / (np.linalg.norm(wavlm) + 1e-8)
                            feature = wavlm
                        elif feature_type == 'all_text':
                            embedding = np.array(transcript[key]['embedding_text'])[:, feature_dim]
                            embedding = embedding.reshape(-1, order="F")
                            sentiment = np.array(transcript[key]['sentiment_text'])[:, feature_dim]
                            sentiment = sentiment.reshape(-1, order="F")
                            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                            sentiment = sentiment / (np.linalg.norm(sentiment) + 1e-8)
                            feature = np.concatenate([embedding, sentiment])
                        elif feature_type == 'all_audio':
                            prosody = np.array(transcript[key]['prosody_audio'])
                            # xvector = np.array(transcript[key]['xvector_audio'])
                            wavlm = np.array(transcript[key]['wavlm_audio'])
                            prosody = prosody / (np.linalg.norm(prosody) + 1e-8)
                            # xvector = xvector / (np.linalg.norm(xvector) + 1e-8)
                            wavlm = wavlm / (np.linalg.norm(wavlm) + 1e-8)
                            feature = np.concatenate([prosody, wavlm])
                            # feature = np.concatenate([prosody, xvector, wavlm])

                        if feature is not None:
                            X.append(feature)
                            y.append(label)
                    except Exception as e:
                        pass

        X = np.array(X)
        print('Num Nan:', np.isnan(X[:, 0]).sum())
        X = np.nan_to_num(X, nan=0.0)
        y = np.array(y)
        X, y = shuffle(X, y, random_state=42)
        return X, y