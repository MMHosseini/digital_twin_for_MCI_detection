import os
from pathlib import Path
import numpy as np
import json
from typing import Any, Dict, Tuple, Union, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from control_panel import ControlPanel

LabelArray = Union[np.ndarray, list]
controller = ControlPanel()

class TextEvaluator:
    def text_based_author_identification(self):
        feature_types = ['all']
        # feature_types = ['embedding', 'stylometric', 'all']
        os.makedirs(controller.result_path, exist_ok=True)
        save_path = os.path.join(controller.result_path, 'text_based_author_identification.txt')
        with open(save_path, "w") as f:
            for feature_type in feature_types:
                X_train, X_val, y_train, y_val = self.load_data(feature_type=feature_type)
                # X_train = np.random.rand(1000, 50)
                # X_val= np.random.rand(200, 50)
                # y_train = np.random.randint(0, 5, size=1000)
                # y_val = np.random.randint(0, 5, size=200)

                model, le = self.train_linear_svm_l2(X_train, y_train, C=1.0)
                res = self.evaluate_linear_svm_l2(model, le, X_val, y_val)

                val1 = res["macro_f1"]
                val2 = res["labels"]
                val3 = res["confusion_matrix"]
                val4 = res["classification_report"]

                print("Feature:", feature_type, "\n")
                print("Macro-F1:", val1, "\n")
                print("Labels:", val2, "\n")
                print("Confusion matrix:\n", val3)
                print("\nClassification report:\n", val4)

                f.write(f"Feature: {feature_type}\n")
                f.write(f"Macro-F1: {val1}\n")
                f.write(f"Labels: {val2}\n")
                f.write(f"Confusion matrix: {val3}\n")
                f.write(f"Classification report: {val4}\n")
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

        report = classification_report(
            y_val_enc,
            y_pred_enc,
            labels=labels_enc,
            target_names=[str(c) for c in le.classes_],
            digits=4,
        )

        y_pred_labels = le.inverse_transform(y_pred_enc)

        info = {
            "macro_f1": float(macro_f1),
            "confusion_matrix": cm,
            "labels": le.classes_,
            "classification_report": report,
            "y_pred_labels": y_pred_labels,
        }
        return info

    def load_data(self, feature_type='all'):
        train_path = controller.train_data_path
        validation_path = controller.validation_data_path
        subsets = [train_path, validation_path]

        X_train = []
        X_val = []
        y_train = []
        y_val = []
        participants = sorted(os.listdir(train_path))
        for subset in subsets:
            for participant in participants:
                label = participants.index(participant)

                current_path = os.path.join(subset, participant)
                transcripts_path = [str(p) for p in Path(current_path).glob("*.json")]
                general_info_path = os.path.join(current_path, 'general_info.json')
                transcripts_path.remove(str(general_info_path))
                for transcript_path in transcripts_path:
                    with open(transcript_path, 'r') as f:
                        data = json.load(f)
                    participant_transcript = data['participant_transcript']
                    key_list = list(participant_transcript.keys())
                    for key in key_list:
                        embedding = np.array(participant_transcript[key]['embedding_text'])
                        embedding = embedding.reshape(-1, order="F")
                        stylometric = np.array(participant_transcript[key]['stylometric_text'])

                        if feature_type == 'embedding':
                            feature = embedding
                        elif feature_type == 'stylometric':
                            feature = stylometric
                        else:
                            feature = np.concatenate([embedding, stylometric])

                        if subset.endswith('train'):
                            X_train.append(feature)
                            y_train.append(label)
                        elif subset.endswith('validation'):
                            X_val.append(feature)
                            y_val.append(label)

        X_train = np.array(X_train)
        X_val = np.array(X_val)
        y_train = np.array(y_train)
        y_val = np.array(y_val)

        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_val, y_val = shuffle(X_val, y_val, random_state=42)

        return X_train, X_val, y_train, y_val