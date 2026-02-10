from persona_fine_tuner import PersonaFineTuner
from control_panel import ControlPanel
from data_handler import DataHandler
from jsonl_handler import JsonlHandler
from text_handler import TextHandler
from text_evaluator import TextEvaluator

persona_fine_tuner = PersonaFineTuner()
jsonl_handler = JsonlHandler()
controller = ControlPanel()
data_handler = DataHandler()
text_handler = TextHandler()
text_evaluator = TextEvaluator()


if __name__ == "__main__":
    if controller.tasks['split_train_validation_sets'] == 1:
        data_handler.split_train_validation_sets()

    if controller.tasks['merge_transcripts'] == 1:
        data_handler.merge_transcripts()
        
    if controller.tasks['correct_transcripts_order'] == 1:
        text_handler.correct_transcripts_order()

    if controller.tasks['create_jsonl_for_supervised_fine_tuning'] == 1:
        jsonl_handler.create_jsonl_for_supervised_fine_tuning()

    if controller.tasks['create_jsonl_for_direct_preference_optimization'] == 1:
        jsonl_handler.create_jsonl_for_direct_preference_optimization()

    if controller.tasks['supervised_fine_tune_language_model'] == 1:
        persona_fine_tuner.supervised_fine_tuning()

    if controller.tasks['dpo_fine_tune_language_model'] == 1:
        persona_fine_tuner.dpo_fine_tuning()

    if controller.tasks['inference_language_model'] == 1:
        persona_fine_tuner.inference_language_model()

    if controller.tasks['extract_text_embeddings'] == 1:
        text_handler.extract_text_embeddings()
        
    if controller.tasks['extract_text_features'] == 1:
        text_handler.extract_text_stylometrics()

    if controller.tasks['text_based_author_identification'] == 1:
        text_evaluator.text_based_author_identification()


