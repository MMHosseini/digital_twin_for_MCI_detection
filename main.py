from persona_fine_tuner import PersonaFineTuner
from control_panel import ControlPanel
from data_handler import DataHandler
from jsonl_handler import JsonlHandler

persona_fine_tuner = PersonaFineTuner()
jsonl_handler = JsonlHandler()
controller = ControlPanel()
data_handler = DataHandler()


if __name__ == "__main__":
    if controller.tasks['split_train_validation_sets'] == 1:
        data_handler.split_train_validation_sets()

    if controller.tasks['merge_transcripts'] == 1:
        data_handler.merge_transcripts()

    if controller.tasks['create_jsonl_for_supervised_fine_tuning'] == 1:
        jsonl_handler.create_jsonl_for_supervised_fine_tuning()

    if controller.tasks['create_jsonl_for_direct_preference_optimization'] == 1:
        jsonl_handler.create_jsonl_for_direct_preference_optimization()

    if controller.tasks['supervised_fine_tune_language_model'] == 1:
        persona_fine_tuner.supervised_fine_tuning()

    if controller.tasks['dpo_fine_tune_language_model'] == 1:
        persona_fine_tuner.dpo_fine_tuning()

    if controller.tasks['inference_fine_tuned_language_model'] == 1:
        persona_fine_tuner.infer_fine_tuned_model()


