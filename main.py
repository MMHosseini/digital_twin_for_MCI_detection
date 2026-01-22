from persona_fine_tuner import PersonaFineTuner
from control_panel import ControlPanel
from data_handler import DataHandler

persona_fine_tuner = PersonaFineTuner()
controller = ControlPanel()
data_handler = DataHandler()


if __name__ == "__main__":
    if controller.tasks['create_jsonl_for_supervised_fine_tuning'] == 1:
        data_handler.create_jsonl_for_supervised_fine_tuning()

    if controller.tasks['create_jsonl_for_direct_preference_optimization'] == 1:
        data_handler.create_jsonl_for_direct_preference_optimization()

    if controller.tasks['fine_tune_language_model'] == 1:
        sft_model = persona_fine_tuner.train(base_model = "gpt-4.1-mini-2025-04-14", method = "supervised", n_epochs = 2,
                                             train_path=controller.language_model_supervised_fine_tune_train_path, 
                                             val_path=controller.language_model_supervised_fine_tune_validation_path)
        print("SFT model:", sft_model)
        dpo_model = persona_fine_tuner.train(base_model = sft_model, method = "dpo", n_epochs = 2,
                                             train_path=controller.language_model_direct_preference_optimization_train_path,
                                             val_path=controller.language_model_direct_preference_optimization_validation_path)
        print("DPO model:", dpo_model)
        out = persona_fine_tuner.infer(model=dpo_model, person_id="P0007", meta={"age": 72, "gender": "F"},
                                       question="How do you usually handle stress?")
        print(out)

