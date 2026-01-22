from persona_fine_tuner import PersonaFineTuner

persona_fine_tuner = PersonaFineTuner()


if __name__ == "__main__":
    ft = PersonaFineTuner(poll_interval_sec=30)

    # 1) Train (SFT)
    # ft_model_id = ft.train(
    #     train_path="train.jsonl",
    #     val_path="val.jsonl",
    #     base_model="gpt-4.1-mini-2025-04-14",
    #     method="supervised",
    #     n_epochs=2,
    # )

    # 2) Inference (after you have ft_model_id)
    # out = ft.infer(
    #     model=ft_model_id,
    #     person_id="P0007",
    #     meta={"age": 72, "gender": "F", "race": "Asian", "moca": 23},
    #     question="How do you usually handle stress?",
    # )
    # print(out)
    pass

