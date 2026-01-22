import os

class ControlPanel:
    current_path = os.getcwd()
    data_path = os.path.join(current_path, 'data')
    api_key_path = os.path.join(data_path, 'api_keys', 'gpt_api_key.txt')
    raw_data_path = os.path.join(data_path, 'raw_data') # includes videos and raw transcripts (need to be processed)
    raw_transcripts_path = os.path.join(raw_data_path, 'transcripts')
    raw_videos_path = os.path.join(raw_data_path, 'videos')
    language_model_supervised_fine_tune_train_path = os.path.join(data_path, 'supervised_fine_tune_train_set.jsonl')
    language_model_supervised_fine_tune_validation_path = os.path.join(data_path, 'supervised_fine_tune_validation_set.jsonl')
    language_model_direct_preference_optimization_train_path = os.path.join(data_path, 'direct_preference_optimization_train_set.jsonl')
    language_model_direct_preference_optimization_validation_path = os.path.join(data_path, 'direct_preference_optimization_validation_set.jsonl')

    tasks = {
        'create_jsonl_for_supervised_fine_tuning': 1,
        'create_jsonl_for_direct_preference_optimization': 0,
        'fine_tune_language_model': 0,
    }

