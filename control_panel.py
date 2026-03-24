import os

class ControlPanel:
    current_path = os.getcwd()
    data_path = os.path.join(current_path, 'data')
    api_key_path = os.path.join(data_path, 'api_keys', 'gpt_api_key.txt')
    raw_data_path = os.path.join(data_path, 'raw_data') # includes videos and raw transcripts (need to be processed)
    raw_transcripts_path = os.path.join(raw_data_path, 'transcripts')
    raw_videos_path = os.path.join(raw_data_path, 'videos')
    train_data_path = os.path.join(data_path, 'train')
    validation_data_path = os.path.join(data_path, 'validation')
    model_path = os.path.join(data_path, 'model')
    result_path = os.path.join(data_path, 'result')
    plot_path = os.path.join(data_path, 'plot')
    language_model_supervised_fine_tune_train_path = os.path.join(data_path, 'supervised_fine_tune_train_set.jsonl')
    language_model_supervised_fine_tune_validation_path = os.path.join(data_path, 'supervised_fine_tune_validation_set.jsonl')
    language_model_direct_preference_optimization_train_path = os.path.join(data_path, 'direct_preference_optimization_train_set.jsonl')
    language_model_direct_preference_optimization_validation_path = os.path.join(data_path, 'direct_preference_optimization_validation_set.jsonl')

    supervised_fine_tuned_model_name = os.path.join(data_path, 'supervised_fine_tune_model_name.txt')
    dpo_fine_tuned_model_name = os.path.join(data_path, 'direct_preference_optimization_model_name.txt')

    tasks = {
        'split_train_validation_sets': 0, # done
        'merge_transcripts': 0, # done
        'extract_mci_info': 0, # done
        'correct_transcripts_order': 0, # done
        'create_jsonl_for_supervised_fine_tuning': 0, # done
        'create_jsonl_for_direct_preference_optimization': 0, # done
        'supervised_fine_tune_language_model': 0, # done
        'dpo_fine_tune_language_model': 0, # done
        'prepare_rag_data': 0, # done
        'inference_language_model': 0, # done for raw, sft, and dpo (maybe later for rag)
        'extract_text_embeddings': 0, # redo it later for dpo and rag (only run. Code is ready)
        'extract_text_sentiments': 0, # redo it later for dpo and rag (only run. Code is ready)
        'extract_title_descriptor': 0, # done
        'text_based_author_identification': 0,
        'extract_audio_from_video': 0,
        'extract_audio_prosody': 0,
        'extract_audio_xvector': 0,
        'extract_audio_wavlm': 0,
        'train_audio_mimicker': 0,
        'validate_audio_mimicker': 0,
        'inference_audio_model': 0,
        'audio_based_author_identification': 1,

        'train_model': 0, #
        'validate_model': 0, #
    }

    batch_size = 200
    epochs = 100
    learning_rate = 0.001


