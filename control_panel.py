import os

class ControlPanel:
    current_path = os.getcwd()
    data_path = os.path.join(current_path, 'data')
    api_key_path = os.path.join(data_path, 'api_keys', 'gpt_api_key.txt')
    raw_data_path = os.path.join(data_path, 'raw_data') # includes videos and raw transcripts (need to be processed)
    raw_transcripts_path = os.path.join(raw_data_path, 'transcripts')
    raw_videos_path = os.path.join(raw_data_path, 'videos')

    tasks = {
        'create_jsonl_for_fine_tuning': 0,
        'create_jsonl_for_dpo': 0, # DPO: Direct Preference Optimization
        'supervised_fine_tune': 0,
        'direct_preference_optimization': 0,
    }

