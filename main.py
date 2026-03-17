from persona_fine_tuner import PersonaFineTuner
from control_panel import ControlPanel
from data_handler import DataHandler
from jsonl_handler import JsonlHandler
from text_handler import TextHandler
from evaluator import Evaluator
from text_model_handler import TextModelHandler
from audio_model_handler import AudioModelHandler
from audio_handler import AudioHandler

persona_fine_tuner = PersonaFineTuner()
jsonl_handler = JsonlHandler()
controller = ControlPanel()
data_handler = DataHandler()
text_handler = TextHandler()
evaluator = Evaluator()
text_model_handler = TextModelHandler()
audio_model_handler = AudioModelHandler()
audio_handler = AudioHandler()


if __name__ == "__main__":
    if controller.tasks['split_train_validation_sets'] == 1:
        data_handler.split_train_validation_sets()

    if controller.tasks['merge_transcripts'] == 1:
        data_handler.merge_transcripts()

    if controller.tasks['extract_mci_info'] == 1:
        data_handler.extract_mci_info()

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
        
    if controller.tasks['prepare_rag_data'] == 1:
        text_handler.prepare_rag_data()

    if controller.tasks['inference_language_model'] == 1:
        persona_fine_tuner.inference_language_model()

    if controller.tasks['extract_text_embeddings'] == 1:
        text_handler.extract_text_embeddings()
        
    if controller.tasks['extract_text_sentiments'] == 1:
        text_handler.extract_text_sentiments()
        
    if controller.tasks['extract_title_descriptor'] == 1:
        text_handler.extract_title_descriptor()

    if controller.tasks['text_based_author_identification'] == 1:
        evaluator.text_based_author_identification()
        
    if controller.tasks['extract_audio_from_video'] == 1:
        audio_handler.extract_audio_from_video()
        
    if controller.tasks['extract_audio_prosody'] == 1:
        audio_handler.extract_audio_prosody()

    if controller.tasks['extract_audio_xvector'] == 1:
        audio_handler.extract_audio_xvector()

    if controller.tasks['extract_audio_wavlm'] == 1:
        audio_handler.extract_audio_wavlm()
        
    if controller.tasks['audio_based_author_identification'] == 1:
        evaluator.audio_based_author_identification()

    if controller.tasks['train_model'] == 1:
        text_model_handler.train_model()
        audio_model_handler.train_model()

    if controller.tasks['validate_model'] == 1:
        text_model_handler.validate_model()
        audio_model_handler.validate_model()
