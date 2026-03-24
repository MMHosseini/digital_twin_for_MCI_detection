import json
import os
import ast
import math
import shutil
from datetime import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from control_panel import ControlPanel

controller = ControlPanel()

class DataHandler:
    def split_train_validation_sets(self):
        def get_closest_date(date_list, target_date):
            target = datetime.strptime(target_date, "%Y-%m-%d")
            parsed_dates = [(i, datetime.strptime(d, "%Y-%m-%d")) for i, d in enumerate(date_list)]
            valid = [(i, d) for i, d in parsed_dates if d <= target]
            selected_index, closest = max(valid, key=lambda x: x[1])
            return selected_index


        raw_transcripts_path = controller.raw_transcripts_path
        raw_videos_path = controller.raw_videos_path

        participants = os.listdir(raw_transcripts_path)
        for participant in participants:
            general_info_path = os.path.join(raw_transcripts_path, participant, 'general_info.json')
            with open(general_info_path, 'r') as f:
                data = json.load(f)
            date_list = ast.literal_eval(data['transcripts_info']['list_view']['date'])
            video_list = ast.literal_eval(data['transcripts_info']['list_view']['video'])
            middle_mci_check_date = data['six_month']['date']

            video_exists = []
            transcript_exists = []
            date_of_interview = []
            for video_name in video_list:
                video_path = os.path.join(raw_videos_path, participant, video_name)
                transcript_path = os.path.join(raw_transcripts_path, participant, video_name.replace('.mp4', '.json'))
                if os.path.isfile(video_path) and os.path.isfile(transcript_path):
                    video_exists.append(video_path)
                    transcript_exists.append(transcript_path)
                    date_of_interview.append(date_list[video_list.index(video_name)])

            train_set = os.path.join(controller.train_data_path, participant)
            validation_set = os.path.join(controller.validation_data_path, participant)
            os.makedirs(train_set, exist_ok=True)
            os.makedirs(validation_set, exist_ok=True)

            shutil.copy(str(general_info_path), str(os.path.join(train_set, Path(str(general_info_path)).name)))
            shutil.copy(str(general_info_path), str(os.path.join(validation_set, Path(str(general_info_path)).name)))

            mid_index = get_closest_date(date_of_interview, middle_mci_check_date)
            train_percentage = 0.85
            train_indices = np.ones((len(video_exists)), dtype='int')
            train_indices[int(math.ceil(train_percentage*mid_index)):mid_index+1] = 0
            train_indices[mid_index + int(math.ceil(train_percentage*(len(video_exists)-mid_index))):] = 0

            for i in range(len(train_indices)):
                src_video = video_exists[i]
                src_transcript = transcript_exists[i]
                if train_indices[i] == 1:
                    dest_video = os.path.join(train_set, Path(str(video_exists[i])).name)
                    dest_transcript = os.path.join(train_set, Path(str(transcript_exists[i])).name)
                else:
                    dest_video = os.path.join(validation_set, Path(str(video_exists[i])).name)
                    dest_transcript = os.path.join(validation_set, Path(str(transcript_exists[i])).name)
                shutil.copy(str(src_video), str(dest_video))
                shutil.copy(str(src_transcript), str(dest_transcript))

    def merge_transcripts(self):
        def get_pause(last_end_time, start_time):
            gap = start_time - last_end_time
            gap = max(0, gap)
            pause = ''
            if gap < 0.2:
                pause = '<PAUSE_NONE>'
            elif 0.2 <= gap < 0.8:
                pause = '<PAUSE_SHORT>'
            elif 0.8 <= gap < 1.8:
                pause = '<PAUSE_MED>'
            elif 1.8 <= gap < 3.5:
                pause = '<PAUSE_LONG>'
            elif gap >= 3.5:
                pause = '<PAUSE_VLONG>'
            return pause
        
        def get_tempo(text, duration):
            num_words = text.count(' ')
            words_per_second = num_words / duration
            tempo = ''
            if words_per_second <= 2:
                tempo = '<TEMPO_SLOW>'
            elif 2 < words_per_second <= 3:
                tempo = '<TEMPO_MED>'
            elif words_per_second > 3:
                tempo = '<TEMPO_FAST>'
            return tempo

        def create_new_transcript_with_tempo_and_pause(index, start_end_ranges, transcript):
            new_transcript = dict()
            for start_end_range in start_end_ranges:
                sentence = ''
                last_end_time = 1000000
                for idx in range(start_end_range[0], start_end_range[1] + 1):
                    idx = str(idx)
                    start_time = transcript[idx]['start']
                    if int(idx) != start_end_range[0]:
                        sentence += get_pause(last_end_time, start_time)
                    sentence += transcript[idx]['text']
                    end_time = transcript[idx]['end']
                    sentence += get_tempo(transcript[idx]['text'], end_time - start_time)
                    last_end_time = end_time

                segment = transcript[str(start_end_range[0])]['segment']
                start_time = transcript[str(start_end_range[0])]['start']
                end_time = transcript[str(start_end_range[-1])]['end']
                duration = end_time - start_time
                new_transcript[index] = {'segment': segment, 'text': sentence, 'start': start_time, 'end': end_time, 'duration': duration}
                index += 2
            return new_transcript

        def create_new_transcript_without_tempo_and_pause(index, start_end_ranges, transcript):
            new_transcript = dict()
            for start_end_range in start_end_ranges:
                sentence = ''
                for idx in range(start_end_range[0], start_end_range[1] + 1):
                    idx = str(idx)
                    sentence += transcript[idx]['text']

                segment = transcript[str(start_end_range[0])]['segment']
                start_time = transcript[str(start_end_range[0])]['start']
                end_time = transcript[str(start_end_range[-1])]['end']
                duration = end_time - start_time
                new_transcript[index] = {'segment': segment, 'text': sentence, 'start': start_time, 'end': end_time, 'duration': duration}
                index += 2
            return new_transcript

        train_set = controller.train_data_path
        validation_set = controller.validation_data_path
        subsets = [train_set, validation_set]

        for subset in subsets:
            participants = os.listdir(subset)

            for participant in participants:
                current_path = os.path.join(subset, participant)
                transcripts_path = [str(p) for p in Path(current_path).glob("*.json")]
                transcripts_path.remove(str(os.path.join(current_path, 'general_info.json')))

                for transcript_path in tqdm(transcripts_path):
                    with open(transcript_path, 'r') as f:
                        data = json.load(f)

                    moderator_transcript = data['moderator_transcript'].copy()
                    participant_transcript = data['participant_transcript'].copy()

                    moderator_transcript_keys = list(map(int, moderator_transcript.keys()))
                    participant_transcript_keys = list(map(int, participant_transcript.keys()))

                    moderator_ranges = self.__find_ranges(moderator_transcript_keys)
                    participant_ranges = self.__find_ranges(participant_transcript_keys)

                    starter = 'moderator' if moderator_ranges[0][0] == 0 else 'participant'
                    if starter == 'moderator':
                        moderator_start_index = 0
                        participant_start_index = 1
                    else:
                        moderator_start_index = 1
                        participant_start_index = 0

                    new_moderator_transcript = create_new_transcript_without_tempo_and_pause(moderator_start_index, moderator_ranges, moderator_transcript)
                    new_participant_transcript = create_new_transcript_with_tempo_and_pause(participant_start_index, participant_ranges, participant_transcript)
                    data['moderator_transcript'] = new_moderator_transcript
                    data['participant_transcript'] = new_participant_transcript

                    with open(transcript_path, 'w') as f:
                        json.dump(data, f, indent=4)

    def extract_mci_info(self):
        subsets = [controller.train_data_path, controller.validation_data_path]
        for subset in subsets:
            participants = os.listdir(subset)
            participants.sort()
            for participant in participants:
                current_path = os.path.join(subset, participant)
                transcripts_path = [str(p) for p in Path(current_path).glob("*.json")]
                general_info_path = os.path.join(current_path, 'general_info.json')
                with open(general_info_path, "r", encoding="utf-8") as f0:
                    general_info = json.load(f0)
                transcripts_path.remove(str(general_info_path))

                id = general_info['id']
                gender = general_info['gender']
                age = general_info['age']
                race = general_info['race']
                transcripts_info = general_info['transcripts_info']
                baseline_info = general_info['baseline']
                six_month_info = general_info['six_month']
                twelve_month_info = general_info['twelve_month']
                baseline_cognitive_status = 1 if general_info['baseline']['normal_cognitive'] == 'yes' else 0
                six_month_cognitive_status = 1 if general_info['six_month']['normal_cognitive'] == 'yes' else 0
                twelve_month_cognitive_status = 1 if general_info['twelve_month']['normal_cognitive'] == 'yes' else 0
                mci_info = dict()
                for transcript_path in tqdm(transcripts_path):
                    with open(transcript_path, "r", encoding="utf-8") as f1:
                        data = json.load(f1)

                    if 'mci_info' in data:
                        continue

                    participant_label = participants.index(id)
                    gender_id = 0 if gender == 'male' else 1

                    interview_date = datetime.strptime(data['date'], "%Y-%m-%d").date()
                    baseline_date = datetime.strptime(baseline_info['date'], "%Y-%m-%d").date()
                    six_month_date = datetime.strptime(six_month_info['date'], "%Y-%m-%d").date()
                    twelve_month_date = datetime.strptime(twelve_month_info['date'], "%Y-%m-%d").date()

                    past_from_age = (interview_date - baseline_date).days
                    age_at_interview = age + (past_from_age / 365)

                    if baseline_info['moca_face_to_face'] is not None:
                        baseline_mci_score = float(baseline_info['moca_face_to_face'])
                    else:
                        baseline_mci_score = float(baseline_info['moca_telephone']) / 22 * 30

                    if six_month_info['moca_face_to_face'] is not None:
                        six_month_mci_score = float(six_month_info['moca_face_to_face'])
                    else:
                        six_month_mci_score = float(six_month_info['moca_telephone']) / 22 * 30

                    if twelve_month_info['moca_face_to_face'] is not None:
                        twelve_month_mci_score = float(twelve_month_info['moca_face_to_face'])
                    else:
                        twelve_month_mci_score = float(twelve_month_info['moca_telephone']) / 22 * 30

                    interview_time_mci_score  = None
                    last_reported_mci_score = None
                    past_days_from_last_mci_evaluation = None
                    interview_period_index = None
                    last_cognitive_status = None
                    if interview_date <= baseline_date:
                        past_days_from_last_mci_evaluation = (interview_date - baseline_date).days
                        interview_time_mci_score = baseline_mci_score
                        last_reported_mci_score = baseline_mci_score
                        interview_period_index = 0
                        last_cognitive_status = baseline_cognitive_status
                    elif baseline_date < interview_date < six_month_date:
                        past_days_from_last_mci_evaluation = (interview_date - baseline_date).days
                        gap = (six_month_date - baseline_date).days
                        interview_time_mci_score = baseline_mci_score + (past_days_from_last_mci_evaluation / gap) * (six_month_mci_score - baseline_mci_score)
                        last_reported_mci_score = baseline_mci_score
                        interview_period_index = 1
                        last_cognitive_status = baseline_cognitive_status
                    elif six_month_date <= interview_date < twelve_month_date:
                        past_days_from_last_mci_evaluation = (interview_date - six_month_date).days
                        gap = (twelve_month_date - six_month_date).days
                        interview_time_mci_score = six_month_mci_score + (past_days_from_last_mci_evaluation / gap) * (twelve_month_mci_score - six_month_mci_score)
                        last_reported_mci_score = six_month_mci_score
                        interview_period_index = 2
                        last_cognitive_status = six_month_cognitive_status
                    elif interview_date >= twelve_month_date:
                        past_days_from_last_mci_evaluation = (interview_date - twelve_month_date).days
                        interview_time_mci_score = twelve_month_mci_score
                        last_reported_mci_score = twelve_month_mci_score
                        interview_period_index = 3
                        last_cognitive_status = twelve_month_cognitive_status

                    mci_info['participant_label'] = participant_label
                    mci_info['gender'] = gender_id
                    mci_info['age_at_start'] = age
                    mci_info['interview_period_index'] = interview_period_index
                    mci_info['age_at_interview'] = age_at_interview
                    mci_info['last_reported_mci_score'] = last_reported_mci_score
                    mci_info['past_days_from_last_mci_evaluation'] = past_days_from_last_mci_evaluation
                    mci_info['interview_time_mci_score'] = interview_time_mci_score
                    mci_info['last_cognitive_status'] = last_cognitive_status

                    data['mci_info'] = mci_info
                    with open(transcript_path, 'w') as f2:
                        json.dump(data, f2, indent=4)

    def load_mci_data(self, subset, features, participant, respondent='participant_transcript'):
        if subset == 'train':
            data_path = controller.train_data_path
        elif subset == 'validation':
            data_path = controller.validation_data_path
        else:
            print('Error in load_data: You have chosen a wrong subset for loading')
            return

        condition_matrix = list()
        question_matrix = list()
        answer_matrix = list()
        mci_vector = list()

        current_path = os.path.join(data_path, participant)
        transcripts_path = [str(p) for p in Path(current_path).glob("*.json")]
        general_info_path = os.path.join(current_path, 'general_info.json')
        transcripts_path.remove(str(general_info_path))

        for transcript_path in tqdm(transcripts_path):
            with open(transcript_path, "r", encoding="utf-8") as f1:
                data = json.load(f1)

            moderator_transcript = data['moderator_transcript']
            respondant_transcript = data[respondent]

            condition_vector = list()
            if respondent == 'simulated_transcript_raw_gpt':
                condition_vector.append(np.random.randint(5, 10))
                condition_vector.append(np.random.randint(0, 2))
                condition_vector.append(1 + np.random.uniform(0, 0.02))
                condition_vector.append(np.random.randint(20, 31))
                condition_vector.append(np.random.randint(0, 365))
                condition_vector.append(np.random.randint(0, 2))
                condition_vector.append(data['topic_descriptor'])
                mci_score = np.random.randint(15, 31) / 30
            else:
                condition_vector.append(data['mci_info']['participant_label'])
                condition_vector.append(data['mci_info']['interview_period_index'])
                condition_vector.append(data['mci_info']['age_at_interview'] / data['mci_info']['age_at_start'])
                condition_vector.append(data['mci_info']['last_reported_mci_score'])
                condition_vector.append(data['mci_info']['past_days_from_last_mci_evaluation'])
                condition_vector.append(data['mci_info']['last_cognitive_status'])
                condition_vector.append(data['topic_descriptor'])
                mci_score = data['mci_info']['interview_time_mci_score'] / 30

            condition_vector = sum(([x] if not isinstance(x, list) else x for x in condition_vector), [])
            condition_vector = list(np.random.rand(len(condition_vector)))

            moderator_transcript_keys = list(moderator_transcript.keys())
            respondant_transcript_keys = list(respondant_transcript.keys())
            for i in range(len(moderator_transcript_keys)):
                moderator_transcript_key = moderator_transcript_keys[i]
                respondant_transcript_key = respondant_transcript_keys[i]
                question_vector = list()
                answer_vector = list()
                for feature in features:
                    question = np.array(moderator_transcript[moderator_transcript_key][feature])
                    answer = np.array(respondant_transcript[respondant_transcript_key][feature])
                    question_vector.append(list(question[:, 0:2].reshape(-1))) # [:, 0]: mean, [:, 1]: std, [:, 2]: all_together
                    answer_vector.append(list(answer[:, 0:2].reshape(-1)))
                question_vector = sum(([x] if not isinstance(x, list) else x for x in question_vector), [])
                answer_vector = sum(([x] if not isinstance(x, list) else x for x in answer_vector), [])

                condition_vector = np.array(condition_vector)
                question_vector = np.array(question_vector)
                answer_vector = np.array(answer_vector)

                norm = np.linalg.norm(condition_vector) # L2 norm
                condition_vector = condition_vector / norm
                norm = np.linalg.norm(question_vector)  # L2 norm
                question_vector = question_vector / norm
                norm = np.linalg.norm(answer_vector)  # L2 norm
                answer_vector = answer_vector / norm

                condition_matrix.append(condition_vector)
                question_matrix.append(question_vector)
                answer_matrix.append(answer_vector)
                mci_vector.append(mci_score)
        condition_matrix = np.array(condition_matrix)
        question_matrix = np.array(question_matrix)
        answer_matrix = np.array(answer_matrix)
        mci_vector = np.array(mci_vector)

        return condition_matrix, question_matrix, answer_matrix, mci_vector

    def load_text_and_audio_data(self, subset, participant, respondent='participant_transcript'):
        if subset == 'train':
            data_path = controller.train_data_path
        elif subset == 'validation':
            data_path = controller.validation_data_path
        else:
            print('Error in load_data: You have chosen a wrong subset for loading')
            return

        current_path = os.path.join(data_path, participant)
        transcripts_path = [str(p) for p in Path(current_path).glob("*.json")]
        general_info_path = os.path.join(current_path, 'general_info.json')
        transcripts_path.remove(str(general_info_path))

        embedding_text_matrix = list()
        sentiment_text_matrix = list()
        prosody_audio_matrix = list()
        xvector_audio_matrix = list()
        wavlm_audio_matrix = list()
        epsilon = 1e-8
        for transcript_path in tqdm(transcripts_path):
            with open(transcript_path, "r", encoding="utf-8") as f1:
                data = json.load(f1)

            respondant_transcript = data[respondent]
            respondant_transcript_keys = list(respondant_transcript.keys())
            for respondant_transcript_key in respondant_transcript_keys:
                try:
                    embedding_text = np.array(respondant_transcript[respondant_transcript_key]['embedding_text'])[:, 0:2]
                    sentiment_text = np.array(respondant_transcript[respondant_transcript_key]['sentiment_text'])[:, 0:2]
                    prosody_audio = np.array(respondant_transcript[respondant_transcript_key]['prosody_audio'])
                    xvector_audio = np.array(respondant_transcript[respondant_transcript_key]['xvector_audio'])
                    wavlm_audio = np.array(respondant_transcript[respondant_transcript_key]['wavlm_audio'])

                    embedding_text = embedding_text.reshape((-1))
                    sentiment_text = sentiment_text.reshape((-1))

                    norm = np.linalg.norm(embedding_text) + epsilon  # L2 norm
                    embedding_text = embedding_text / norm
                    norm = np.linalg.norm(sentiment_text) + epsilon
                    sentiment_text = sentiment_text / norm
                    norm = np.linalg.norm(prosody_audio) + epsilon
                    prosody_audio = prosody_audio / norm
                    norm = np.linalg.norm(xvector_audio) + epsilon
                    xvector_audio = xvector_audio / norm
                    norm = np.linalg.norm(wavlm_audio) + epsilon
                    wavlm_audio = wavlm_audio / norm

                    embedding_text_matrix.append(embedding_text)
                    sentiment_text_matrix.append(sentiment_text)
                    prosody_audio_matrix.append(prosody_audio)
                    xvector_audio_matrix.append(xvector_audio)
                    wavlm_audio_matrix.append(wavlm_audio)
                except Exception as e:
                    print(e)

        embedding_text_matrix = np.array(embedding_text_matrix)
        sentiment_text_matrix = np.array(sentiment_text_matrix)
        prosody_audio_matrix = np.array(prosody_audio_matrix)
        xvector_audio_matrix = np.array(xvector_audio_matrix)
        wavlm_audio_matrix = np.array(wavlm_audio_matrix)

        text_matrix = np.concatenate((embedding_text_matrix, sentiment_text_matrix), axis=1)
        mean_xvector_matrix = np.mean(xvector_audio_matrix, axis=0, keepdims=True)
        mean_xvector_matrix = np.repeat(mean_xvector_matrix, repeats=text_matrix.shape[0], axis=0)

        return text_matrix, mean_xvector_matrix, prosody_audio_matrix, wavlm_audio_matrix

    def make_tf_dataset(self, arrays, batch_size=32, shuffle=True, seed=42, dtype=np.float32):
        arrays = [np.asarray(arr, dtype=dtype) for arr in arrays]
        ds = tf.data.Dataset.from_tensor_slices(tuple(arrays))
        if shuffle:
            n = len(arrays[0])
            ds = ds.shuffle(buffer_size=n, seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        return ds

    # def make_tf_dataset(self, question_np, condition_np, answer_np, mci_np, batch_size=32, shuffle=True, seed=42):
    #     # Ensure float32 for TF
    #     question_np = np.asarray(question_np, dtype=np.float32)
    #     condition_np = np.asarray(condition_np, dtype=np.float32)
    #     answer_np = np.asarray(answer_np, dtype=np.float32)
    #     mci_np = np.asarray(mci_np, dtype=np.float32)
    #
    #     ds = tf.data.Dataset.from_tensor_slices((question_np, condition_np, answer_np, mci_np))
    #     if shuffle:
    #         ds = ds.shuffle(buffer_size=len(mci_np), seed=seed, reshuffle_each_iteration=True)
    #     ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    #     return ds

    def __find_ranges(self, nums):
        if not nums:
            return []

        nums = sorted(nums)
        ranges = []
        start = prev = nums[0]

        for n in nums[1:]:
            if n == prev + 1:
                prev = n
            else:
                ranges.append([start, prev])
                start = prev = n

        ranges.append([start, prev])
        return ranges

