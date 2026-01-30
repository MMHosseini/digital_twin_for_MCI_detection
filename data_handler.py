import json
import os
import ast
import math
import shutil
from datetime import datetime
import numpy as np
from pathlib import Path
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

                for transcript_path in transcripts_path:
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

