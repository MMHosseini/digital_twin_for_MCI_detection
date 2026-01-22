import json
import os
from control_panel import ControlPanel

controller = ControlPanel()

SYSTEM_PROMPT = (
    "You are simulating the respondent in a research dataset. "
    "Match the respondent's style (tone, verbosity, hedging, formality). "
    "Do NOT mention META fields unless explicitly asked. "
    "Do NOT invent personal facts; if unknown, answer in-character with uncertainty."
)

class DataHandler:
    def create_jsonl_for_supervised_fine_tuning(self):
        raw_transcripts_path = controller.raw_transcripts_path
        agents = os.listdir(raw_transcripts_path)

        new_dataset_save_path = controller.language_model_supervised_fine_tune_train_path
        with open(new_dataset_save_path, "w", encoding="utf-8") as f1:
            for agent in agents:
                old_transcript_path = os.path.join(raw_transcripts_path, agent)

                general_info_path = os.path.join(raw_transcripts_path, old_transcript_path, 'general_info.json')
                with open(general_info_path, "r", encoding="utf-8") as f2:
                    general_info = json.load(f2)
                age = general_info['age']
                gender = general_info['gender']

                old_transcripts = os.listdir(old_transcript_path)
                old_transcripts.remove("general_info.json")

                for old_transcript in old_transcripts:
                    json_path = os.path.join(raw_transcripts_path, old_transcript_path, old_transcript)

                    with open(json_path, "r", encoding="utf-8") as f3:
                        old_data = json.load(f3)

                    id = old_data['id']
                    order_of_interview = old_data['order_of_interview']
                    date = old_data['date']
                    topic = old_data['topic']
                    moderator_transcript = old_data['moderator_transcript']
                    participant_transcript = old_data['participant_transcript']

                    moderator_transcript_keys = list(map(int, moderator_transcript.keys()))
                    participant_transcript_keys = list(map(int, participant_transcript.keys()))

                    moderator_ranges = self.__find_ranges(moderator_transcript_keys)
                    participant_ranges = self.__find_ranges(participant_transcript_keys)

                    if participant_ranges[0][0] < moderator_ranges[0][0]: # participant cannot start chat
                        participant_ranges = participant_ranges[1:]

                    while len(moderator_ranges) != 0 and len(participant_ranges) != 0:
                        question_range = moderator_ranges[0]
                        answer_range = participant_ranges[0]

                        question = ''
                        for question_index in range(question_range[0], question_range[1]+1):
                            question += moderator_transcript[str(question_index)]['text']

                        answer = ''
                        for answer_index in range(answer_range[0], answer_range[1] + 1):
                            answer += participant_transcript[str(answer_index)]['text']

                        system = {"role": "system", "content": SYSTEM_PROMPT}
                        user = {"role": "user", "content": f"PERSON_ID: {id}\nMETA: age={age} | gender={gender} | "
                                                           f"date_of_interview={date} | topic={topic}\nQUESTION: {question}"}
                        assistant = {"role": "assistant", "content": answer}

                        message = [system, user, assistant]
                        sample = {"message": message}

                        f1.write(json.dumps(sample, ensure_ascii=False) + "\n")

                        del moderator_ranges[0]
                        del participant_ranges[0]


    def create_jsonl_for_direct_preference_optimization(self):
        pass


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