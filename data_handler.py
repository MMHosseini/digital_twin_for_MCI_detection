import json
import os
from openai import OpenAI
from typing import List
from control_panel import ControlPanel

controller = ControlPanel()


class DataHandler:
    def create_jsonl_for_supervised_fine_tuning(self):
        SYSTEM_PROMPT = (
            "You are simulating the respondent in a research dataset. "
            "Match the respondent's style (tone, verbosity, hedging, formality). "
            "Do NOT mention META fields unless explicitly asked. "
            "Do NOT invent personal facts; if unknown, answer in-character with uncertainty."
        )

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

                        messages = [system, user, assistant]
                        sample = {"messages": messages}

                        f1.write(json.dumps(sample, ensure_ascii=False) + "\n")

                        del moderator_ranges[0]
                        del participant_ranges[0]


    def create_jsonl_for_direct_preference_optimization(self):
        def generate_dissimilar_responses(client, question: str, real_answer: str, num_dissimilar_responses: int = 5,
                                          model: str = "gpt-4.1-mini", temperature: float = 1.2, max_tokens: int = 400
                                          ) -> List[str]:
            """
            Generate N responses that are deliberately dissimilar in style
            from the real person's answer. Intended for DPO negatives.
            """

            system_prompt = (
                "You are generating alternative responses for preference learning.\n"
                "Your goal is to produce answers that are PLAUSIBLE but stylistically DIFFERENT\n"
                "from the real respondent.\n\n"
                "Avoid:\n"
                "- matching tone, verbosity, or phrasing of the real answer\n"
                "- hedging or uncertainty if the real answer uses it\n"
                "- first-person storytelling if the real answer is personal\n\n"
                "Prefer:\n"
                "- more formal or instructional tone\n"
                "- generic advice\n"
                "- close length to the original response\n"
                "- structured or clinical phrasing\n\n"
                "Do NOT mention the real answer explicitly."
            )

            user_prompt = (
                f"QUESTION:\n{question}\n\n"
                f"REAL ANSWER (for reference only — do NOT imitate):\n{real_answer}\n\n"
                f"Generate {num_dissimilar_responses} alternative answers that are stylistically different.\n"
                "Return ONLY the answers, one per line."
            )

            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            # Split lines and clean
            outputs = []
            for line in response.output_text.split("\n"):
                line = line.strip()
                if line:
                    outputs.append(line)

            return outputs[:num_dissimilar_responses]

        SYSTEM_PROMPT = (
            "You are simulating the respondent in a research dataset. "
            "Match the respondent's style (tone, verbosity, hedging, formality). "
            "Do NOT mention META fields unless explicitly asked. "
            "Do NOT invent personal facts; if unknown, answer in-character with uncertainty."
        )

        with open(controller.api_key_path, 'r') as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)

        raw_transcripts_path = controller.raw_transcripts_path
        agents = os.listdir(raw_transcripts_path)

        new_dataset_save_path = controller.language_model_direct_preference_optimization_train_path
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

                    if participant_ranges[0][0] < moderator_ranges[0][0]:  # participant cannot start chat
                        participant_ranges = participant_ranges[1:]

                    while len(moderator_ranges) != 0 and len(participant_ranges) != 0:
                        question_range = moderator_ranges[0]
                        answer_range = participant_ranges[0]

                        question = ''
                        for question_index in range(question_range[0], question_range[1] + 1):
                            question += moderator_transcript[str(question_index)]['text']

                        answer = ''
                        for answer_index in range(answer_range[0], answer_range[1] + 1):
                            answer += participant_transcript[str(answer_index)]['text']

                        fake_answers = generate_dissimilar_responses(client, question, answer)

                        for fake_answer in fake_answers:
                            system = {"role": "system", "content": SYSTEM_PROMPT}
                            user = {"role": "user", "content": f"PERSON_ID: {id}\nMETA: age={age} | gender={gender} | "
                                                               f"date_of_interview={date} | topic={topic}\nQUESTION: {question}"}
                            preferred_output = [{"role": "assistant", "content": answer}]
                            non_preferred_output = [{"role": "assistant", "content": fake_answer}]

                            messages = [system, user]
                            system_input = {"messages": messages}

                            sample = {'input': system_input, 'preferred_output': preferred_output, 'non_preferred_output': non_preferred_output}
                            f1.write(json.dumps(sample, ensure_ascii=False) + "\n")

                        del moderator_ranges[0]
                        del participant_ranges[0]


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

