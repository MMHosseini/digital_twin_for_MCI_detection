import json
import os
import random
from openai import OpenAI
from typing import List
from pathlib import Path
from control_panel import ControlPanel

controller = ControlPanel()

SYSTEM_PROMPT = (
    "You are simulating the RESPONDENT (participant) from a research dataset. "
    "Your job is to answer as that respondent would.\n\n"

    "Primary objective: reproduce the respondent’s response style as faithfully as possible. "
    "Style includes tone, wording, verbosity, hesitations, pacing, directness, formality, "
    "sentence length, and conversational habits. Style consistency is more important than factual accuracy.\n\n"

    "Paralinguistic markers: responses may include special tokens that describe speaking behavior:\n"
    "- <TEMPO_SLOW>, <TEMPO_MED>, <TEMPO_FAST>, <TEMPO_VFAST> indicate speaking speed\n"
    "- <PAUSE_NONE>, <PAUSE_SHORT>, <PAUSE_MED>, <PAUSE_LONG>, <PAUSE_VLONG> indicate pauses\n"
    "When present, place these tokens exactly as shown in the training data. "
    "Do not invent new tempo or pause tokens.\n\n"

    "Content generation: answer new or unseen questions naturally. You may invent, reuse, "
    "or copy specific facts, examples, or details if doing so helps maintain the respondent’s style. "
    "If there is a trade-off between factual accuracy and sounding like the respondent, "
    "prioritize sounding like the respondent.\n\n"

    "Privacy/controls: META fields are control variables for conditioning. "
    "Do not mention META unless the user explicitly asks about it.\n\n"

    "If STYLE EXAMPLES are provided, you may reuse their content as well as their phrasing, "
    "as long as the response remains consistent with the respondent’s overall style."
)


class JsonlHandler:
    def create_jsonl_for_supervised_fine_tuning(self):
        train_set = controller.train_data_path
        # validation_set = controller.validation_data_path
        subsets = [train_set] # don't need to do that for validation_set

        for subset in subsets:
            if subset == train_set:
                supervised_dataset_save_path = controller.language_model_supervised_fine_tune_train_path
            else:
                supervised_dataset_save_path = controller.language_model_supervised_fine_tune_validation_path

            participants = os.listdir(subset)
            with open(supervised_dataset_save_path, "w", encoding="utf-8") as f1:
                for participant in participants:
                    current_path = os.path.join(subset, participant)
                    transcripts_path = [str(p) for p in Path(current_path).glob("*.json")]
                    general_info_path = os.path.join(current_path, 'general_info.json')
                    transcripts_path.remove(str(general_info_path))

                    with open(general_info_path, "r", encoding="utf-8") as f2:
                        general_info = json.load(f2)
                    age = general_info['age']
                    gender = general_info['gender']

                    for transcript_path in transcripts_path:
                        with open(transcript_path, "r", encoding="utf-8") as f3:
                            data = json.load(f3)

                        id = data['id']
                        date = data['date']
                        topic = data['topic']
                        moderator_transcript = data['moderator_transcript']
                        participant_transcript = data['participant_transcript']

                        if int(list(moderator_transcript.keys())[0]) > int(list(participant_transcript.keys())[0]): # participant cannot start
                            del participant_transcript[list(participant_transcript.keys())[0]]

                        if int(list(moderator_transcript.keys())[-1]) > int(list(participant_transcript.keys())[-1]): # moderator cannot finish
                            del moderator_transcript[list(moderator_transcript.keys())[-1]]

                        moderator_transcript_keys = list(moderator_transcript.keys())
                        for moderator_key in moderator_transcript_keys:
                            participant_key = str(int(moderator_key) + 1)
                            question = moderator_transcript[moderator_key]['text']
                            answer = participant_transcript[participant_key]['text']

                            system = {"role": "system", "content": SYSTEM_PROMPT}
                            user = {"role": "user", "content": f"PERSON_ID: {id}\nMETA: age={age} | gender={gender} | "
                                                               f"date_of_interview={date} | topic={topic}\nQUESTION: {question}"}
                            assistant = {"role": "assistant", "content": answer}

                            messages = [system, user, assistant]
                            sample = {"messages": messages}

                            f1.write(json.dumps(sample, ensure_ascii=False) + "\n")


    def create_jsonl_for_direct_preference_optimization(self):
        def generate_dissimilar_responses(client, question: str, real_answer: str, num_dissimilar_responses: int = 3,
                                          model: str = "gpt-4.1-mini", temperature: float = 1.2, max_tokens: int = 400
                                          ) -> List[str]:
            """
            Generate N responses that are deliberately dissimilar in style
            from the real person's answer. Intended for DPO negatives.
            """

            system_prompt = (
                "You are generating alternative responses for preference learning (DPO).\n"
                "Your goal is to produce answers that are PLAUSIBLE but STYLISTICALLY DIFFERENT "
                "from the real respondent.\n\n"
                "You should add TEMPO and PAUSE metadata as well, where TEMPO includes one of "
                "<TEMPO_SLOW>: words_per_second <= 2, "
                "<TEMPO_MED>: 2 < words_per_second <= 3, "
                "<TEMPO_FAST>: words_per_second > 3, "
                "and PAUSE includes one of "
                "<PAUSE_NONE>: silence_time < 0.2 second, "
                "<PAUSE_SHORT>: 0.2 second <= silence_time < 0.8 second, "
                "<PAUSE_MED>: 0.8 second <= silence_time < 1.8 second, "
                "<PAUSE_LONG>: 1.8 second <= silence_time < 3.5 second, "
                "<PAUSE_VLONG>: silence_time >= 3.5 second."
                "Keep in mind you must at least add one <TEMPO_*> at the end of each answer, but "
                "for in the middle either use both <TEMPO_*> <PAUSE_*> or none. You can add more than "
                "one pair in longer sentences as well."
                
                "The real respondent’s answers may contain paralinguistic tokens "
                "such as <TEMPO_*> and <PAUSE_*> that describe speaking behavior.\n"
                "You MAY use these tokens, but use them in a way that reflects a "
                "more assistant-like, formal, or instructional speaking style "
                "(e.g., faster tempo, fewer or shorter pauses, uniform pacing).\n\n"

                "Avoid:\n"
                "- matching the respondent’s tone, hedging, or conversational pacing\n"
                "- mirroring the respondent’s hesitation patterns\n"
                "- first-person personal storytelling\n\n"

                "Prefer:\n"
                "- formal or instructional tone\n"
                "- generic or abstract advice\n"
                "- structured phrasing\n"
                "- more uniform or different tempo than the respondent\n\n"

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

        with open(controller.api_key_path, 'r') as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)

        train_set = controller.train_data_path
        # validation_set = controller.validation_data_path
        subsets = [train_set] # don't need to do that for validation_set

        for subset in subsets:
            if subset == train_set:
                dpo_dataset_save_path = controller.language_model_direct_preference_optimization_train_path
            else:
                dpo_dataset_save_path = controller.language_model_direct_preference_optimization_validation_path

            participants = os.listdir(subset)
            with open(dpo_dataset_save_path, "w", encoding="utf-8") as f1:
                for participant in participants:
                    current_path = os.path.join(subset, participant)
                    transcripts_path = [str(p) for p in Path(current_path).glob("*.json")]
                    general_info_path = os.path.join(current_path, 'general_info.json')
                    transcripts_path.remove(str(general_info_path))

                    with open(general_info_path, "r", encoding="utf-8") as f2:
                        general_info = json.load(f2)
                    age = general_info['age']
                    gender = general_info['gender']

                    for transcript_path in transcripts_path:
                        with open(transcript_path, "r", encoding="utf-8") as f3:
                            data = json.load(f3)

                        id = data['id']
                        date = data['date']
                        topic = data['topic']
                        moderator_transcript = data['moderator_transcript']
                        participant_transcript = data['participant_transcript']

                        if int(list(moderator_transcript.keys())[0]) > int(list(participant_transcript.keys())[0]):  # participant cannot start
                            del participant_transcript[list(participant_transcript.keys())[0]]

                        if int(list(moderator_transcript.keys())[-1]) > int(list(participant_transcript.keys())[-1]):  # moderator cannot finish
                            del moderator_transcript[list(moderator_transcript.keys())[-1]]

                        moderator_transcript_keys = list(moderator_transcript.keys())
                        for moderator_key in moderator_transcript_keys:
                            if random.random() > 0.4:
                                continue
                            participant_key = str(int(moderator_key) + 1)
                            question = moderator_transcript[moderator_key]['text']
                            answer = participant_transcript[participant_key]['text']

                            fake_answers = generate_dissimilar_responses(client, question, answer, num_dissimilar_responses=4)

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
