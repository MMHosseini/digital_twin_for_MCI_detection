import time
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from openai import OpenAI
from control_panel import ControlPanel

controller = ControlPanel()


class PersonaFineTuner:
    def __init__(self):
        with open(controller.api_key_path, 'r') as f:
            api_key = f.read().strip()
        self.client = OpenAI(api_key=api_key)

    def supervised_fine_tuning(self):
        sft_model_name = self._train(base_model="gpt-4.1-mini-2025-04-14", method="supervised", n_epochs=3,
                               train_path=controller.language_model_supervised_fine_tune_train_path)
        supervised_fine_tuned_model_name = controller.supervised_fine_tuned_model_name
        with open(supervised_fine_tuned_model_name, 'w') as f:
            f.write(sft_model_name) # ft:gpt-4.1-mini-2025-04-14:personal::D3p5VSZ7

    def dpo_fine_tuning(self):
        supervised_fine_tuned_model_name = controller.supervised_fine_tuned_model_name
        with open(supervised_fine_tuned_model_name, 'r') as f:
            sft_model_name = f.read().strip()
        dpo_model_name = self._train(base_model = sft_model_name, method = "dpo", n_epochs = 2,
                               train_path=controller.language_model_direct_preference_optimization_train_path)
        dpo_fine_tuned_model_name = controller.dpo_fine_tuned_model_name
        with open(dpo_fine_tuned_model_name, 'w') as f:
            f.write(dpo_model_name)

    def inference_language_model(self): # simulate validation responses by raw GPT-4.1, fine-tuned, dpo, and rag
        raw_model = "gpt-4.1-mini-2025-04-14"
        self._inference(model=raw_model, key_name='simulated_transcript_raw_gpt', rag=False)
        
        with open(controller.supervised_fine_tuned_model_name) as f:
            sft_model = f.read().strip()
        self._inference(model=sft_model, key_name='simulated_transcript_sft', rag=False)

        with open(controller.dpo_fine_tuned_model_name) as f:
            dpo_model = f.read().strip()
        self._inference(model=dpo_model, key_name='simulated_transcript_dpo', rag=False)

        with open(controller.dpo_fine_tuned_model_name) as f:
            dpo_model = f.read().strip()
        self._inference(model=dpo_model, key_name='simulated_transcript_rag', rag=True)

    # -------------------------
    # TRAINING
    # -------------------------
    def _train(self, train_path: str, base_model: str, method: str = "supervised",  # "supervised" or "dpo"
            val_path: Optional[str] = None, suffix: Optional[str] = None, n_epochs: Optional[int] = None,
            extra_method_config: Optional[Dict[str, Any]] = None) -> str:

        method = method.strip().lower()
        if method not in {"supervised", "dpo"}:
            raise ValueError("method must be 'supervised' or 'dpo'")

        # 1) Upload files (purpose="fine-tune") :contentReference[oaicite:5]{index=5}
        train_file = self.client.files.create(file=open(train_path, "rb"), purpose="fine-tune")
        val_file_id = None
        if val_path:
            val_file = self.client.files.create(file=open(val_path, "rb"), purpose="fine-tune")
            val_file_id = val_file.id

        # 2) Create job :contentReference[oaicite:6]{index=6}
        method_payload: Dict[str, Any] = {"type": method}

        if method == "supervised":
            # Optional hyperparameters, per best practices page :contentReference[oaicite:7]{index=7}
            if n_epochs is not None:
                method_payload["supervised"] = {"hyperparameters": {"n_epochs": int(n_epochs)}}

        # Merge any extra method config (e.g., DPO hyperparameters)
        if extra_method_config:
            # Shallow merge is usually sufficient for small configs
            for k, v in extra_method_config.items():
                method_payload[k] = v

        job_kwargs: Dict[str, Any] = {
            "model": base_model,
            "training_file": train_file.id,
            "method": method_payload,
        }
        if val_file_id:
            job_kwargs["validation_file"] = val_file_id
        if suffix:
            job_kwargs["suffix"] = suffix

        job = self.client.fine_tuning.jobs.create(**job_kwargs)

        # 3) Poll until done, streaming events optionally
        job_id = job.id
        start = time.time()
        seen_event_ids = set()

        '''
        timeout_sec : int
            Max time to wait.
        print_events : bool
            If True, prints new fine-tuning events while polling.
        '''
        poll_interval_sec = 30
        timeout_sec = 6 * 60 * 60
        print_events = True
        while True:
            if time.time() - start > timeout_sec:
                raise RuntimeError(f"Fine-tuning job {job_id} timed out after {timeout_sec} seconds")

            current = self.client.fine_tuning.jobs.retrieve(job_id)

            if print_events:
                try:
                    ev = self.client.fine_tuning.jobs.list_events(job_id)
                    # Print only new events
                    for e in reversed(ev.data):
                        # Some SDKs expose an id; if not, fallback to (created_at, message)
                        eid = getattr(e, "id", None) or (getattr(e, "created_at", None), getattr(e, "message", None))
                        if eid in seen_event_ids:
                            continue
                        seen_event_ids.add(eid)
                        ts = getattr(e, "created_at", "")
                        msg = getattr(e, "message", "")
                        print(f"[{ts}] {msg}")
                except Exception:
                    # Events are helpful but non-critical; ignore transient issues.
                    pass

            status = current.status
            print(f"Job {job_id} status: {status}")

            if status == "succeeded":
                ft_model = current.fine_tuned_model
                if not ft_model:
                    raise RuntimeError(f"Job {job_id} succeeded but fine_tuned_model is empty")
                return ft_model

            if status in {"failed", "cancelled"}:
                raise RuntimeError(f"Fine-tuning job {job_id} ended with status={status}")

            time.sleep(poll_interval_sec)

    # -------------------------
    # INFERENCE
    # -------------------------
    def infer(self, model: str, system_prompt: str, user_text: str, max_output_tokens: int = 400, temperature: float = 0.0) -> str:
        resp = self.client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=float(temperature),
            max_output_tokens=int(max_output_tokens),
        )
        return resp.output_text

    def _inference(self, model, key_name, rag=False):
        system_prompt = (
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
            
            "Output formatting constraints for tokens:\n"
            "- Allowed tokens: <TEMPO_*> and <PAUSE_*>\n"
            "- Sentence-initial tokens are forbidden: a sentence must not start with '<TEMPO_' or '<PAUSE_'.\n"
            "- Tokens may appear only after at least one character of a sentence, either mid-sentence or sentence-final.\n"
            "- If multiple sentences are produced: for sentences 1..N-1, you may use <TEMPO_*><PAUSE_*>; "
            "for sentence N (the last sentence), use <TEMPO_*> only and do not output any <PAUSE_*> after it.\n\n"
            "Examples:\n"
            "- Sentence 1,<TEMPO_><PAUSE_> sentence 2.<TEMPO_><PAUSE_> Sentence 3.<TEMPO_><PAUSE_> Sentence 4.<TEMPO_>\n"
            "- Sentence 1<TEMPO_>\n"
            "- Sentence 1.<TEMPO_><PAUSE_> sentence 2,<TEMPO_><PAUSE_> Sentence 3.<TEMPO_>\n"
            "- Sentence 1.<TEMPO_><PAUSE_> sentence 2.<TEMPO_>\n"

        )

        validation_path = controller.validation_data_path
        participants = os.listdir(validation_path)
        for participant in participants:
            current_path = os.path.join(validation_path, participant)
            transcripts_name = os.listdir(current_path)
            general_info_path = os.path.join(current_path, 'general_info.json')

            with open(general_info_path, "r", encoding="utf-8") as f1:
                info = json.load(f1)
            id = info['id']
            age = info['age']
            gender = info['gender']

            transcripts_name.remove('general_info.json')
            for transcript_name in transcripts_name:
                if not transcript_name.endswith('.json'):
                    continue
                transcript_path = os.path.join(current_path, transcript_name)
                with open(transcript_path, "r", encoding="utf-8") as f2:
                    data = json.load(f2)
                moderator_transcript = data['moderator_transcript']
                date = data['date']
                topic = data['topic']
                moderator_transcript_keys = list(moderator_transcript.keys())
                response_dict = dict()
                for moderator_key in moderator_transcript_keys:
                    question = moderator_transcript[moderator_key]['text']
                    user_text = f"PERSON_ID: {id} META: age={age} | gender={gender} | date_of_interview={date} | topic={topic} QUESTION: {question}"

                    if rag is True:
                        pass # later add some examples to user_text

                    resp = self.client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_text},
                        ],
                        temperature=float(0.1),
                        max_output_tokens=int(400),
                    )

                    generated_response = resp.output_text
                    new_info = {'text': generated_response}
                    response_dict[str(int(moderator_key) + 1)] = new_info

                data[key_name] = response_dict
                with open(transcript_path, 'w') as f3:
                    json.dump(data, f3, indent=4)
