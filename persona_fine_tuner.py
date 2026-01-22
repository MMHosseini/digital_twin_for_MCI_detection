"""
Persona fine-tuning + inference helper (OpenAI)

Supports:
- SFT (supervised fine-tuning): method="supervised"
- DPO (direct preference optimization): method="dpo"

Docs:
- Fine-tuning API reference: https://platform.openai.com/docs/api-reference/fine-tuning :contentReference[oaicite:0]{index=0}
- SFT guide: https://platform.openai.com/docs/guides/supervised-fine-tuning :contentReference[oaicite:1]{index=1}
- DPO guide: https://platform.openai.com/docs/guides/direct-preference-optimization :contentReference[oaicite:2]{index=2}
- Files API: https://platform.openai.com/docs/api-reference/files/create :contentReference[oaicite:3]{index=3}

Usage:
  pip install --upgrade openai
  export OPENAI_API_KEY="..."

  trainer = PersonaFineTuner()
  ft_model = trainer.train(
      train_path="train.jsonl",
      val_path="val.jsonl",
      base_model="gpt-4.1-mini-2025-04-14",
      method="supervised",
      n_epochs=2
  )

  text = trainer.infer(
      model=ft_model,
      person_id="P0007",
      meta={"age":72,"gender":"F","race":"Asian","moca":23},
      question="How do you usually handle stress?",
  )
  print(text)
"""

import time
from typing import Any, Dict, Optional
from openai import OpenAI
from control_panel import ControlPanel

controller = ControlPanel()

with open(controller.api_key_path, 'r') as f:
    api_key = f.read()


class PersonaFineTuner:
    def __init__(self):
        self.client = OpenAI(api_key=api_key)

    # -------------------------
    # TRAINING
    # -------------------------
    def train(self, train_path: str, base_model: str, method: str = "supervised",  # "supervised" or "dpo"
            val_path: Optional[str] = None, suffix: Optional[str] = None, n_epochs: Optional[int] = None,
            extra_method_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Creates a fine-tuning job and waits for completion.

        Parameters
        ----------
        train_path : str
            Path to training .jsonl (SFT: {"messages":[...]} per line; DPO: preference format per guide).
        base_model : str
            Model to fine-tune (e.g., "gpt-4.1-mini-2025-04-14" or a previously fine-tuned model id).
        method : str
            "supervised" (SFT) or "dpo".
        val_path : Optional[str]
            Optional validation .jsonl.
        suffix : Optional[str]
            Optional suffix for the fine-tuned model name (if supported by your account).
        n_epochs : Optional[int]
            For supervised jobs, you can set epochs via method.supervised.hyperparameters.n_epochs. :contentReference[oaicite:4]{index=4}
        extra_method_config : Optional[dict]
            Extra method config merged into the request (e.g., DPO hyperparameters).

        Returns
        -------
        str
            Fine-tuned model id (e.g., "ft:gpt-4.1-mini-...") on success.

        Raises
        ------
        RuntimeError if job fails or times out.
        """
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
    def infer(self, model: str, person_id: str, meta: Dict[str, Any], question: str,
              system_prompt: Optional[str] = None, context: Optional[str] = None,
              max_output_tokens: int = 400, temperature: float = 0.7) -> str:
        """
        Runs inference with your fine-tuned model using the Responses API.

        Parameters
        ----------
        model : str
            Fine-tuned model id from train().
        person_id : str
            e.g., "P0007"
        meta : dict
            e.g., {"age":72,"gender":"F","race":"Asian","moca":23}
        question : str
            New question to answer in-persona.
        system_prompt : Optional[str]
            Optional override for system behavior. Keep consistent with training.
        context : Optional[str]
            Optional short prior dialog context (1–2 turns) if your deployment is conversational.
        """
        if system_prompt is None:
            system_prompt = (
                "You are simulating the respondent in a research dataset. "
                "Match the respondent's style (tone, verbosity, hedging, formality). "
                "Do NOT mention META fields unless the user explicitly asks about them. "
                "Do NOT invent personal facts; if unknown, answer in-character with uncertainty."
            )

        # Stable, consistent formatting (match your training format)
        meta_str = " | ".join(f"{k}={meta[k]}" for k in sorted(meta.keys()))
        user_parts = []
        if context:
            user_parts.append(f"CONTEXT:\n{context}\n")
        user_parts.append(f"PERSON_ID: {person_id}")
        user_parts.append(f"META: {meta_str}")
        user_parts.append(f"QUESTION: {question}")
        user_text = "\n".join(user_parts)

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



