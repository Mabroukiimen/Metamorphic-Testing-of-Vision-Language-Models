import base64
import json
import re
from pathlib import Path
from typing import Dict, Optional

import requests

CAPTION_PROMPT = "Describe this image in one sentence."


class VLMRunner:
    def __init__(
        self,
        model_name: str,
        backend: str = "api",
        worker_url: Optional[str] = None,
        api_model_name: Optional[str] = None,
        generation: Optional[Dict] = None,
        request_timeout: int = 300,
    ):
        self.model_name = model_name
        self.backend = backend
        self.worker_url = worker_url
        self.api_model_name = api_model_name or model_name
        self.request_timeout = request_timeout
        self.generation = generation or {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_new_tokens": 128,
        }

        if self.backend != "api":
            raise ValueError("Expected backend='api'")
        if not self.worker_url:
            raise ValueError("worker_url is required")

    def _encode_image_base64(self, image_path: str) -> str:
        return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")

    def _clean_text(self, text: str, prompt: str) -> str:
        text = text.strip()

        prefix = f"<image>\n{prompt}".strip()
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

        text = re.sub(r"^\s*ASSISTANT:\s*", "", text).strip()
        return text

    def caption(self, image_path: str, prompt: str = CAPTION_PROMPT) -> str:
        payload = {
            "model": self.api_model_name,
            "prompt": f"<image>\n{prompt}",
            "images": [self._encode_image_base64(image_path)],
            "temperature": self.generation.get("temperature", 0.2),
            "top_p": self.generation.get("top_p", 0.9),
            "max_new_tokens": self.generation.get("max_new_tokens", 128),
            "stop": "</s>",
        }

        resp = requests.post(
            self.worker_url,
            json=payload,
            timeout=self.request_timeout,
            stream=True,
        )
        resp.raise_for_status()

        last_obj = None
        for chunk in resp.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if not chunk:
                continue
            last_obj = json.loads(chunk.decode("utf-8"))

        if last_obj is None:
            raise RuntimeError("Empty response from LLaVA worker")

        if last_obj.get("error_code", 0) != 0:
            raise RuntimeError(f"LLaVA worker error: {last_obj}")

        return self._clean_text(last_obj["text"], prompt)