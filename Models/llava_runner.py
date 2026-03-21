import base64
import re
from pathlib import Path
from typing import Any, Dict, Optional

import requests

CAPTION_PROMPT = "Describe this image in one sentence."


class VLMRunner:
    def __init__(
        self,
        model_name: str,
        backend: str = "api",
        vlm_root: Optional[str] = None,
        python_cmd: Optional[str] = None,
        worker_url: Optional[str] = None,
        generation: Optional[Dict[str, Any]] = None,
        request_timeout: int = 300,
    ):
        self.model_name = model_name
        self.backend = backend
        self.vlm_root = vlm_root
        self.python_cmd = python_cmd
        self.worker_url = worker_url
        self.request_timeout = request_timeout
        self.generation = generation or {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_new_tokens": 128,
        }

        if self.backend != "api":
            raise ValueError("This runner now expects backend='api'")

        if not self.worker_url:
            raise ValueError("worker_url is required when backend='api'")

    def _encode_image_base64(self, image_path: str) -> str:
        return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")

    def _extract_text_from_response(self, data: Any) -> str:
        if isinstance(data, dict):
            if "text" in data and isinstance(data["text"], str):
                return data["text"].strip()

            if "output" in data and isinstance(data["output"], str):
                return data["output"].strip()

            if "response" in data and isinstance(data["response"], str):
                return data["response"].strip()

            if "detail" in data:
                raise ValueError(str(data["detail"]))

        if isinstance(data, str):
            return data.strip()

        raise ValueError(f"Unexpected worker response: {data}")

    def caption(self, image_path: str, prompt: str = CAPTION_PROMPT) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [self._encode_image_base64(image_path)],
            "temperature": self.generation.get("temperature", 0.2),
            "top_p": self.generation.get("top_p", 0.9),
            "max_new_tokens": self.generation.get("max_new_tokens", 128),
        }

        resp = requests.post(
            self.worker_url,
            json=payload,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()

        try:
            data = resp.json()
        except Exception:
            text = resp.text.strip()
            return re.sub(r"^\s*ASSISTANT:\s*", "", text).strip()

        text = self._extract_text_from_response(data)
        text = re.sub(r"^\s*ASSISTANT:\s*", "", text).strip()
        return text