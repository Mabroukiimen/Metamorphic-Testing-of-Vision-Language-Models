# Models/vlm_runner.py
import subprocess
from pathlib import Path


CAPTION_PROMPT = "Describe this image in one sentence."


class VLMRunner:
    def __init__(self, model_name: str, vlm_root: str, python_cmd: str, script_path: str = None):
        self.model_name = model_name
        self.vlm_root = vlm_root
        self.python_cmd = python_cmd
        self.script_path = script_path

    def caption(self, image_path: str, prompt: str = CAPTION_PROMPT) -> str:
        if self.model_name == "llava_v15":
            return self._caption_llava(image_path, prompt)

        if self.model_name == "blip_salesforce":
            return self._caption_blip(image_path, prompt)

        raise ValueError(f"Unknown VLM model: {self.model_name}")

    def _caption_llava(self, image_path: str, prompt: str) -> str:
        import re
        import pexpect

        cmd = (
            f'cd {self.vlm_root} && '
            f'{self.python_cmd} -m llava.serve.cli '
            f'--model-path liuhaotian/llava-v1.5-7b '
            f'--image-file "{image_path}" '
            f'--load-4bit'
        )

        child = pexpect.spawn('/bin/bash', ['-lc', cmd], encoding='utf-8', timeout=600)
        child.expect_exact('USER:')
        child.sendline(prompt)
        child.expect_exact('ASSISTANT:')
        child.expect_exact('USER:')
        ans = child.before.strip()
        ans = re.sub(r'^\s*ASSISTANT:\s*', '', ans).strip()

        try:
            child.sendline('exit')
            child.close()
        except Exception:
            pass

        return ans

    def _caption_blip(self, image_path: str, prompt: str) -> str:
        if not self.script_path:
            raise ValueError("script_path must be set for blip_salesforce")

        cmd = [
            self.python_cmd,
            self.script_path,
            "--image", str(image_path),
            "--prompt", str(prompt),
        ]

        result = subprocess.run(
            cmd,
            cwd=self.vlm_root,
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )

        return result.stdout.strip()