# Models/vlm_runner.py
import re
import pexpect

CAPTION_PROMPT = "Describe this image in one sentence."


class VLMRunner:
    def __init__(self, model_name: str, vlm_root: str, python_cmd: str):
        """
        model_name: e.g. "llava_v15"
        vlm_root: path to VLM repo (here LLaVA), e.g. "/home/ubuntu/LLaVA"
        python_cmd: python executable of llava env,
                    e.g. "/home/ubuntu/miniconda3/envs/llava/bin/python"
        """
        self.model_name = model_name
        self.vlm_root = vlm_root 
        self.python_cmd = python_cmd

    def _build_cmd(self, image_path: str) -> str:
        if self.model_name == "llava_v15":
            return (
                f'cd {self.vlm_root} && '
                f'{self.python_cmd} -m llava.serve.cli '
                f'--model-path liuhaotian/llava-v1.5-7b '
                f'--image-file "{image_path}" '
                f'--load-4bit'
            )
        raise ValueError(f"Unknown VLM model: {self.model_name}")

    def caption(self, image_path: str, prompt: str = CAPTION_PROMPT) -> str:
        cmd = self._build_cmd(image_path)
        child = pexpect.spawn('/bin/bash', ['-lc', cmd], encoding='utf-8', timeout=600)

        # Wait for model prompt
        child.expect_exact('USER:')
        child.sendline(prompt)

        # Read assistant answer
        child.expect_exact('ASSISTANT:')
        child.expect_exact('USER:')   # next turn

        ans = child.before.strip()
        ans = re.sub(r'^\s*ASSISTANT:\s*', '', ans).strip()

        try:
            child.sendline('exit')
            child.close()
        except Exception:
            pass

        return ans
