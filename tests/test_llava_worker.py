import base64
import json
import requests

img_path = "/home/ubuntu/last_version/train2017/000000000025.jpg"

with open(img_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "llava-v1.5-7b",
    "prompt": "<image>\nDescribe this image in one sentence.",
    "images": [img_b64],
    "temperature": 0.2,
    "top_p": 0.9,
    "max_new_tokens": 128,
    "stop": "</s>"
}

resp = requests.post(
    "http://127.0.0.1:40000/worker_generate_stream",
    json=payload,
    timeout=300,
    stream=True,
)

print("STATUS:", resp.status_code)

last_obj = None
for chunk in resp.iter_lines(decode_unicode=False, delimiter=b"\0"):
    if not chunk:
        continue
    obj = json.loads(chunk.decode("utf-8"))
    last_obj = obj
    print(obj)

print("\nFINAL TEXT:")
print(last_obj["text"] if last_obj else "NO RESPONSE")