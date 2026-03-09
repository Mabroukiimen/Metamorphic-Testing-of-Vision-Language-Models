import os, json, requests
from typing import Dict, Any, Tuple
from judge.normalize import normalize_1_to_5

def build_prompt(row: Dict[str, Any]) -> str:
    base_cap = row["base_caption"]
    trf_cap  = row["transformed_caption"]
    sa_type  = row.get("sa", {}).get("sa_type", 0)

    sa = row.get("sa", {})
    inserted = sa.get("inserted", {})
    target   = sa.get("target_det", {})
    replaced = sa.get("replaced_with", {})

    if sa_type == 1:
        mr_text = f"INSERT object class: {inserted.get('class_name','UNKNOWN')}"
    elif sa_type == 2:
        mr_text = f"REMOVE target object class: {target.get('cls_name','UNKNOWN')}"
    elif sa_type == 3:
        mr_text = f"REPLACE target {target.get('cls_name','UNKNOWN')} WITH {replaced.get('class_name','UNKNOWN')}"
    else:
        mr_text = "SP (semantic-preserving) transform only; caption should keep same meaning"

    prompt = f"""
You are a strict evaluator for metamorphic testing of vision-language captions.

Metamorphic relation applied:
- {mr_text}

Base caption:
\"\"\"{base_cap}\"\"\"

Transformed caption:
\"\"\"{trf_cap}\"\"\"

Return ONLY valid JSON (no markdown, no extra text) with this schema:
{{
  "score_1_to_5": 2.5,  # float, 1=worst, 5=best; for SP transform, higher is better; for MR with object change, 1 means the change is fully reflected in the caption and 5 means the change is not reflected at all; partial reflection can be scored in between
  "is_failure": true,
  "failure_types": ["omission"],
  "explanation": "..."
}}
""".strip()
    return prompt

def call_hf(model_url: str, api_key: str, prompt: str, model_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model_id, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 220}
    resp = requests.post(model_url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code}: {resp.text}")
    text = resp.json()["choices"][0]["message"]["content"]
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        return {"score_1_to_5": 3, "is_failure": None, "failure_types": [], "explanation": "No JSON found"}
    try:
        return json.loads(text[s:e+1])
    except Exception:
        return {"score_1_to_5": 3, "is_failure": None, "failure_types": [], "explanation": "Invalid JSON"}

def judge_score_row(row: Dict[str, Any], llm_cfg: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
    model_url = llm_cfg["model_url"]
    api_key = os.getenv(llm_cfg.get("api_key_env", "HF_TOKEN"), "")
    model_id = llm_cfg.get("judge_model", "meta-llama/Llama-3.1-8B-Instruct")
    prompt = build_prompt(row)
    out = call_hf(model_url, api_key, prompt, model_id)
    score = float(out.get("score_1_to_5", 3))
    norm_wrongness= normalize_1_to_5(score)  # 0=correct, 1=wrong
    return score, norm_wrongness, out