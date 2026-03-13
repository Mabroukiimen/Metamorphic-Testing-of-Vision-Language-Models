import os, json, requests
from typing import Dict, Any, Tuple
from judge.normalize import normalize_1_to_5


def build_prompt(row: Dict[str, Any]) -> str:
    base_cap = row["base_caption"]
    trf_cap  = row["transformed_caption"]
    
    base_object_classes = row.get("base_image_object_classes", [])
    allowed_new_class = None

    sa = row.get("sa", {})
    sa_type = sa.get("sa_type", 0)
    
    
    if sa_type == 1:
        inserted_class = sa.get("inserted_class_name", "UNKNOWN")
        allowed_new_class = inserted_class
        mr_text = f"INSERT object class: {inserted_class}"
        expected_change_text = (
        f"An object of class '{inserted_class}' was inserted. "
        f"The transformed caption should reflect this inserted object or a close synonym / near-equivalent description or clearly reflect its presence."
    )
        
    elif sa_type == 2:
        removed_class = sa.get("matched_base_cls_name", sa.get("removed_cls_name", "UNKNOWN"))
        allowed_new_class = None
        mr_text = f"REMOVE target object class: {removed_class}"
        expected_change_text = (
            f"An object of class '{removed_class}' was removed. "
            f"The transformed caption should stop mentioning this removed object, unless it existed in plural form before."
    )
        
    elif sa_type == 3:
        old_class = sa.get("matched_base_cls_name", sa.get("replaced_cls_name", "UNKNOWN"))
        new_class = sa.get("replacement_class_name", "UNKNOWN")
        allowed_new_class = new_class
        mr_text = f"REPLACE target {old_class} WITH {new_class}"
        expected_change_text = (
            f"An object of class '{old_class}' was replaced with an object of class '{new_class}'. "
            f"The transformed caption should reflect '{new_class}' rather than '{old_class}'. If the object of class '{old_class}' existed more than one time (plural form), then it's correct to still be mentioned, however, the caption should also mention '{new_class}'."
    )
        
    else:
        allowed_new_class = None
        mr_text = "SP (semantic-preserving) transform only; caption should keep same meaning"
        expected_change_text = (
        "No semantic object-level change was applied. "
        "The transformed caption should preserve the same meaning as the base caption."
    )

    prompt = f"""
You are a strict evaluator for metamorphic testing of vision-language captions.

Metamorphic relation applied:
- {mr_text}

Expected semantic change:
- {expected_change_text}

Base image detected object classes:
- {base_object_classes}

Allowed newly appearing object class under this MR:
- {allowed_new_class if allowed_new_class is not None else "none"}

Object-class interpretation rules:

The object class names come from an object detector (COCO-style categories).
These categories are coarse labels.

Examples:
- "person" may appear in captions as: man, woman, child, boy, girl, people, person.
- "car" may appear as: vehicle, automobile, sedan, truck.
- "bowl" may appear as: dish, plate, container.

When evaluating captions:
- Treat specific descriptions as valid instances of the broader class.
- Do NOT mark it as misclassification if the caption uses a more specific description of the same class.

Example:
If the expected class is "person", captions mentioning "woman", "man", "girl", or "boy" are correct.

Base image detected object classes:
- {base_object_classes}

Allowed newly appearing object class under this MR:
- {allowed_new_class if allowed_new_class is not None else "none"}

Base caption:
\"\"\"{base_cap}\"\"\"

Transformed caption:
\"\"\"{trf_cap}\"\"\"

Decide whether the transformed caption is a FAILURE under the MR.
Use semantic reasoning rather than exact word matching.
Judge based on meaning, not literal class label equality.

Use these failure types:
- omission
- misclassification
- quantity
- hallucination
- substitution
- no-change

Guidance:
- omission: the expected inserted/replaced object was not mentioned in the transformed caption
- misclassification: the object is mentioned but with the wrong class
- quantity: the count/number is wrong
- hallucination: an unsupported object or attribute appears; in particular, if the transformed caption mentions an object class not present in the base-image object list and that was not explicitly inserted or introduced by replacement under the metamorphic relation, then classify this as hallucination, unless it is clearly just a paraphrase, near-synonym, subtype or natural description of an existing object.
- substitution: the caption swaps one object/class for another incorrect one
- no-change: the caption stays essentially unchanged when a semantic change was expected

Interpretation:
- Treat the base-image object list as grounding evidence for which object classes are originally present. If the transformed caption mentions an object class that is not present in the base-image object list, and that object class was not explicitly inserted or introduced by replacement under the metamorphic relation, then classify this as hallucination, unless it is clearly just a paraphrase or near-synonym of an existing object.
- Do not require exact class-name matching.
- If the transformed caption uses a close synonym, near-equivalent object name, or a reasonable broader/narrower term for the expected object, do not mark it as a failure for omission or misclassification.
- Examples: bowl ~ dish ~ plate in some contexts; person ~ man; car ~ vehicle.
- For semantic-preserving transforms, failure means the meaning changed unexpectedly.
- For semantic-altering transforms, failure means the expected semantic change was not reflected correctly.

IMPORTANT:
Scoring:
- 1 = correct caption under the MR
- 2 = minor issue
- 3 = moderate issue
- 4 = strong failure
- 5 = severe failure

Return ONLY valid JSON.
Do NOT write any text outside JSON.
Do NOT use markdown.

Return exactly this JSON schema:
{{
    "score_1_to_5": 1,
    "is_failure": false,
    "failure_types": [],
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
    score = max(1.0, min(5.0, score)) # clamp to [1,5] in case of invalid LLM outputs
    norm_wrongness= normalize_1_to_5(score)  # 0=correct, 1=wrong
    return score, norm_wrongness, out