import os, json, requests
from typing import Dict, Any, Tuple
from judge.normalize import normalize_1_to_5

from typing import Dict, Any


def build_prompt(row: Dict[str, Any]) -> str:
    base_cap = row["base_caption"]
    trf_cap = row["transformed_caption"]
    detections_list = row.get("yolo_topk", [])
    base_object_classes = row.get("base_image_object_classes", [])

    sa = row.get("sa", {})
    sa_type = sa.get("sa_type", 0)

    allowed_new_class = None
    removed_class = None
    target_class = None

    if sa_type == 1:
        inserted_class = sa.get("inserted_class_name", "UNKNOWN")
        allowed_new_class = inserted_class
        mr_text = f"INSERT object class: {inserted_class}"
        expected_change_text = (
            f"An object of class '{inserted_class}' was inserted into the image. "
            f"The transformed caption should mention this inserted object, or clearly refer to it using a close synonym, "
            f"near-equivalent term, or natural description."
        )

    elif sa_type == 2:
        removed_class = sa.get("matched_base_cls_name", sa.get("removed_cls_name", "UNKNOWN"))
        mr_text = f"REMOVE target object class: {removed_class}"
        expected_change_text = (
            f"An object of class '{removed_class}' was removed from the image. "
            f"The transformed caption should stop mentioning this removed object, unless other instances of the same class "
            f"still remain in the scene."
        )

    elif sa_type == 3:
        old_class = sa.get("matched_base_cls_name", sa.get("replaced_cls_name", "UNKNOWN"))
        new_class = sa.get("replacement_class_name", "UNKNOWN")
        allowed_new_class = new_class
        mr_text = f"REPLACE target {old_class} WITH {new_class}"
        expected_change_text = (
            f"An object of class '{old_class}' was replaced with an object of class '{new_class}'. "
            f"The transformed caption should reflect the presence of '{new_class}'. "
            f"If other instances of '{old_class}' still remain, it is acceptable for '{old_class}' to still be mentioned, "
            f"but the caption should also mention '{new_class}' or a close synonym / near-equivalent term."
        )

    elif sa_type == 4:
        target_class = sa.get("matched_base_cls_name", sa.get("target_cls_name", "UNKNOWN"))
        mr_text = f"LOCAL_SP on target object {target_class}"
        expected_change_text = (
            f"An object of class '{target_class}' was locally transformed. "
            f"Only that object was modified, not the whole image. "
            f"If the local transformation causes a clear visible semantic change to that object, the transformed caption should reflect it. "
            f"If the change is only low-level and does not create a meaningful semantic difference, it is acceptable for the caption to remain semantically similar."
        )

    elif sa_type == 5:
        target_class = sa.get("matched_base_cls_name", sa.get("target_cls_name", "UNKNOWN"))
        scale_factor = sa.get("obj_scale_factor", None)

        if scale_factor is not None:
            if scale_factor > 1.0:
                scale_text = "became bigger"
            elif scale_factor < 1.0:
                scale_text = "became smaller"
            else:
                scale_text = "kept the same size"
        else:
            scale_text = "changed scale"

        mr_text = f"SCALE_OBJECT target {target_class}"
        expected_change_text = (
            f"An object of class '{target_class}' had its size changed in the image. "
            f"In this case, the object {scale_text}. "
            f"If this size change is clearly visible and semantically relevant, the transformed caption should reflect it. "
            f"If the size change is not clearly meaningful at caption level, it is acceptable for the caption to remain semantically similar."
        )

    else:
        mr_text = "SP (semantic-preserving) transform only"
        expected_change_text = (
            "No semantic object-level change was applied. "
            "The transformed caption should preserve the same meaning as the base caption."
        )

    prompt = f"""
You are a strict evaluator for metamorphic testing of vision-language captions.

Your job is to judge whether the transformed caption is correct under the applied metamorphic relation (MR).

TOP-PRIORITY RULES:
1. Judge the transformed caption against the semantics of the TRANSFORMED image, not only the base image.
2. For semantic-altering MRs, the transformed-image semantics are:
   - objects originally present in the base image
   PLUS
   - objects explicitly introduced by the MR
   MINUS
   - objects explicitly removed by the MR
3. Therefore:
   - for sa_type == 1, the inserted object is expected to newly appear; mentioning it is correct and MUST NOT be classified as hallucination.
   - for sa_type == 3, the replacement object is expected to newly appear; mentioning it is correct and MUST NOT be classified as hallucination.
   - for sa_type == 2, the removed object should no longer be mentioned unless other instances of that same class still remain.
4. Do not classify an object as hallucination only because it is absent from the base-image object list.
5. An object is hallucinated only if it is unsupported by both:
   - the base image content, and
   - the semantic change introduced by the MR.

OBJECT-CLASS INTERPRETATION RULES:
The object class names come from an object detector and may be coarse labels.
Treat more specific natural words as valid realizations of a broader class.

Examples:
- "person" may appear as: man, woman, child, boy, girl, people
- "car" may appear as: vehicle, automobile, sedan, truck
- "bowl" may appear as: dish, plate, container

When evaluating:
- Use semantic reasoning, not exact word matching.
- A close synonym, subtype, broader/narrower natural description, or near-equivalent term is acceptable.
- Do NOT mark a caption wrong just because it uses a more natural or more specific word.

MR-SPECIFIC JUDGMENT RULES:
- sa_type == 0:
  No semantic object-level change was applied. The transformed caption should preserve the same meaning as the base caption.
- sa_type == 1 (insert):
  The inserted object is expected to appear in the transformed caption. If it is not mentioned, that is an omission failure.
  Mentioning the inserted object is evidence of correctness, not hallucination.
- sa_type == 2 (remove):
  The removed object should disappear from the transformed caption, unless other instances of the same class still remain.
- sa_type == 3 (replace):
  The new replacement object is expected to appear in the transformed caption.
  If the caption only keeps mentioning the old object and ignores the new one, that is a failure.
  Mentioning the new object is not hallucination.
- sa_type == 4 (local object transformation):
  If a clear visible semantic change happened to the target object, the transformed caption should reflect it.
  If no meaningful semantic change is visible at caption level, keeping similar meaning is acceptable.
- sa_type == 5 (object scale change):
  If a clear and meaningful size change is visible, the transformed caption should reflect it.
  If the size change is not semantically important at caption level, keeping similar meaning is acceptable.

FAILURE TYPES:
- omission: an expected object/change is missing from the transformed caption
- misclassification: the caption refers to the object/change using the wrong class
- quantity: the count or number is wrong
- hallucination: the caption introduces an unsupported object/attribute not grounded in either the base image or the MR-induced change
- substitution: the caption swaps one object/class for another incorrect one
- no-change: the caption stays essentially unchanged when a clear semantic change should have been reflected
- others: a small custom label if needed

IMPORTANT INTERPRETATION NOTES:
- Base-image object classes are grounding evidence for what was originally present.
- They are NOT the full truth for semantic-altering MRs.
- For insert and replace MRs, newly introduced objects are valid even if they were absent from the base-image object list.
- For remove MRs, continued mention of the removed object is only acceptable if multiple instances of that class still remain.
- For local transformation and scale-change MRs, only count it as failure when the applied change is clearly visible and semantically meaningful for captioning.

Base caption:
\"\"\"{base_cap}\"\"\"

Transformed caption:
\"\"\"{trf_cap}\"\"\"

Metamorphic relation applied:
- {mr_text}

Expected semantic change:
- {expected_change_text}

Base image detected object classes:
- {base_object_classes}

Base-image detections:
{detections_list}

Allowed newly appearing object class under this MR:
- {allowed_new_class if allowed_new_class is not None else "none"}

Decide whether the transformed caption is a FAILURE under the MR.

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
'''
def call_hf(model_url: str, api_key: str, prompt: str, model_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model_id, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 220}
    resp = requests.post(model_url, headers=headers, json=payload, timeout=60) #timeout 180 for 70B model, 60 for 8B model
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
'''


def call_hf(model_url: str, api_key: str, prompt: str, model_id: str) -> Dict[str, Any]:
    print(f"LLM used: {model_id}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 220
    }

    resp = requests.post(model_url, headers=headers, json=payload, timeout=180)

    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code}: {resp.text}")

    text = resp.json()["choices"][0]["message"]["content"]

    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e < s:
        return {
            "score_1_to_5": 3,
            "is_failure": None,
            "failure_types": [],
            "explanation": "No JSON found"
        }

    try:
        return json.loads(text[s:e+1])
    except Exception:
        return {
            "score_1_to_5": 3,
            "is_failure": None,
            "failure_types": [],
            "explanation": "Invalid JSON"
        }



def judge_score_row(row: Dict[str, Any], llm_cfg: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
    model_url = llm_cfg["model_url"]
    api_key = os.getenv(llm_cfg.get("api_key_env", "HF_TOKEN"), "")
    model_id = llm_cfg.get("judge_model", "meta-llama/Llama-3.1-8B-Instruct")
    print("LLM used:", model_id)
    prompt = build_prompt(row)
    #out = call_hf(model_url, api_key, prompt, model_id)
    out = call_hf(
        model_url="https://router.huggingface.co/v1/chat/completions",
        api_key=api_key,
        prompt=prompt,
        model_id="google/gemma-2-27b-it"
    )
    
    score = float(out.get("score_1_to_5", 3))
    score = max(1.0, min(5.0, score)) # clamp to [1,5] in case of invalid LLM outputs
    norm_wrongness= normalize_1_to_5(score)  # 0=correct, 1=wrong
    return score, norm_wrongness, out