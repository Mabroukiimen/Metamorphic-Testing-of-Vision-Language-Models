import os, json, requests
from typing import Dict, Any, Tuple
from judge.normalize import normalize_1_to_5


def build_prompt(row: Dict[str, Any]) -> str:
    base_cap = row["base_caption"]
    trf_cap  = row["transformed_caption"]
    detections_list=row.get("yolo_topk", [])
    
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
            f"The transformed caption should reflect '{new_class}' rather than '{old_class}'. If the object of class '{old_class}' existed more than one time (plural form), then it's correct to still be mentioned, however, the caption should also mention '{new_class}' or a close synonym, near-equivalent object name, or a reasonable broader/narrower term for '{new_class}'."
    )
        
    elif sa_type == 4:
        target_class = sa.get("matched_base_cls_name", sa.get("target_cls_name", "UNKNOWN"))
        mr_text = f"LOCAL_SP on target object {target_class}"
        expected_change_text = (
            f"An object of class '{target_class}' was locally transformed in the image. "
            f"The transformation was applied only to that object, not to the whole image. "
            f"The transformed caption should reflect visible changes affecting that object if they are semantically noticeable. "
            f"If the object’s appearance changed in a meaningful way, the caption should mention those changes rather than describing the object exactly as before. "
            f"If the transformation is only low-level and does not produce a semantically meaningful visible difference, then it is acceptable for the caption to still describe the same object."
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
            f"An object of class '{target_class}' had its scale changed in the image. "
            f"In this case, the object {scale_text}. "
            f"The transformed caption should reflect that this object is larger or smaller if that size change is visually meaningful and relevant to the scene description. "
            f"If the caption describes the object exactly as before and ignores a clear size change, that should be considered a no-change failure."
            )
        
    else:
        allowed_new_class = None
        mr_text = "SP (semantic-preserving) transform only; caption should keep same meaning"
        expected_change_text = (
        "No semantic object-level change was applied. "
        "The transformed caption should preserve the same meaning as the base caption."
    )

    prompt = f"""
You are a strict evaluator for metamorphic testing of vision-language captions, depending on the Metamorphic relations applied, yould evaluate wether the transformed captions reflect those chnages or and signals any failures.
Ps: 
if the transformed caption mentions an object class that is not present in the base_object_classes list, and that object class was not explicitly inserted or introduced by replacement under the metamorphic relation, then classify this as hallucination failure, unless it is clearly just a paraphrase or near-synonym of an existing object.
In case of sa_type = 1, an object that was inserted, but the transformed caption did not mention it neither directly nor via a close synonym or near-equivalent description, that should be classified as omission failure; 
In case, of sa_type = 2, an object that was removed, but the transformed caption still mentioned it without acknowledging its removal, that should be classified as omission failure; unless that object was in plural form in the base caption (or there was more that one instance of that class in the detections_list)
In case of sa_type = 3, an object that was replaced with another, but the transformed caption still only mentions the original object without acknowledging the new one, that should be classified as an omission failure; 
In case of sa_type = 4, if an object had trasnformations applied on it then the transformed caption should describe those changes, otherwise that should be classified as omission failure; if the caption describes a clearly visible change to the object but describes it with the wrong class, that should be classified as misclassification failure; if the caption fails to reflect a clear semantic change that was applied to the object, that should be classified as omission failure; if the caption mentions a new object class that was not present in the base image and was not part of the applied metamorphic relation, that should be classified as hallucination failure, unless it is clearly just a paraphrase or near-synonym of an existing object.
In case of sa_type = 5, If the scale of an object was modified but the transformed caption still describes it as before without reflecting any visible change, that should be classified as omission failure; if the caption describes a clearly visible change to an object but describes it with the wrong class, that should be classified as misclassification failure; if the caption fails to reflect a clear semantic change that was applied to an object, that should be classified as omission failure; if the caption mentions a new object class that was not present in the base image and was not part of the applied metamorphic relation, that should be classified as hallucination failure, unless it is clearly just a paraphrase or near-synonym of an existing object.



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

-Objects detected in the image:
{detections_list}

Allowed newly appearing object class under this MR:
- {allowed_new_class if allowed_new_class is not None else "none"}

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