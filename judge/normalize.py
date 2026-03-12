def normalize_1_to_5(score_1_to_5: float) -> float:
    # 1 -> 0.0, 5 -> 1.0
    s = score_1_to_5
    return (s - 1.0) / 4.0