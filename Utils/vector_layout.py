from dataclasses import dataclass

@dataclass(frozen=True)
class Vec:
    # indices
    B_BRIGHT = 0
    BRIGHT_FACTOR = 1
    B_BLUR = 2
    BLUR_RADIUS = 3
    B_ROTATE = 4
    ROT_ANGLE = 5
    B_TRANSLATE = 6
    TX = 7
    TY = 8

    SA_TYPE = 9
    TARGET_DET_IDX = 10
    INS_CORPUS_ID = 11
    INS_SCALE = 12
    REP_CORPUS_ID = 13
    REP_SCALE = 14