from dataclasses import dataclass

@dataclass(frozen=True)
class Vec:
    B_BRIGHT = 0
    BRIGHT_FACTOR = 1
    B_BLUR = 2
    BLUR_RADIUS = 3
    B_CONTRAST = 4
    CONTRAST_FACTOR = 5
    B_SATURATION = 6
    SATURATION_FACTOR = 7
    B_NOISE = 8
    NOISE_STD = 9

    B_ROTATE = 10
    ROT_ANGLE = 11
    B_TRANSLATE = 12
    TX = 13
    TY = 14
    B_FLIP = 15
    B_SHEAR = 16
    SHEAR_X = 17
    SHEAR_Y = 18
    B_ZOOM = 19
    ZOOM_FACTOR = 20

    SA_TYPE = 21
    INS_CORPUS_ID = 22
    INS_SCALE = 23
    TARGET_DET_IDX = 24
    REP_CORPUS_ID = 25
    REP_SCALE = 26
    OBJ_SCALE_FACTOR = 27

    N = 28
    