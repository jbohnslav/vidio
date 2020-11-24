import os
from typing import Union

import cv2

COLORSPACES = ['BGR', 'RGB', 'GRAY', 'YUV420']


def convert_colorspace(frame, in_color: str, out_color: str):
    assert in_color in COLORSPACES
    assert out_color in COLORSPACES
    if in_color == out_color:
        return frame
    if in_color == 'BGR':
        if out_color == 'YUV420':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        elif out_color == 'RGB':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif out_color == 'GRAY':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    elif in_color == 'RGB':
        if out_color == 'YUV420':
            return cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420)
        elif out_color == 'BGR':
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif out_color == 'GRAY':
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    elif in_color == 'GRAY':
        if out_color == 'YUV420':
            return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2YUV_I420)
        elif out_color == 'RGB':
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif out_color == 'BGR':
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    elif in_color == 'YUV420':
        if out_color == 'GRAY':
            return cv2.cvtColor(frame, cv2.COLOR_YUV2GRAY_I420)
        elif out_color == 'RGB':
            return cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
        elif out_color == 'BGR':
            return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)