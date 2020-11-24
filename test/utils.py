import cv2
import numpy as np



def make_numbered_image(num: int):
    num_formatted = '{:07d}'.format(num)
    font = cv2.FONT_HERSHEY_SIMPLEX
    im = np.zeros((224,224,3), dtype=np.uint8)+255
    cv2.putText(im, num_formatted, (10,125), font, 1.5, (0, 0, 0) , 2, cv2.LINE_AA)
    return im



