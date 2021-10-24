import os

import cv2
import numpy as np
from tensorflow import image

from prz.definitions.strings import Strings

class ImageSample:
    @staticmethod
    def read(img_path: str, mode:str='color'):
        cv_color = {
            'color': cv2.IMREAD_COLOR,
            'grayscalre': cv2.IMREAD_GRAYSCALE,
            'unchanged': cv2.IMREAD_UNCHANGED,
        }

        assert mode in cv_color, (
            Strings.invalid_parameter_value % (mode, 'mode')
        )

        img = cv2.imread(img_path, cv_color[mode])

        if (mode == 'color'):
            img = ImageSample.cvt_color(img)

        return img

    @staticmethod
    def write(img_src:np.array, out_path:str, file_name:str='out.png'):
        assert os.path.exists(out_path), Strings.no_path

        return cv2.imwrite(os.path.join(out_path, file_name) , img_src)

    @staticmethod
    def cvt_color(
            img_src:np.array,
            from_cs:str='bgr',
            to_cs:str='rgb',
            cv_cvt_code=None
        ):
        color_cvt_codes = {
            'bgr': {'rgb': cv2.COLOR_BGR2RGB, 'hsv': cv2.COLOR_BGR2HSV},
            'rgb': {'bgr': cv2.COLOR_RGB2BGR, 'hsv': cv2.COLOR_RGB2HSV},
            'hsv': {'rgb': cv2.COLOR_HSV2RGB, 'bgr': cv2.COLOR_HSV2BGR},
        }

        if (cv_cvt_code):
            code = cv_cvt_code
        else:
            assert from_cs in color_cvt_codes
            assert to_cs in color_cvt_codes[from_cs]

            code = color_cvt_codes[from_cs][to_cs]

        return cv2.cvtColor(img_src, code)

    @staticmethod
    def to_grayscale(
            img_src:np.array,
            src_color_sch:str='rgb'
        ):

        assert src_color_sch in {'rgb', 'bgr', 'hsv'}

        if src_color_sch == 'rgb':
            img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
        elif src_color_sch == 'bgr':
            img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        elif src_color_sch == 'hsv':
            img_src = img_src[:, :, 2]

        return img_src

    @staticmethod
    def resize(img_scr: np.array, target_shape: tuple):
        return image.resize(img_scr, target_shape)