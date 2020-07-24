import cv2
import numpy as np

from prz.resources.data.io import DataIO
from prz.definitions.strings import Strings


class ImageSample:
    @staticmethod
    def read(img_path, mode='COLOR'):
        cv_color = {
            'COLOR': cv2.IMREAD_COLOR,
            'GS': cv2.IMREAD_GRAYSCALE,
            'UNC': cv2.IMREAD_UNCHANGED,
        }

        assert mode in cv_color.keys(), Strings.invalidParameterValue % (mode, 'mode')

        img = cv2.imread(img_path, cv_color[mode])

        if (mode == 'COLOR'):
            img = ImageSample.cvt_color(img)

        return img

    @staticmethod
    def write(img=np.array([]), out_path=''):
        assert DataIO.pathExists(out_path), Strings.noPath

        return cv2.imwrite(out_path, img)

    @staticmethod
    def cvt_color(img_src=np.array([]),
                  config={'from': 'BGR', 'to': 'RGB'},
                  cv_cvt_code=None):
        color_cvt_codes = {
            'BGR': {'RGB': cv2.COLOR_BGR2RGB, 'HSV': cv2.COLOR_BGR2HSV},
            'RGB': {'BGR': cv2.COLOR_RGB2BGR, 'HSV': cv2.COLOR_RGB2HSV},
            'HSV': {'RGB': cv2.COLOR_HSV2RGB, 'BGR': cv2.COLOR_HSV2BGR},
        }

        if (cv_cvt_code):
            code = cv_cvt_code
        else:
            assert {'from', 'to'}.issubset(set(config.keys()))
            assert config['from'] in color_cvt_codes.keys()
            assert config['to'] in color_cvt_codes[config['from']].keys()

            code = color_cvt_codes[config['from']][config['to']]

        return cv2.cvtColor(img_src, code)

    @staticmethod
    def to_grayscale(img_src=np.array([]),
                     src_color_sch='RGB'):

        assert src_color_sch in ['RGB', 'BGR', 'HSV']

        if src_color_sch == 'RGB':
            img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
        elif src_color_sch == 'BGR':
            img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        elif src_color_sch == 'HSV':
            img_src = img_src[:, :, 2]

        return img_src
