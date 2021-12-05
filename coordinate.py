# python 3.7
"""Utility functions to help define the region in an image."""

import numpy as np


def get_mask_by_coordinates(image_size, coordinate):
    """Gets a masked region from coordinates."""
    mask = np.zeros([image_size, image_size], dtype=np.float32)
    center_x, center_y = coordinate[0], coordinate[1]
    crop_x, crop_y = coordinate[2], coordinate[3]
    xx = center_x - crop_x // 2
    yy = center_y - crop_y // 2
    mask[xx:xx + crop_x, yy:yy + crop_y] = 1.
    return mask


def get_mask_by_segmentation(seg_mask, label):
    """Gets a masked region from a segmentaton map."""
    zeros = np.zeros_like(seg_mask)
    ones = np.ones_like(seg_mask)
    mask = np.where(seg_mask == label, ones, zeros)
    return mask


def get_mask(image_size, coordinate=None, seg_mask=None, labels='1'):
    """Gets a masked region from either coordinates or segmentation map."""
    if coordinate is not None:
        print(f'Using coordinate to get mask!')
        mask = get_mask_by_coordinates(image_size, coordinate)
    else:
        print('Using segmentation to get the mask!')
        print(f'Using label {labels}')
        mask = np.zeros_like(seg_mask)
        for label_ in labels:
            mask += get_mask_by_segmentation(seg_mask, int(label_))
        mask = np.clip(mask, a_min=0, a_max=1)

    return mask


# For FFHQ [center_x, center_y, height, width]
# Those coordinates suitable for both ffhq and metface.
COORDINATE_face = {'left_eye': [120, 95, 20, 38],
                   'right_eye': [120, 159, 20, 38],
                   'eyes': [120, 128, 20, 115],
                   'nose': [142, 131, 40, 46],
                   'mouth': [184, 127, 30, 70],
                   'chin': [217, 130, 42, 110],
                   'left_region': [128, 74, 128, 64],
                   'point_eye': [120, 95, 2, 2],
                   'point_mouth': [184, 127, 2, 2],
                   }


seg_mapping = {
    'ffhq': {
        0: 'background',
        1: 'skin',
        2: 'nose',
        3: 'eye_g',
        4: 'l_eye',
        5: 'r_eye',
        6: 'l_brow',
        7: 'r_brow',
        8: 'l_ear',
        9: 'r_ear',
        10: 'mouth',
        11: 'u_lip',
        12: 'l_lip',
        13: 'hair',
        14: 'hat',
        15: 'ear_r',
        16: 'neck_l',
        17: 'neck',
        18: 'cloth'
    },
}
