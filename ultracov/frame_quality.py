# -*- coding: utf-8 -*-
"""
Functions for frame quality assessment
"""

import numpy as np
from skimage import measure


def analyze_frame(frame, mask):
    """Get image quality parameters of input frame and mask

    Parameters
    ----------
    frame
    mask

    Returns
    -------

    """

    mask = np.int32(np.round(np.squeeze(mask)))
    mask_labels = measure.label(mask)
    mask_regions = measure.regionprops(mask_labels)
    mask_regions.sort(key=lambda x: x.area, reverse=True)

    return mask_labels, mask_regions


def is_valid(mask_labels, mask_regions):
    """Defines image quality criteria. Returns True/False

    Parameters
    ----------
    mask_labels
    mask_regions

    Returns
    -------

    """

    valid = True
    # At least one region is detected
    if len(mask_regions) == 0:
        valid = False

    # Biggest region minimum area
    elif mask_regions[0].area < 5:
        valid = False

    return valid

# if __name__ == "__main__":
#     " Example: get quality parameters of all frames in a video. Display video "

#     video_path = r"C:\Principal\ultracov\videos\10_L4_0.BIN"
#     bfile = BinFile(video_path)
#     dset = Dataset(bfile)
#     dset.ScanConvert()

#     model = load_model()

#     dset_frame = dset.frames[:,:,0]
#     input_img = load_img(dset_frame)
#     mask = predict(input_img, model)

#     video_frame_histogram = []
#     video_mask_labels = []
#     video_mask_regions = []

#     for i in range(dset.nimg):
#         dset_frame = dset.frames[:,:,i]
#         input_img = load_img(dset_frame)
#         mask = predict(input_img, model)

#         frame_histogram, mask_labels, mask_regions = analyze_frame(dset_frame, mask)

#         video_frame_histogram.append(frame_histogram)
#         video_mask_labels.append(mask_labels)
#         video_mask_regions.append(mask_regions)

#     video_histogram = np.array(video_frame_histogram)
#     video_labels = np.array(video_mask_labels)
