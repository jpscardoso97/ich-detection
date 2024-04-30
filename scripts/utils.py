import cv2
import numpy as np
import pydicom

RESIZE_DIM = 224

def read_dcm(img_path):
    img_dicom = pydicom.dcmread(img_path)
    # convert to numpy array
    # Check the PhotometricInterpretation metadata
    if img_dicom.PhotometricInterpretation == 'MONOCHROME1':
        image = np.invert(img_dicom.pixel_array)
    else:
        image = img_dicom.pixel_array

    # Rescale the pixel values to [0, 255]
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Resize the image to desired size
    return cv2.resize(image, (RESIZE_DIM, RESIZE_DIM))