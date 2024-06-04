from PIL import Image
from PIL.ExifTags import TAGS

import cv2

def get_exif_data(pimage):
    image = Image.open(pimage)
    exif_data = {}
    try:
        info = image._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                exif_data[decoded] = value
    except AttributeError:
        print("No EXIF data found")
    return exif_data