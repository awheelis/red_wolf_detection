"""
Created by George Wheelis
Updated 11/14/22
"""

## these are commented out so you can test the application script
## in order to pass, the app has to have these packages already downloaded
import numpy as np
from scipy.fft import fft
import cv2
import sys
sys.path.append("/Users/alex_wheelis/Documents/programming/red_wolf_detection/backend/algo_dev")
from misc import *

def dummy():
    """
    dummy func to test front end and back end communication:
     - checks to see if numpy, cv2, scipy are downloaded (should be done on server...)
     - writes a mp4 file "test_vid.mp4"
     - writes a txt file "test_txt.txt"
    """
    try: 
        # numpy download test
        frames = 100
        w, h = 64, 64
        test_vid = np.zeros((frames,w,h))

        # scipy download test
        test_img = np.zeros((30,30))
        Z =  fft(test_img)

        # opencv download test
        rot_img = cv2.rotate(test_img, cv2.ROTATE_90_CLOCKWISE)

        # video writing test
        save_video(test_vid, "test_vid.mp4", fps = 3)

        # text writing test 
        with open("test_txt.txt", "w") as file: 
            test_text = """
            This text is a test...
            """
            file.write(test_text)
        file.close()
        

    except Exception as e: 
        print("TEST FAILED")
        print(e)


if __name__ == "__main__":
    dummy()