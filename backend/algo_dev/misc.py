"""

this file will contain useful functions for video processing

1) reading a video into an array 
2) reducing resolution of a video


"""

import cv2
import numpy as np
from PIL import Image
from glob import glob
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

import pdb

def load_video(file_name):
    vidcap = cv2.VideoCapture(file_name)
    success, image = vidcap.read()
    img_array = []
    count = 0

    while success and count < 220:
        success, image = vidcap.read()
        if success:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_array.append(img_gray)
            count += 1
    return np.array(img_array)

def decrease_res_by_x(video, width, height):
    """
    input: 
        height, width (int)
        - this is how much you want to decrease the resolution by
        - eg. resolution = 2 --> 1080x1920 cvt 540x960
    output: 
        none
    """
    decreased_res = np.zeros((video.shape[0], width, height), dtype = np.uint8)
    for i, frame in enumerate(video):
        decreased_res[i,...] = np.array(Image.fromarray(frame).resize((width, height)))
    return decreased_res

def save_video(video, file_name, fps = 30):
    video = video.astype(np.uint8)
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    frame_width, frame_height = video.shape[2], video.shape[1]
    vid_path = "./videos/"
    if not os.path.exists(vid_path):
        os.mkdir("./videos/")
    out = cv2.VideoWriter(f'./videos/{file_name}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width,frame_height), 0)
    for i, frame in enumerate(video):
        out.write(frame)
    out.release()
    print(os.getcwd(), f"/video/{file_name}.mp4",sep='')

if __name__ == "__main__":
    wolf_dir = "/Users/alex_wheelis/Documents/Fall2022/ECE 484/DCIM/100_BTCF"
    wolf_vid_fs = glob(wolf_dir + "/*")
    test_vid_f = wolf_vid_fs[0]
    wolf_vid = load_video(test_vid_f)
    pdb.set_trace()



    # what do I want to do?
    """
    read in video 
    manipulate video
    check what I my manipulations

    in order to do this I think I'll need this process...
    read in video
    manipulate video
    save video
    display video w/ file name
    
    
    
    """