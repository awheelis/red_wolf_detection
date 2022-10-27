"""

this file will contain useful functions for video processing

1) reading a video into an array 
2) reducing resolution of a video
3) saving a video from an array 


"""

from re import L
import cv2
import numpy as np
from PIL import Image
from glob import glob
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import bottleneck as bn

import logging

import pdb

K = 2 # background suppression threshold
H_PERCENT = 99 # upper clipping thresh
L_PERCENT = 1 # lower clipping thresh

def load_video(file_name, frames=100):
    vidcap = cv2.VideoCapture(file_name)
    success, image = vidcap.read()
    img_array = []
    count = 0

    while success:
        if count >= frames:
            break
        success, image = vidcap.read()
        if success:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_array.append(img_gray)
            count += 1
    return np.array(img_array)

def decrease_res_by_x(video, width = 1920, height = 1080):
    """
    input: 
        height, width (int)
        - this is how much you want to decrease the resolution by
        - eg. resolution = 2 --> 1080x1920 cvt 540x960
    output: 
        none
    """
    decreased_res = np.zeros((video.shape[0], height, width), dtype = np.uint8)
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

def save_video_w_mvmt_annotations(video, annotations_arr, file_name, fps = 30):

    video = video.astype(np.uint8)
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    frame_width, frame_height = video.shape[2], video.shape[1]
    vid_path = "./videos/"
    if not os.path.exists(vid_path):
        os.mkdir("./videos/")
    out = cv2.VideoWriter(f'./videos/{file_name}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width,frame_height), 0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i, frame in enumerate(video):
        cv2.putText(frame, f"P(Wolf Present) = {annotations_arr[i]:.2f}", (10, 500), font, 4,(255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
    out.release()
    print(os.getcwd(), f"/video/{file_name}.mp4",sep='')

def clip_img(arr, h, l): 
    """
    takes in an array and clips by the h percentile and l percentile
    arr: array
        image array
    h: float
        higher thresh
    l: float
        lower thresh
    """
    clip_max = np.percentile(arr, h)
    clip_min = np.percentile(arr, l)

    arr = np.clip(arr, clip_min, clip_max)
    return arr

def linear_normalization(arr):
    """
    scales from 0-1 with x' = (x - xmin)/(xmax - xmin)
    arr: array
        image/video array
    """
    arr = (arr - np.min(arr))/(np.max(arr) - np.min(arr))
    return arr

def bs(video_array):
    "Background Suppress"
    # use bottleneck (bn) for cal   culations
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
    
    logging.info("converting list to array")
    video_array = np.array(video_array, dtype = np.uint8)

    logging.info("clipping video based on thresh")
    video_array = clip_img(video_array, H_PERCENT, L_PERCENT)

    logging.info("Getting MEDIAN frame")
    mean_vid = bn.median(video_array, axis = 0)
    
    logging.info("Getting STD frame")
    std_vid = bn.nanstd(video_array, axis = 0) + 1
    
    logging.info("subtracting mean and dividing std") 
    video_array -= mean_vid
    video_array /= std_vid

    logging.info(f"creating mask based off of K = {K}")
    video_array = (video_array > K).astype(np.uint8)
    
    video_array *= 255

    video_array = video_array.astype(np.uint8)

    # logging.info("blurring")
    logging.info("******** DONE! ********")
    return video_array


if __name__ == "__main__":
    wolf_dir = "/Users/alex_wheelis/Documents/Fall2022/ECE 484/DCIM/100_BTCF"
    wolf_vid_fs = glob(wolf_dir + "/*")
    test_vid_f = wolf_vid_fs[0]
    wolf_vid = load_video(test_vid_f)
    # res = decrease_res_by_x(wolf_vid, 500, 500)
    save_video(bs(wolf_vid), 'bs')
    
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