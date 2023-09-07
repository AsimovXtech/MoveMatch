import cv2
import time
import numpy as np
from utils import utilFunctions
import argparse
import os
from mediapipe.python.solutions import pose as mp_pose
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn import preprocessing

def process_video_and_image(video_path, ref_image_path, result_folder):
    # Initialize pose estimation model
    pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.1, min_tracking_confidence=0.2, static_image_mode=True)

    # Extract the coordinates and confidence scores for reference image
    ref_img, ref_coords, ref_norm_coords, ref_cf = utilFunctions.extract_coordinates(pose, ref_image_path)

    # Extract video coordinates (please implement vid_coords extraction from video in utilFunctions)
    vid_frames, vid_coords, vid_cf = utilFunctions.process_video(video_path)
    print('\nVideo processed')
    # Apply smoothing
    smooth_coords = savgol_filter(np.array(vid_coords), 10, 1, axis=0)
    print('Coordinates Smoothed')

    # Compute scores
    scores = []
    for coords in smooth_coords:
        coords = utilFunctions.get_new_coords(coords.copy())
        coords_std = preprocessing.normalize(coords, norm='l2')
        score = utilFunctions.weighted_distance_matching(ref_norm_coords.flatten(), ref_cf, coords_std.flatten())
        scores.append(score)

    match_idx = np.argmax(scores)

    # Visualize results
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle('MoveMatch Results', fontsize=14)
    axs[0].set_title('Reference Image')
    axs[0].imshow(utilFunctions.draw_pose(ref_img, ref_coords, 4))
    axs[1].set_title('Matched Frame')
    axs[1].imshow(utilFunctions.draw_pose(vid_frames[match_idx], vid_coords[match_idx], 4))
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    result_path = os.path.join(result_folder, 'result.png')
    plt.savefig(result_path)
    plt.show()

    return match_idx, result_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoveMatch CLI for Pose Analysis and Comparison")
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--reference", required=True, help="Path to the reference image")
    parser.add_argument("--output", required=True, help="Path to the result folder")

    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    match_idx, result_path = process_video_and_image(args.video, args.reference, args.output)
    
    print(f"Matched frame index: {match_idx}")
    print(f"Result saved at: {result_path}")

