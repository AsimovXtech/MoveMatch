# Import required modules
import cv2
import time
import numpy as np

import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.signal import savgol_filter
from mediapipe.python.solutions import pose as mp_pose

class utilFunctions:

    #Function to get the bounding box from the coordinate list
    def bounding_box(coords):
        min_x = 100000 
        min_y = 100000
        max_x = -100000 
        max_y = -100000

        for item in coords:
            if item[0] < min_x:
                min_x = item[0]

            if item[0] > max_x:
                max_x = item[0]

            if item[1] < min_y:
                min_y = item[1]

            if item[1] > max_y:
                max_y = item[1]
        return [(int(min_x),int(min_y)),(int(max_x),int(min_y)),(int(max_x),int(max_y)),(int(min_x),int(max_y))]

    #Function to standardise coordinates
    def get_new_coords(coords):
        bbox = utilFunctions.bounding_box(coords)
        coords[:,:1] = coords[:,:1] - bbox[0][0]
        coords[:,1:2] = coords[:,1:2] - bbox[0][1]
        return coords

    #Function to extract the coordinates by using MediaPipe Landmarks
    def get_pose_coords(pose, image):
        height, width, _ = image.shape
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            get_pose = results.pose_landmarks.landmark
            lm = mp_pose.PoseLandmark

            left_wrist_x = get_pose[lm.LEFT_WRIST].x * width
            left_wrist_y = get_pose[lm.LEFT_WRIST].y * height
            left_wrist_v = get_pose[lm.LEFT_WRIST].visibility
            left_elbow_x = get_pose[lm.LEFT_ELBOW].x * width
            left_elbow_y = get_pose[lm.LEFT_ELBOW].y * height
            left_elbow_v = get_pose[lm.LEFT_ELBOW].visibility
            left_shoulder_x = get_pose[lm.LEFT_SHOULDER].x * width
            left_shoulder_y = get_pose[lm.LEFT_SHOULDER].y * height
            left_shoulder_v = get_pose[lm.LEFT_SHOULDER].visibility
            left_hip_x = get_pose[lm.LEFT_HIP].x * width
            left_hip_y = get_pose[lm.LEFT_HIP].y * height
            left_hip_v = get_pose[lm.LEFT_HIP].visibility
            left_knee_x = get_pose[lm.LEFT_KNEE].x * width
            left_knee_y = get_pose[lm.LEFT_KNEE].y * height
            left_knee_v = get_pose[lm.LEFT_KNEE].visibility
            left_ankle_x = get_pose[lm.LEFT_ANKLE].x * width
            left_ankle_y = get_pose[lm.LEFT_ANKLE].y * height
            left_ankle_v = get_pose[lm.LEFT_ANKLE].visibility
            left_foot_x = get_pose[lm.LEFT_FOOT_INDEX].x * width
            left_foot_y = get_pose[lm.LEFT_FOOT_INDEX].y * height
            left_foot_v = get_pose[lm.LEFT_FOOT_INDEX].visibility

            right_wrist_x = get_pose[lm.RIGHT_WRIST].x * width
            right_wrist_y = get_pose[lm.RIGHT_WRIST].y * height
            right_wrist_v = get_pose[lm.RIGHT_WRIST].visibility
            right_elbow_x = get_pose[lm.RIGHT_ELBOW].x * width
            right_elbow_y = get_pose[lm.RIGHT_ELBOW].y * height
            right_elbow_v = get_pose[lm.RIGHT_ELBOW].visibility
            right_shoulder_x = get_pose[lm.RIGHT_SHOULDER].x * width
            right_shoulder_y = get_pose[lm.RIGHT_SHOULDER].y * height
            right_shoulder_v = get_pose[lm.RIGHT_SHOULDER].visibility
            right_hip_x = get_pose[lm.RIGHT_HIP].x * width
            right_hip_y = get_pose[lm.RIGHT_HIP].y * height
            right_hip_v = get_pose[lm.RIGHT_HIP].visibility
            right_knee_x = get_pose[lm.RIGHT_KNEE].x * width
            right_knee_y = get_pose[lm.RIGHT_KNEE].y * height
            right_knee_v = get_pose[lm.RIGHT_KNEE].visibility
            right_ankle_x = get_pose[lm.RIGHT_ANKLE].x * width
            right_ankle_y = get_pose[lm.RIGHT_ANKLE].y * height
            right_ankle_v = get_pose[lm.RIGHT_ANKLE].visibility
            right_foot_x = get_pose[lm.RIGHT_FOOT_INDEX].x * width
            right_foot_y = get_pose[lm.RIGHT_FOOT_INDEX].y * height
            right_foot_v = get_pose[lm.RIGHT_FOOT_INDEX].visibility

            nose_x = get_pose[lm.NOSE].x * width
            nose_y = get_pose[lm.NOSE].y * height
            nose_v = get_pose[lm.NOSE].visibility
            
            coords = np.array([(left_wrist_x, left_wrist_y), (left_elbow_x, left_elbow_y), (left_shoulder_x, left_shoulder_y), (left_hip_x, left_hip_y), (left_knee_x, left_knee_y), (left_ankle_x, left_ankle_y), (left_foot_x, left_foot_y),
                    (right_wrist_x, right_wrist_y), (right_elbow_x, right_elbow_y), (right_shoulder_x, right_shoulder_y), (right_hip_x, right_hip_y), (right_knee_x, right_knee_y), (right_ankle_x, right_ankle_y), (right_foot_x, right_foot_y),
                    (nose_x,nose_y)])
            v_scores = np.array([left_wrist_v, left_elbow_v, left_shoulder_v, left_hip_v, left_knee_v, left_ankle_v, left_foot_v,
                    right_wrist_v, right_elbow_v, right_shoulder_v, right_hip_v, right_knee_v, right_ankle_v, right_foot_v, nose_v])
            
            return coords, v_scores
        else:
            return results

    #Scoring function to compare the poses    
    def weighted_distance_matching(ref_coords, ref_v, frame_coords):
        # Initialize reference and frame coordinates
        vector_1_pose_xy = ref_coords
        vector_1_confidences = ref_v
        vector_1_confidence_sum = ref_v.sum()
        vector_2_pose_xy = frame_coords

        # Compute weighted Euclidean distance
        summation_1 = 1 / vector_1_confidence_sum
        summation_2 = 0
        for i in range(len(vector_1_pose_xy)):
            temp_conf = i // 2
            temp_sum = vector_1_confidences[temp_conf] * abs(vector_1_pose_xy[i] - vector_2_pose_xy[i])
            summation_2 += temp_sum

        return 1-(summation_1 * summation_2)

    def extract_coordinates(poseModel, image_input):
        # Check if the input is a string (indicating a path) or a numpy array (indicating an image)
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise ValueError("Input should be either an image path or a numpy array representing an image")

        # Convert image color from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get pose coordinates from the reference image
        coords, cf = utilFunctions.get_pose_coords(poseModel, image)
        coords_std = utilFunctions.get_new_coords(coords.copy())

        # Normalize pose coordinates
        coords_norm = preprocessing.normalize(coords_std, norm='l2')
        return image, coords, coords_norm, cf

            
    def extract_video_coordinates(pose, video_path):
        vid_coords = []
        vid_frames = []
        
        # Read the video file
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            
            if ret:
                # Extract coordinates for the current frame
                coords, _, _, _ = utilFunctions.extract_coordinates(pose, frame) # Assuming you have a similar function for a single frame
                vid_coords.append(coords)
                vid_frames.append(frame)
            else:
                cap.release()
                break
        return vid_coords, vid_frames

    #Function to process and extract coordinates from a video file
    def process_video(video_path):
        # Initialize arrays to store output frames and pose coordinates
        output_frames = []
        vid_coords = []
        vid_cf = []

        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize pose estimation model
        pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.1, min_tracking_confidence=0.2)
        
        vid_start = time.time()
        if cap.isOpened(): 
            while(cap.isOpened()):
                # Start timer
                start_time = time.time()

                # Read a frame from the video
                ret, frame = cap.read()

                if ret:
                    # Convert frame color from BGR to RGB
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Get pose coordinates from the frame
                    coords, v_scores = utilFunctions.get_pose_coords(pose, image)

                    # Store pose coordinates and output frame
                    vid_coords.append(coords)
                    vid_cf.append(v_scores)
                    output_frames.append(image)

                    # End timer and print processing time
                    end_time = time.time()
                    print(f'Processing Video: {video_path.split("/")[-1]} | FPS: {1/(end_time-start_time) :.2f}', end='\r')
                else:
                    vid_end = time.time()
                    break
        print(f'Successfully processed the video! | Frames: {n_frames} | FPS: {1/(vid_end-vid_start) :.2f}')
        return output_frames, vid_coords, vid_cf

    def draw_pose(image, coords, thicc=2):
    #     width, height = image.shape[1], image.shape[0]
        for i in coords:
            cv2.circle(image, (int(i[0]),int(i[1])), 5, (255,255,255), thicc)
        cv2.line(image, (int(coords[0][0]),int(coords[0][1])), (int(coords[1][0]),int(coords[1][1])), (0,255,145), thicc)
        cv2.line(image, (int(coords[1][0]),int(coords[1][1])), (int(coords[2][0]),int(coords[2][1])), (0,255,145), thicc)
        cv2.line(image, (int(coords[2][0]),int(coords[2][1])), (int(coords[3][0]),int(coords[3][1])), (0,255,145), thicc)
        cv2.line(image, (int(coords[3][0]),int(coords[3][1])), (int(coords[4][0]),int(coords[4][1])), (0,255,145), thicc)
        cv2.line(image, (int(coords[4][0]),int(coords[4][1])), (int(coords[5][0]),int(coords[5][1])), (0,255,145), thicc)
        cv2.line(image, (int(coords[5][0]),int(coords[5][1])), (int(coords[6][0]),int(coords[6][1])), (0,255,145), thicc)

        cv2.line(image, (int(coords[7][0]),int(coords[7][1])), (int(coords[8][0]),int(coords[8][1])), (0,255,145), thicc)
        cv2.line(image, (int(coords[8][0]),int(coords[8][1])), (int(coords[9][0]),int(coords[9][1])), (0,255,145), thicc)
        cv2.line(image, (int(coords[9][0]),int(coords[9][1])), (int(coords[10][0]),int(coords[10][1])), (0,255,145), thicc)
        cv2.line(image, (int(coords[10][0]),int(coords[10][1])), (int(coords[11][0]),int(coords[11][1])), (0,255,145), thicc)
        cv2.line(image, (int(coords[11][0]),int(coords[11][1])), (int(coords[12][0]),int(coords[12][1])), (0,255,145), thicc)
        cv2.line(image, (int(coords[12][0]),int(coords[12][1])), (int(coords[13][0]),int(coords[13][1])), (0,255,145), thicc)

        cv2.line(image, (int(coords[2][0]),int(coords[2][1])), (int(coords[9][0]),int(coords[9][1])), (255,255,255), thicc)
        cv2.line(image, (int(coords[3][0]),int(coords[3][1])), (int(coords[10][0]),int(coords[10][1])), (255,255,255), thicc)

        return image 
