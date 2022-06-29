# Importing OpenCV Library for basic image processing functions
import cv2

# Numpy for array related functions
import numpy as np

# Dlib for deep learning based Modules and face landmark detection
import dlib

# face_utils for basic operationsns of conversion
from imutils import face_utils

# Calcuate euclidean distance 
from scipy.spatial import distance as dist

# Initializing the camera and taking the instance
capture = cv2.VideoCapture(0)

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()

# Detect 68 landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0,0,0)

yawn_threshold = 22

# Eye closing detection
def calculate_EAR(a,b,c,d,e,f):
	up = dist.euclidean(b,d) + dist.euclidean(c,e)
	down = dist.euclidean(a,f)

    # Eye Aspect Ratio
	EAR = up / (2.0 * down)

	# Checking if it is calculate_EAR
    # Active
	if(EAR > 0.25):
		return 2
    # Drowsy
	elif(EAR > 0.21 and EAR <= 0.25):
		return 1
    # Sleeping
	else:
		return 0

# Yawning detection
def calculate_yawning(shape): 
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
  
    bottom_lip = shape[56:59]
    bottom_lip = np.concatenate((bottom_lip, shape[65:68]))
  
    top_mean = np.mean(top_lip, axis = 0)
    bottom_mean = np.mean(bottom_lip, axis = 0)
  
    distance = dist.euclidean(top_mean,bottom_mean)
    return distance

# Capture image till keyboard interrupt received
while True:
    success,frame = capture.read()
  
    # If video capture not successful
    if not success : 
        break

    # Change to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect landmarks
    faces = detector(gray)

    # Detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Find landmarks on the face
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # # Calculate EAR
        left_blink = calculate_EAR(landmarks[36],landmarks[37], 
        	landmarks[38], landmarks[41], landmarks[40], landmarks[39])

        right_blink = calculate_EAR(landmarks[42],landmarks[43], 
        	landmarks[44], landmarks[47], landmarks[46], landmarks[45])
  
        # Calculating the lip distance
        lip = landmarks[48:60]
        lip_dist = calculate_yawning(landmarks)
        
        # Conditions 
        if(left_blink == 0 or right_blink == 0):
            sleep += 1
            drowsy = 0
            active = 0
            if(sleep > 6):
                status="SLEEPING !!!"
                color = (255,0,0)

        elif(lip_dist > yawn_threshold):
                sleep += 1
                drowsy = 0
                active = 0
                if(sleep > 3):
                    status="Drowsy !"
                    color = (0,0,255)


        elif(left_blink == 1 or right_blink == 1):
            sleep += 1
            drowsy = 0
            active = 0
            if(sleep > 6):
                status="Drowsy !"
                color = (0,0,255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if(active > 6):
                status = "Active :)"
                color = (0,255,0)
        	
        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

        for n in range(0, 68):
            (x,y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        cv2.imshow("Frame", frame)
        cv2.imshow("Result of detector", face_frame)

    # Wait for any keyboard event to happen  
    key = cv2.waitKey(1)
    if key == 27:
      	break