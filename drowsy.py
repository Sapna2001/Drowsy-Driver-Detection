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

# To create web app
from Tkinter import *
import tkMessageBox

# To play alert sound
from pygame import mixer

# Creating UI
root = Tk()
root.attributes('-fullscreen', True)
frame = Frame(root, relief = RIDGE, borderwidth = 2)
frame.pack(fill = BOTH, expand = 1)
root.title('Drowsy Driver Detection')
frame.config(background = 'light blue')
label = Label(frame, text="Drowsy Driver Detection",
              bg = 'light blue', font = ('Times 35 bold'))
label.pack(side = TOP)
filename = PhotoImage(file = "demo.png")
background_label = Label(frame, image = filename)
background_label.pack(side = TOP)

# Open the open cv help box
def help_box():
    help(cv2)

# open the about section
def about_box():
    tkMessageBox.showinfo(
        "About", 'Driver Cam version v1.0\n Made Using\n-OpenCV\n-Numpy\n-Tkinter\n In Python 3')

# Menu options
menu = Menu(root)
root.config(menu = menu)

sub_menu1 = Menu(menu)
menu.add_cascade(label = "Tools", menu = sub_menu1)
sub_menu1.add_command(label = "Open CV Docs", command = help_box)

sub_menu2 = Menu(menu)
menu.add_cascade(label = "About", menu = sub_menu2)
sub_menu2.add_command(label = "About the software", command = about_box)

# Close the app
def close_window():
    quit()


def web_detect():
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
    # status = ""

    yawn_threshold = 22

    # Eye closing detection
    def calculate_EAR(a, b, c, d, e, f):
        up = dist.euclidean(b, d) + dist.euclidean(c, e)
        down = dist.euclidean(a, f)

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

        top_mean = np.mean(top_lip, axis=0)
        bottom_mean = np.mean(bottom_lip, axis=0)

        distance = dist.euclidean(top_mean, bottom_mean)
        return distance

    # Capture image till keyboard interrupt received
    while True:
        success, frame = capture.read()

        # If video capture not successful
        if not success:
            break

        # Change to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect landmarks
        faces = detector(gray)

        # Detected face in faces array
        for face in faces:
            # Find landmarks on the face
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            # Calculate EAR
            left_blink = calculate_EAR(landmarks[36], landmarks[37],
                                       landmarks[38], landmarks[41], landmarks[40], landmarks[39])

            right_blink = calculate_EAR(landmarks[42], landmarks[43],
                                        landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            # Calculating the lip distance
            lip_dist = calculate_yawning(landmarks)

            # Mixer settings
            mixer.init()

            # Conditions
            if(left_blink == 0 or right_blink == 0):
                sleep += 1
                active = 0
                drowsy = 0
                if(sleep > 6):
                    # status = "SLEEPING !!!"
                    mixer.music.load("audio/sleeping.wav")
                    mixer.music.set_volume(0.8)
                    mixer.music.play()

            elif(lip_dist > yawn_threshold):
                drowsy += 1
                sleep = 0
                active = 0
                if(drowsy > 3):
                    # status = "Drowsy !"
                    mixer.music.load("audio/drowsy.WAV")
                    mixer.music.set_volume(0.8)
                    mixer.music.play()

            elif(left_blink == 1 or right_blink == 1):
                drowsy += 1
                sleep = 0
                active = 0
                if(drowsy > 6):
                    # status = "Drowsy !"
                    mixer.music.load("audio/drowsy.WAV")
                    mixer.music.set_volume(0.8)
                    mixer.music.play()

            else:
                sleep = 0
                drowsy = 0
                active += 1
                if(active > 6):
                    # status = "Active :)"
                    mixer.music.stop()

            cv2.imshow("Frame", frame)

        # Wait for any keyboard event to happen
        if cv2.waitKey(1) & 0xFF == ord('q'):
            mixer.music.stop()
            cv2.destroyAllWindows()
            capture.release()
            break

# Buttons
button_open = Button(frame, padx = 5, pady = 5, width = 39, bg = 'white', fg = 'black',
                     relief = GROOVE, command = web_detect, text = 'Open Camera & Detect', font = ('helvetica 15 bold'))
button_open.place(relx = 0.5, rely = 0.3, anchor = CENTER)

button_exit = Button(frame, padx = 5, pady = 5, width = 5, bg='white', fg='black',
                     relief = GROOVE, text = 'EXIT', command = close_window, font = ('helvetica 15 bold'))
button_exit.place(relx = 0.5, rely = 0.7, anchor = CENTER)

# Run app
root.mainloop()
