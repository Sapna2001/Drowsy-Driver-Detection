# Importing OpenCV Library for basic image processing functions
import re
import cv2

# Dlib for deep learning based Modules and face landmark detection
import dlib

# face_utils for basic operations of conversion
from imutils import face_utils

# Calcuate euclidean distance
from scipy.spatial import distance as dist

# To play alert sound
from pygame import mixer

# To create web app
from Tkinter import *
import tkMessageBox
from PIL import ImageTk, Image

# To send message alert
import requests

# Hide API Key
import os
from dotenv import load_dotenv
load_dotenv()

# Regex for phone number validation

# Creating UI
root = Tk()
root.title("Drowsy Driver Detection")
# root.iconbitmap('Images/Drowsy _icon.ico')
root.configure(background="#FFFDD0")
root.attributes('-fullscreen', True)

# Login page
drowsy_img = ImageTk.PhotoImage(Image.open("Images/icon.png"))
img_label = Label(image=drowsy_img)
img_label.grid(row=5, column=0, padx=200, pady=200)

label_signup = Label(root, text="Enter Your Details", width=20, font=(
    "helvetica 30 bold"), bg="#FFFDD0", fg="black")
label_signup.place(relx=0.57, rely=0.35, anchor=W)

name = Label(root, text="Name", width=15, font=(
    "bold", 20), bg="#FFFDD0", fg="black")
name.place(relx=0.5, rely=0.45)
str_name = StringVar(root)
entry_name = Entry(root, textvariable=str_name, width=20, font=("bold", 20))
entry_name.place(relx=0.7, rely=0.45)

emergency = Label(root, text="Emergency Contact", width=18,
                  font=("bold", 20), bg="#FFFDD0", fg="black")
emergency.place(relx=0.5, rely=0.55)
str_emergency = StringVar(root)
entry_emergency = Entry(root, textvariable=str_emergency,
                        width=20, font=("bold", 20))
entry_emergency.place(relx=0.7, rely=0.55)


def close_window():
    quit()


def isValid(s):
    # Check if no begins with 0 or 91, then contains 7 or 8 or 9, then contains 9 digits
    Pattern = re.compile("(0|91)?[7-9][0-9]{9}")
    return Pattern.match(s)


def detecting():
    if entry_name.get() == "" and entry_emergency.get() == "":
        tkMessageBox.showinfo(
            'Message', 'Fill out the details'
        )

    else:
        str_name = entry_name.get()
        str_emergency = entry_emergency.get()
        if not isValid(str_emergency):
            tkMessageBox.showinfo(
                'Message', 'Invalid Phone Number'
            )
        else:
            cam_detect = Toplevel(root)
            root.destroy()
            cam_detect = Tk()
            cam_detect.title('Drowsy Driver Detection')
            # cam_detect.iconbitmap("Drowsy _icon.ico")
            cam_detect.attributes('-fullscreen', True)
            frame = Frame(cam_detect, relief=RIDGE, borderwidth=2)
            frame.pack(fill=BOTH, expand=1)
            frame.config(background='#F5F5DC')
            label = Label(frame, text="Drowsy Driver Detection",
                          bg='#F5F5DC', font=('Times 35 bold'))
            label.pack(side=TOP)

            bg_image = ImageTk.PhotoImage(Image.open("Images/background.png"))
            bg_label = Label(cam_detect, image=bg_image)
            bg_label.pack()

            # Open the open cv help box
            def help_box():
                tkMessageBox.showinfo(
                    "Help", 'Step 1: Enter name and emergency contact\n Step 2: Open the camera\n Step 3: Press Q to switch off camera\n Step 4: exit'
                )

            # Open the about section
            def about():
                tkMessageBox.showinfo(
                    "About", 'Driver Cam version v1.0\n Made Using\n-OpenCV\n-Numpy\n-Tkinter\n-Pygame\n In Python 3'
                )

            # Contributors
            def contributors():
                tkMessageBox.showinfo(
                    "Contributors", 'Arathi Karuna M S\nBhagya M S\nDevapriya Suresh V\nHari Sapna Nair'
                )

            menu = Menu(cam_detect)
            cam_detect.config(menu=menu)
            sub_menu1 = Menu(menu)
            menu.add_cascade(label="Help", menu=sub_menu1)
            sub_menu1.add_command(label="How to use", command=help_box)

            sub_menu2 = Menu(menu)
            menu.add_cascade(label="About", menu=sub_menu2)
            sub_menu2.add_command(label="About the Software", command=about)
            sub_menu2.add_command(label="Made By", command=contributors)

            con_label = Label(cam_detect, text="Hi " + str_name + ", ready to drive safely...Click on the button to start detecting...\nHappy Journey",
                              width=70, font=("Helvitica 20 bold"), bg="#F5F5DC")
            con_label.place(relx=0.5, rely=0.2, anchor=CENTER)

            def web_detect():
                # Initializing the camera and taking the instance
                capture = cv2.VideoCapture(0)

                # Initializing the face detector and landmark detector
                detector = dlib.get_frontal_face_detector()

                # Detect 68 landmarks
                predictor = dlib.shape_predictor(
                    "shape_predictor_68_face_landmarks.dat")

                # Current state
                sleep = 0
                drowsy = 0
                active = 0

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
                def calculate_MAR(a, b, c, d, e, f, g, h):
                    # vertical lip distances
                    v_lip = dist.euclidean(
                        c, d) + dist.euclidean(e, f) + dist.euclidean(g, h)
                    h_lip = dist.euclidean(a, b)  # horizontal lip distance

                    # Mouth Aspect Ratio
                    MAR = v_lip/(3*h_lip)
                    return MAR

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

                        # Calculate MAR
                        MAR = calculate_MAR(landmarks[60], landmarks[64], landmarks[61], landmarks[67],
                                            landmarks[62], landmarks[66], landmarks[63], landmarks[65])

                        # Mixer settings
                        mixer.init()

                        # Conditions
                        if(left_blink == 0 or right_blink == 0):
                            sleep += 1
                            active = 0
                            drowsy = 0
                            if(sleep > 6):
                                mixer.music.load("audio/sleeping.wav")
                                mixer.music.set_volume(0.8)
                                mixer.music.play()
                                for i in range(0, 1):
                                    url = "https://www.fast2sms.com/dev/bulkV2"
                                    querystring = {
                                        "authorization": os.getenv('PROJECT_API_AUTHORIZATION'),
                                        "sender_id": "TEXTIND",
                                        "message": "Hello\n" + str_name + " is not driving safely.\nPlease call "+str_name+" immediately and ensure they are safe.",
                                        "language": "english",
                                        "route": "v3",
                                        "numbers": str_emergency
                                    }
                                    headers = {
                                        'cache-control': "no-cache"
                                    }
                                    response = requests.request(
                                        "GET", url, headers=headers, params=querystring)
                                    print(response.text)
                                break

                        elif(MAR > 0.43):
                            drowsy += 1
                            sleep = 0
                            active = 0
                            if(drowsy > 3):
                                mixer.music.load("audio/drowsy.mp3")
                                mixer.music.set_volume(0.8)
                                mixer.music.play()

                        elif(left_blink == 1 or right_blink == 1):
                            drowsy += 1
                            sleep = 0
                            active = 0
                            if(drowsy > 6):
                                mixer.music.load("audio/drowsy.mp3")
                                mixer.music.set_volume(0.8)
                                mixer.music.play()

                        else:
                            sleep = 0
                            drowsy = 0
                            active += 1
                            if(active > 6):
                                mixer.music.stop()

                        cv2.imshow("Frame", frame)

                    # Wait for any keyboard event to happen
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        capture.release()
                        break

            # Buttons
            button_open = Button(cam_detect, padx=5, pady=5, width=39, bg='#F5F5DC', fg='black',
                                 relief=GROOVE, command=web_detect, text='Open Camera & Detect', font=('helvetica 15 bold'))
            button_open.place(relx=0.5, rely=0.4, anchor=CENTER)

            button_exit = Button(cam_detect, padx=5, pady=5, width=5, bg='#F5F5DC', fg='black',
                                 relief=GROOVE, text='EXIT', command=cam_detect.destroy, font=('helvetica 15 bold'))
            button_exit.place(relx=0.5, rely=0.8, anchor=CENTER)

            cam_detect.mainloop()

            return


button_next = Button(root, text="SUBMIT", font=(
    "bold", 20), bg="#fbe878", fg="black", relief=RAISED, command=detecting, state=ACTIVE)
button_next.place(relx=0.67, rely=0.65)

button_next = Button(root, text="Exit", font=(
    "bold", 20), bg="#fbe878", fg="black", relief=RAISED, command=close_window, state=ACTIVE)
button_next.place(relx=0.5, rely=.8, anchor=CENTER)

# Run app
root.mainloop()
