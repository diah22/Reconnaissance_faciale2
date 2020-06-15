from tkinter import *
import cv2
import tkinter.messagebox as tkmsg
import os
import shutil
import pandas as pd
import tkinter.font as tkFont
import csv
import numpy as np
from PIL import Image, ImageTk
import tkinter.ttk as ttk
from pathlib import Path
import time
import datetime
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade= cv2.CascadeClassifier(cascadePath)
font= cv2.FONT_HERSHEY_SIMPLEX

class Interface:
    def __init__(self):
        self.root= Tk()
        self.root.geometry('800x543+150+0')
        self.root.configure(bg='steelblue')
        self.root.title('Face recognition')
        self.root.resizable(width= False, height=False)
        self.count=0
        self.names=[]
        self.users={}
        self.fontImage= PhotoImage(file= "Background\sys.png")
        self.fontauth1= PhotoImage(file= "Background\sys2.png")

    def __corps__(self):
        self.container= Canvas(self.root, width= 800, height= 600, bg='steelblue')
        self.container.place(relx=-0.001, rely=-0.001)
        self.container.create_image(400,290, image= self.fontImage)
        self.cap= Button(self.container, bg= 'darkslategray', text='Authentification',cursor='hand2',fg='white', width=30, height=3, command=self.auth)
        self.cap.bind("<Enter>", self.auth_mouse)
        self.cap.bind("<Leave>", self.auth_leave)
        self.cap.place(relx=0.37, rely=0.25)
        self.train= Button(self.container, bg='darkslategray', text='Train',fg='white',cursor='hand2', width=30, height=3, command=self.training)
        self.train.place(relx=0.37, rely=0.40)
        self.train.bind("<Enter>" ,self.train_mouse)
        self.train.bind("<Leave>", self.train_leave)
        self.recognize= Button(self.container, bg='darkslategray',fg='white',cursor='hand2', text='Identification',width=30, height=3, command=self.recognition)
        self.recognize.place(relx=0.37, rely=0.55)
        self.recognize.bind("<Enter>", self.recognize_mouse)
        self.recognize.bind("<Leave>", self.recognize_leave)

    def auth_mouse(self,e):
        self.cap['bg']= 'slategray'
    def auth_leave(self,e):
        self.cap['bg']= 'darkslategray'

    def train_mouse(self,e):
        self.train['bg']= 'slategray'
    def train_leave(self, e):
        self.train['bg'] = 'darkslategray'

    def recognize_mouse(self,e):
        self.recognize['bg']= 'slategray'
    def recognize_leave(self, e):
        self.recognize['bg'] = 'darkslategray'

    def auth(self):
        self.fenfille= Toplevel(self.root)
        self.fenfille.title('Configuration nom')
        self.fenfille.geometry('500x250+0+0')
        self.fenfille.resizable(width=False, height=False)
        self.canv= Canvas(self.fenfille, width= 500, height=250, bg='steelblue');
        self.canv.place(relx=-0.001, rely=-0.001)
        self.fontauth= self.fontauth1.subsample(2,2)
        self.canv.create_image(250,117,image= self.fontauth)
        self.canv.create_text(250,65, text="QUI ETES VOUS ? ", fill='white')
        self.valNom= StringVar()
        self.entry_name= Entry(self.fenfille, width=35, textvariable=self.valNom)
        self.entry_name.place(relx=0.29, rely=0.33)
        self.valide= Button(self.fenfille, text='OK', width=10, highlightthickness=0, command=self.detect)
        self.valide.place(relx=0.40, rely=0.55)

    def detect(self):
        self.count+=1
        self.Id= self.count
        self.name= self.valNom.get()
        self.fenfille.destroy()
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video widht
        cam.set(4, 480)  # set video height

        # Define min window size to be recognized as a face
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)
        nbImg=0
        while True:
            ret, img = cam.read()
            # img = cv2.flip(img, -1) # Flip vertically

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )
            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y),(x+w, y+h),(255,0,0),2)
                nbImg+=1
                cv2.imwrite("images/"+self.name+"."+str(self.count)+"."+str(nbImg)+".jpg", gray[y:y+h, x:x+w])
                cv2.imshow('image', img)
            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif nbImg >= 30:  # Take 30 face sample and stop video
                break
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
        self.users[self.count-1]= {'id': self.count, 'nom: ':self.name}
        self.names.append(self.name)
        print(self.users)

    def getImagesAndLabels(self, path):

        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = faceCascade.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        return faceSamples, ids

    def training(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces, ids = self.getImagesAndLabels('images')
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.write('trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

    def recognition(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        # iniciate id counter
        id = 0

        # names related to ids: example ==> Marcelo: id=1,  etc
        #names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']

        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video widht
        cam.set(4, 480)  # set video height

        # Define min window size to be recognized as a face
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)
        # getName= pd.read_csv('UserDetails/userdetails.csv')
        while True:

            ret, img = cam.read()
            # img = cv2.flip(img, -1) # Flip vertically

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                # Check if confidence is less them 100 ==> "0" is perfect match
                if (confidence < 100):
                    id = self.names[id-1]
                    # aa= getName.loc[getName['face_id']==id]['Name'].values
                    # tt= str(id)+"-"+aa
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))


                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            cv2.imshow('camera', img)

            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

    def __final__(self):
        self.root.mainloop()

if (__name__ =='__main__'):
    fen= Interface()
    fen.__corps__()
    fen.__final__()
