import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


# from PIL import ImageGrab

path = 'C:/Training_images' #importing the images from it's path
images = []                 #
classNames = []
myList = os.listdir(path)       #grabbing the list of files/images in this path
print(myList)
for cl in myList:               #
    curImg = cv2.imread(f'{path}/{cl}')     #reading the images from the path
    images.append(curImg)               #appending the current image
    classNames.append(os.path.splitext(cl)[0])      #removing the extensions(.jpg/.png) by grabbing the first element of each picture name
print(classNames)


def findEncodings(images):          # Function or method to compute facial encodings
    encodeList = []                 #an empty list that would have all the encodining later on


    for img in images:          # looping through all the images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      #converting the images to RGB
        encode = face_recognition.face_encodings(img)[0]    #finding the facial encodings
        encodeList.append(encode)       #appending the encodings to the encodingList
    return encodeList


def markAttendance(name):
    with open('C:/attendance.csv', 'r+') as f:
        myDataList = f.readlines()


        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')      #prints this after some time of finding the encodings.

cap = cv2.VideoCapture(0)  #initializing the camera

while True:         #looping through each frame one by one
    success, img = cap.read()
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)       #reducing the size of the image for faster processing in realtime
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)        #converting the small imgs to RGB

    facesCurFrame = face_recognition.face_locations(imgS)       # finding the locations in small images
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):     #iterating through all the faces that we have found in our current frame and compare them with those found before
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)   #matching/comparing the faces
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)   #comparing the face distances
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()   #the names of the pictures in upper case
            print(name)
            #creating the bounding box around the face
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4     ## TO Rescale the image back to scale to fit the bounding box ont the face locations.
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key != ord("q"):
        continue
    break

cv2.destroyAllWindows()
