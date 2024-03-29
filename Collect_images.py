# -*- coding: utf-8 -*-
'''
@author : Swagat Sourav

'''
import cv2
import urllib
import numpy as np
face_data = r"haarcascade_frontalface_default.xml"
classifier = cv2.CascadeClassifier(face_data)

url = "http://192.168.29.191:8080/shot.jpg"

data =[]

while len(data)<100:
    image_from_url = urllib.request.urlopen(url)
    frame =np.array(bytearray(image_from_url.read()),np.uint8)
    frame = cv2.imdecode(frame,-1)
    # 1.5 here refers to the scaling factor(how much the image should be reduced at each image scale)
    #5 refers to the no.of neighbors the rectangle should have.
    # detectMultiScale returns rectangle of x,y,w,h 
    faces = classifier.detectMultiScale(frame,1.5,5)
    
    if len(faces)>0:
         for x,y,w,h in faces:
             face_frame = frame[y:y+h,x:x+w].copy()
             cv2.imshow("Only Face",face_frame)
             
             if len(data)<=100:
                 print(len(data)+1,"/100")
                 
                 data.append(face_frame)
                 
             else: 
                break
    cv2.imshow("capture",frame)
    if cv2.waitKey(30) == ord('a'):
        break
    
cv2.destroyAllWindows()


if len(data) == 100:
    name = input("Enter the name ")
    for i in range(100):
        cv2.imwrite("image/"+name+"_"+str(i+1)+".jpg",data[i])
    
    print("Complete")

else:
    
    print("Need More Data")
