import cv2



img = cv2.imread("RES/1.jpg" , 1 )




Face_Detection = cv2.CascadeClassifier("RES/haarcascade_frontalface_default.xml")
Eye_Detection = cv2.CascadeClassifier("RES/haarcascade_eye.xml")
Smile_Detection = cv2.CascadeClassifier("RES/haarcascade_smile.xml")

faces_rect = Face_Detection.detectMultiScale(img, scaleFactor=1.1, minNeighbors=9)
Eye_rect = Eye_Detection.detectMultiScale(img, scaleFactor=1.1, minNeighbors=9)
Smile_rect = Smile_Detection.detectMultiScale(img, scaleFactor=1.8, minNeighbors=30 )



for (x, y, w, h) in faces_rect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
for (x, y, w, h) in Eye_rect:
    cv2.circle(img , (int(x+w / 2),int(y+h / 2)) ,35 , (255,100,200) , 2)
for (x, y, w, h) in Smile_rect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 150), thickness=2)

cv2.imshow("img" , img)
cv2.waitKey()





