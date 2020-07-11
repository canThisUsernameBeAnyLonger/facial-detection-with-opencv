import cv2
#setting classifier file
detect = cv2.CascadeClassifier("C:/Users/Deepak/Practice/facialRecognitionWithPython/haarcascade_frontalface_default.xml")
#setting target image
inp_img = cv2.VideoCapture("C:/Users/Deepak/Practice/facialRecognitionWithPython/elon.jpg")
#reads the target image and returns dimension and result (boolean)
result, img = inp_img.read()
#takes a picture and converts it into a specified colour i.e grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#detect faces of diff size in input image
#takes the grayscale image, resize it and neighbouring code
faces = detect.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
	cv2.rectangle(img, (x,y), (x+w, y+h), (255,255, 0), 2)

cv2.imshow("Image",img)
#time you want to process the image (0=open as stable window)
cv2.waitKey(0)
inp_img.release()
cv2.destroyAllWindows()
