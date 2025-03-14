import cv2
import mediapipe as mp
import time

vidObj = cv2.VideoCapture(2)

mpHands=mp.solutions.hands #like a formality , the constructer does 2 thing either scan and conf lv check, if the inital condition is false 
hands=mpHands.Hands(min_detection_confidence=0.70, min_tracking_confidence=0.50)

objCoordinates = mp.solutions.drawing_utils

pTime=0
cTime=0

while True:
    success, img = vidObj.read() 

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert the image in RGB format
    output =  hands.process(imgRGB) #output is being processed by the module
    #print(output.multi_hand_landmarks) #.multi_hand_landmarks gives the coordinates

    if output.multi_hand_landmarks: #if we found the  co-ordinates ratio is true
        for coordinates in output.multi_hand_landmarks:
            for id,lm in enumerate(coordinates.landmark):
                #print(id,lm) #id is given to every point by defalt and lm (landmark ) has x,y,z co-ordinates(ratio) of every id , x,y,z are the ratio of the image
                h,w,c = img.shape
                cx,cy= int(lm.x*w),int(lm.y*h) #### ### #### #### ##### #### ### #co-ordinate x,y or position of the center
                print(id,cx,cy)
                if id ==4:
                    side_length = 50  # Size of the square
                    top_left = (cx - side_length // 2, cy - side_length // 2)
                    bottom_right = (cx + side_length // 2, cy + side_length // 2)
                    cv2.rectangle(img, top_left, bottom_right, (0, 0, 250), 2)     
                if id == 8:
                    side_length = 50  # Size of the square
                    top_left = (cx - side_length // 2, cy - side_length // 2)
                    bottom_right = (cx + side_length // 2, cy + side_length // 2)
                    cv2.rectangle(img, top_left, bottom_right, (0, 0, 250), 2)  # 2 is the thickness of the outline
                if id ==12:
                    cv2.circle(img,(cx,cy),8,(0,255,255),cv2.FILLED)
                if id ==16:
                    cv2.circle(img,(cx,cy),8,(0,255,255),cv2.FILLED)
                if id ==20:
                    circle_radius=30
                    cv2.circle(img,(cx,cy),circle_radius,(250,255,0),thickness=2)
              


                objCoordinates.draw_landmarks(img, coordinates, mpHands.HAND_CONNECTIONS, objCoordinates.DrawingSpec(color=(255,0, 0), thickness=2, circle_radius=3),objCoordinates.DrawingSpec(color=(0, 255,0), thickness=1)) #connect all the points , saves our time to connect it manuall by using maths
    cTime =time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(0,35), cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),3)

    cv2.imshow("Image",img) #use to open simple webcam
    cv2.waitKey(2)



