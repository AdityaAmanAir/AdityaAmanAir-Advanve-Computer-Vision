import cv2
import mediapipe as mp
import time

mpPose=mp.solutions.pose
pose=mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

pTime=0
cTime=0
cap=cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS, mpDraw.DrawingSpec(color=(255,0, 0), thickness=2, circle_radius=3),mpDraw.DrawingSpec(color=(0, 255,0), thickness=1))
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c =img.shape
            print(id, lm)#ratio of the image
            cx,cy=int(lm.x*w), int(lm.y*h)#pixel value
            cv2.circle(img,(cx,cy),color=(255,0, 0), thickness=1, radius=3)
            if id ==0:
                    side_length = 120  # Size of the square
                    top_left = (cx - side_length // 2, cy - side_length // 2)
                    bottom_right = (cx + side_length // 2, cy + side_length // 2)
                    cv2.rectangle(img, top_left, bottom_right, (0, 0, 250), 2)     
            
    
    cTime =time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv2.putText(img,str(int(fps)),(0,35),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)