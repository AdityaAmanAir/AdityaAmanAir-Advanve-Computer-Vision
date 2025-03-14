import mediapipe as mp
import cv2
import time

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

pTime=0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=10,min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(color=(0,255,0),thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    imgRGB= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = faceMesh.process(imgRGB)
    if result.multi_face_landmarks:
        for faceLms in result.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,drawSpec,drawSpec)
            
            for id,lm in enumerate (faceLms.landmark):
                #print(lm)
                ih, iw, ic=img.shape
                x,y=int(lm.x*iw),int(lm.y*ih)
                print(id,x,y)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS:{(int(fps))}',(0,35),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)

