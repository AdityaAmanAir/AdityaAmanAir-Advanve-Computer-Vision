import cv2
import mediapipe as mp
import time

cap =cv2.VideoCapture(0)
cTime=0
pTime=0
mpFaceDetection =mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
faceDetection=mpFaceDetection.FaceDetection(0.5,1)

while True:
    success, img=cap.read()
    
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(img,detection,mpDraw.DrawingSpec(color=(0, 255, 255), thickness=8),mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2))
            #print(id,detection)
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)
            bboxC= detection.location_data.relative_bounding_box
            ih, iw, ic =img.shape
            bbox=(int(bboxC.xmin*iw),int(bboxC.ymin*ih),
            int(bboxC.width*iw),int(bboxC.height*ih))
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img,f'{int(detection.score[0]*100)}%',
                (bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
            
            x,y,w,h=bbox
            x1,y1=x+w,y+h
            l=10
            t=5
            rt=1
            cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 255), rt)
            cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
            cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)

            cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
            cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)

            cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
            cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

            cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
            cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
            
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS:{(int(fps))}',(0,35),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
    cv2.imshow("Image",img)
    cv2.waitKey(1)