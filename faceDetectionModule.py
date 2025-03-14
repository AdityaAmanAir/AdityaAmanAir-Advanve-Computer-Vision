import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.mpFaceDetection=minDetectionCon
        self.mpFaceDetection =mp.solutions.face_detection
        self.mpDraw=mp.solutions.drawing_utils
        self.faceDetection=self.mpFaceDetection.FaceDetection(minDetectionCon,1)


    def findFaces(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.faceDetection.process(imgRGB)
        #print(self.results)
        bboxs =[]

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC= detection.location_data.relative_bounding_box
                ih, iw, ic =img.shape
                bbox=(int(bboxC.xmin*iw),int(bboxC.ymin*ih),
                int(bboxC.width*iw),int(bboxC.height*ih))
                bboxs.append([id,bbox,detection.score])
                self.fancyDraw(img,bbox)

                cv2.putText(img,f'{int(detection.score[0]*100)}%',
                (bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
        return img,bboxs
    def fancyDraw(self,img,bbox,l=10,t=5,rt=1):
        x,y,w,h=bbox
        x1,y1=x+w,y+h
        cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 255), rt)
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)

        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)

        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img


def main():
    cap =cv2.VideoCapture(0)
    cTime=0
    pTime=0
    detector=FaceDetector()

    while True:
        success, img =cap.read()
        img, bbox =detector.findFaces(img)
       
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,f'FPS:{(int(fps))}',(0,35),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()    