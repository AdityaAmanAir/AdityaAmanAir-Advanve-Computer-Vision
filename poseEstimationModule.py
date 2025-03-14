import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self,mode=False,upBody =False, smooth=True,detectionCon=0.7,trackCon=0.5):
        self.mode =mode
        self.upBody=upBody
        self.smooth=smooth
        self.detectionCon=detectionCon
        self.trackCon=trackCon


        self.mpPose=mp.solutions.pose
        self.pose = self.mpPose.Pose(
        static_image_mode=self.mode,
        smooth_landmarks=self.smooth,
        min_detection_confidence=self.detectionCon,
        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS, self.mpDraw.DrawingSpec(color=(255,0, 0), thickness=2, circle_radius=3),self.mpDraw.DrawingSpec(color=(0, 255,0), thickness=1))
        return img  

    def findPosition(self,img,draw=True):
        lmList=[]
        if self.results.pose_landmarks :    
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c =img.shape
                #print(id, lm)
                cx,cy=int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),color=(255,0, 0), thickness=1, radius=3)
                    if id ==0:
                        side_length = 120 
                        top_left = (cx - side_length // 2, cy - side_length // 2)
                        bottom_right = (cx + side_length // 2, cy + side_length // 2)
                        cv2.rectangle(img, top_left, bottom_right, (0, 0, 250), 2)     
        return lmList    
    
    


def main():
    cap=cv2.VideoCapture(0)
    cTime=0
    pTime=0
    detector =poseDetector()
    while True:
        success, img = cap.read()
        detector.findPose(img)
        lmList=detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[14])
            cv2.circle(img,(lmList[14][1],lmList[14][2]),15,(0,0,255),cv2.FILLED)
        cTime =time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
    
        cv2.putText(img,str(int(fps)),(0,35), cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),3)
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__=="__main__":
     main()    