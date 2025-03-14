import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode=False,maxHands =2,detectionCon=0.70, trackCon=0.50):
        self.mode=mode
        self.maxHands=maxHands 
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpHands=mp.solutions.hands 
        self.hands = self.mpHands.Hands(
    static_image_mode=self.mode,
    max_num_hands=self.maxHands,
    min_detection_confidence=self.detectionCon,
    min_tracking_confidence=self.trackCon
)

        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        self.result =  self.hands.process(imgRGB) 
        if self.result.multi_hand_landmarks: 
            for coordinates in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, coordinates, self.mpHands.HAND_CONNECTIONS, self.mpDraw.DrawingSpec(color=(255,0, 0), thickness=2, circle_radius=3),self.mpDraw.DrawingSpec(color=(0, 255,0), thickness=1)) 
        return  img   

    def findposition(self,img,handNo=0,draw=True):
        lmList=[]
        if self.result.multi_hand_landmarks:
            myHand=self.result.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy= int(lm.x*w),int(lm.y*h) 
                lmList.append([id,cx,cy])
                if id ==4:
                    side_length = 50  
                    top_left = (cx - side_length // 2, cy - side_length // 2)
                    bottom_right = (cx + side_length // 2, cy + side_length // 2)
                    cv2.rectangle(img, top_left, bottom_right, (0, 0, 250), 2)     
                if id == 8:
                    side_length = 50  
                    top_left = (cx - side_length // 2, cy - side_length // 2)
                    bottom_right = (cx + side_length // 2, cy + side_length // 2)
                    cv2.rectangle(img, top_left, bottom_right, (0, 0, 250), 2) 
                if id ==12:
                    cv2.circle(img,(cx,cy),8,(0,255,255),cv2.FILLED)
                if id ==16:
                    cv2.circle(img,(cx,cy),8,(0,255,255),cv2.FILLED)
                if id ==20:
                    circle_radius=30
                    cv2.circle(img,(cx,cy),circle_radius,(250,255,0),thickness=2)
        return lmList     
                
def main():
    pTime=0
    cTime=0
    cap = cv2.VideoCapture(2)
    detector=handDetector()
    while True:
        success, img = cap.read() 
        img = detector.findHands(img)
        lmList=detector.findposition(img)
        if len(lmList) !=0:
            print(lmList[8])

        cTime =time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps)),(0,35), cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),3)

        cv2.imshow("Image",img) 
        cv2.waitKey(2)

if __name__=="__main__" :
    main()   
