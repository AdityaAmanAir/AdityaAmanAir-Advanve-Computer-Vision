import mediapipe as mp
import cv2
import time

class FaceMeshDetector():
    def __init__(self,staticMode=False, maxfaces=10, minDetectionCon=0.5, minTrackCon=0.1):
        self.staticMode= staticMode
        self.maxFaces=maxfaces
        self.minDetectionCon=minDetectionCon
        self.mintrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh=mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,  
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.mintrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0,255,0),thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result = self.faceMesh.process(self.imgRGB)
        faces=[]
        if self.result.multi_face_landmarks:
            for faceLms in self.result.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,self.drawSpec,self.drawSpec)
                face=[]
                for id,lm in enumerate (faceLms.landmark):
                    #print(lm)
                    ih, iw, ic=img.shape
                    x,y=int(lm.x*iw),int(lm.y*ih)
                    
                    face.append([x,y])
                faces.append(face)        
        return img, faces      

def main():
    cap=cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    pTime=0
    detector = FaceMeshDetector()
    
    while True:
        success, img = cap.read()
        img,faces = detector.findFaceMesh(img,True)
        if len(faces)!=0:
            print(f'Faces detected: {len(faces)}')
            for face in faces:
                for point in face[:1]:
                    print(f'Landmark: {point}')
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,f'FPS:{(int(fps))}',(0,35),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)

        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()