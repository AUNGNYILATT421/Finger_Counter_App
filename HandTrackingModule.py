import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.img = None
        self.results = None
        self.labels = []

    def findHands(self, img=None, draw=True):
        if img is not None:
            self.img = img

        if self.img is None:
            return None

        imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(self.img, handlms, self.mpHands.HAND_CONNECTIONS)

        return self.img
    
    def classifyHand(self):
        # set self.labels to []
        self.labels = []

        if self.results is None:
            return self.labels

        if self.results.multi_handedness:
            for hand in self.results.multi_handedness:
                self.labels.append(hand.classification[0].label)

        return self.labels

    def findPositions(self, handNo=0, draw=True):
        # set self.lmList to []
        self.lmList = []

        if self.results is None or not self.results.multi_hand_landmarks:
            return self.lmList

        if handNo >= len(self.results.multi_hand_landmarks):
            return self.lmList

        myHand = self.results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            h, w, _ = self.img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            self.lmList.append([id, cx, cy])
            if draw:
                cv2.circle(self.img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        return self.lmList
