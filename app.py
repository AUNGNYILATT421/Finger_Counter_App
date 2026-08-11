import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

import HandTrackingModule as htm
from FingerCounter import fingerCount

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class FingerCountProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = htm.HandDetector(detectionCon=0.7)
        self.pTime = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = self.detector.findHands(img)
        labels = self.detector.classifyHand()
        totalHands = min(
            len(labels),
            len(self.detector.results.multi_hand_landmarks)
            if self.detector.results and self.detector.results.multi_hand_landmarks
            else 0,
        )

        totalFingers = 0
        for i in range(totalHands):
            lmList = self.detector.findPositions(handNo=i)
            if lmList:
                totalFingers += fingerCount(lmList, labels[i])

        if totalHands:
            cv2.rectangle(img, (20, 225), (250, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(totalFingers), (60, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

        cTime = time.time()
        fps = 1 / (cTime - self.pTime) if cTime != self.pTime else 0
        self.pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (400, 65), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.set_page_config(page_title="Finger Counter", page_icon="✋")
st.title("Finger Counter")
st.write("Click **Start**, allow webcam access, and show your hand(s) to the camera.")

webrtc_streamer(
    key="finger-counter",
    video_processor_factory=FingerCountProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
