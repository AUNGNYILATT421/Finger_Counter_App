import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 720, 560

tipIds = [4, 8, 12, 16, 20]

def fingerCount(lmList, label):
    fingers = []
    thumb_tip_x = lmList[tipIds[0]][1]
    pinky_root_x = lmList[17][1]
    swap = None

    # Thumb
    if label == 'Right':
        swap = False if thumb_tip_x >= pinky_root_x else True
        if swap:
            fingers.append(1 if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1] else 0)
        else:
            fingers.append(1 if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] else 0)
    else:
        swap = False if thumb_tip_x <= pinky_root_x else True
        if swap:
            fingers.append(1 if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] else 0)
        else:
            fingers.append(1 if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1] else 0)      

        # Fingers 
    for i in range(1, len(tipIds)):
        if lmList[tipIds[i]][2] < lmList[tipIds[i] - 2][2]: fingers.append(1)
        else: fingers.append(0)

    return fingers.count(1)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = htm.HandDetector(detectionCon=0.7)

    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            print('Cannot grab frame')
            break

        img = detector.findHands(img)
        labels = detector.classifyHand()
        totalHands = min(
            len(labels),
            len(detector.results.multi_hand_landmarks) if detector.results and detector.results.multi_hand_landmarks else 0
        )
        totalFingers = 0
        if totalHands != 0:
            for i in range(totalHands):
                lmList = detector.findPositions(handNo=i)

                if len(lmList) != 0:
                    totalFingers += fingerCount(lmList, labels[i])

            cv2.rectangle(img, (20, 225), (250, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(totalFingers), (60, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (400, 65), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

        cv2.imshow('Image', img)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
