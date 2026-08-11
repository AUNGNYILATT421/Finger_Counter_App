import unittest

import numpy as np

import FingerCounter as fc
import HandTrackingModule as htm


def make_lm_list(overrides):
    lm = [[i, 100, 100] for i in range(21)]
    for idx, (x, y) in overrides.items():
        lm[idx] = [idx, x, y]
    return lm


class FingerCountTests(unittest.TestCase):
    def test_right_hand_all_fingers_extended(self):
        lm = make_lm_list({
            3: (100, 100), 4: (150, 100), 17: (120, 100),
            6: (0, 100), 8: (0, 50),
            10: (0, 100), 12: (0, 50),
            14: (0, 100), 16: (0, 50),
            18: (0, 100), 20: (0, 50),
        })
        self.assertEqual(fc.fingerCount(lm, 'Right'), 5)

    def test_right_hand_fist_is_zero(self):
        lm = make_lm_list({
            3: (100, 100), 4: (120, 100), 17: (150, 100),
            6: (0, 100), 8: (0, 100),
            10: (0, 100), 12: (0, 100),
            14: (0, 100), 16: (0, 100),
            18: (0, 100), 20: (0, 100),
        })
        self.assertEqual(fc.fingerCount(lm, 'Right'), 0)

    def test_left_hand_all_fingers_extended(self):
        lm = make_lm_list({
            3: (100, 100), 4: (50, 100), 17: (100, 100),
            6: (0, 100), 8: (0, 50),
            10: (0, 100), 12: (0, 50),
            14: (0, 100), 16: (0, 50),
            18: (0, 100), 20: (0, 50),
        })
        self.assertEqual(fc.fingerCount(lm, 'Left'), 5)

    def test_left_hand_only_index_extended(self):
        lm = make_lm_list({
            3: (90, 100), 4: (100, 100), 17: (100, 100),
            6: (0, 100), 8: (0, 50),
            10: (0, 100), 12: (0, 100),
            14: (0, 100), 16: (0, 100),
            18: (0, 100), 20: (0, 100),
        })
        self.assertEqual(fc.fingerCount(lm, 'Left'), 1)

    def test_right_hand_thumb_swap_branch(self):
        # thumb_tip_x (50) < pinky_root_x (200) triggers the 'swap' branch
        lm = make_lm_list({
            3: (100, 100), 4: (50, 100), 17: (200, 100),
            6: (0, 100), 8: (0, 100),
            10: (0, 100), 12: (0, 100),
            14: (0, 100), 16: (0, 100),
            18: (0, 100), 20: (0, 100),
        })
        self.assertEqual(fc.fingerCount(lm, 'Right'), 1)

    def test_left_hand_thumb_swap_branch(self):
        # thumb_tip_x (100) > pinky_root_x (50) triggers the 'swap' branch
        lm = make_lm_list({
            3: (80, 100), 4: (100, 100), 17: (50, 100),
            6: (0, 100), 8: (0, 100),
            10: (0, 100), 12: (0, 100),
            14: (0, 100), 16: (0, 100),
            18: (0, 100), 20: (0, 100),
        })
        self.assertEqual(fc.fingerCount(lm, 'Left'), 1)


class FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class FakeHandLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks


class FakeClassification:
    def __init__(self, label):
        self.label = label


class FakeHandedness:
    def __init__(self, label):
        self.classification = [FakeClassification(label)]


class FakeResults:
    def __init__(self, multi_hand_landmarks=None, multi_handedness=None):
        self.multi_hand_landmarks = multi_hand_landmarks
        self.multi_handedness = multi_handedness


class HandDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = htm.HandDetector(detectionCon=0.7)

    def test_classify_hand_before_find_hands_returns_empty(self):
        self.assertEqual(self.detector.classifyHand(), [])

    def test_find_positions_before_find_hands_returns_empty(self):
        self.assertEqual(self.detector.findPositions(), [])

    def test_find_hands_with_no_image_returns_none(self):
        self.assertIsNone(self.detector.findHands())

    def test_find_hands_processes_blank_frame_without_crashing(self):
        blank = np.zeros((100, 100, 3), dtype='uint8')
        img = self.detector.findHands(blank)
        self.assertIsNotNone(img)
        self.assertEqual(self.detector.classifyHand(), [])
        self.assertEqual(self.detector.findPositions(), [])

    def test_classify_hand_returns_labels_from_results(self):
        self.detector.results = FakeResults(
            multi_handedness=[FakeHandedness('Right'), FakeHandedness('Left')]
        )
        self.assertEqual(self.detector.classifyHand(), ['Right', 'Left'])

    def test_find_positions_scales_landmarks_to_pixel_coords(self):
        self.detector.img = np.zeros((100, 200, 3), dtype='uint8')  # h=100, w=200
        hand = FakeHandLandmarks([
            FakeLandmark(0.0, 0.0),
            FakeLandmark(0.5, 0.5),
            FakeLandmark(1.0, 1.0),
        ])
        self.detector.results = FakeResults(multi_hand_landmarks=[hand])

        lmList = self.detector.findPositions(handNo=0, draw=False)

        self.assertEqual(lmList, [[0, 0, 0], [1, 100, 50], [2, 200, 100]])

    def test_find_positions_handno_out_of_range_returns_empty(self):
        hand = FakeHandLandmarks([FakeLandmark(0.0, 0.0)])
        self.detector.results = FakeResults(multi_hand_landmarks=[hand])
        self.detector.img = np.zeros((10, 10, 3), dtype='uint8')

        self.assertEqual(self.detector.findPositions(handNo=1), [])


if __name__ == '__main__':
    unittest.main()
