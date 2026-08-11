# Finger Counter

A real-time hand-tracking application that detects raised fingers from a webcam feed and displays the running total on screen. Built with [OpenCV](https://opencv.org/) for video capture/rendering and [MediaPipe](https://google.github.io/mediapipe/) for hand landmark detection, and supports counting fingers on both hands simultaneously.

## How it works

1. **`HandTrackingModule.py`** wraps MediaPipe's `Hands` solution in a `HandDetector` class that:
   - Detects hands in a frame and draws landmarks (`findHands`)
   - Reports whether each detected hand is `Left` or `Right` (`classifyHand`)
   - Returns the pixel coordinates of all 21 landmarks for a given hand (`findPositions`)
2. **`FingerCounter.py`** drives the webcam loop:
   - For each of the 5 fingertip landmarks (thumb, index, middle, ring, pinky), it compares the tip position against a lower joint to decide whether the finger is extended.
   - The thumb is handled specially since its motion is horizontal, and the comparison direction is flipped depending on the hand's label (`Left`/`Right`) so it works correctly no matter which hand is shown or which way it's facing the camera.
   - The extended-finger count across all detected hands is summed and rendered on the video frame (both as a number and as the matching image from `FingerImages/`), along with a live FPS counter.

## Project structure

```
.
├── FingerCounter.py         # Main application entry point (webcam loop, finger-counting logic)
├── HandTrackingModule.py    # Reusable MediaPipe hand-detection wrapper (HandDetector class)
└── FingerImages/            # Overlay images (1.jpg – 6.jpg) shown for each finger count
```

## Requirements

- Python 3.8+
- A connected webcam
- [OpenCV](https://pypi.org/project/opencv-python/) (`opencv-python`)
- [MediaPipe](https://pypi.org/project/mediapipe/)

## Platform support

Developed and tested on **macOS**. The code has no OS-specific paths or APIs (paths are built with `os.path.join`, camera/GUI calls are plain `cv2`), and every pinned dependency — including the trickier native ones (`mediapipe`, `aiortc`/`av`/`cryptography` used by `streamlit-webrtc`) — publishes prebuilt **Windows** wheels for Python 3.10, so `pip install -r requirements.txt` is expected to succeed there without needing a compiler. That said, it hasn't been run end-to-end on Windows, so treat Windows support as untested-but-expected rather than verified. If `cv2.VideoCapture(0)` opens slowly or fails to find the camera on Windows, try `cv2.VideoCapture(0, cv2.CAP_DSHOW)` in `FingerCounter.py`.

## Setup

1. Clone or download this repository.
2. (Recommended) create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the app

From the project root:

```bash
python FingerCounter.py
```

- A window titled **Image** will open showing the webcam feed with hand landmarks drawn and the current finger count overlaid.
- Press **Esc** to quit.

## Testing

`test.py` covers the finger-counting math in `FingerCounter.fingerCount` (thumb logic for both hands and both orientation branches, full-fist and full-open poses) and `HandTrackingModule.HandDetector`'s guards and coordinate scaling (using synthetic landmarks, no real webcam/hand image needed). Run it with:

```bash
python test.py
```

`FingerCounter.py` now guards its webcam loop behind `if __name__ == "__main__":` so it can be imported by the test suite without opening a camera.

## Known issues

- `requirements.txt` pins `mediapipe<1.0.0`. MediaPipe 1.0.0 removed the legacy `mp.solutions` API that `HandTrackingModule.py` relies on, so installing the latest release breaks hand detection at startup (`AttributeError: module 'mediapipe' has no attribute 'solutions'`).
- `mediapipe` pulls in `opencv-contrib-python` as its own dependency — this project relies on that rather than declaring a separate `opencv-python`, since `opencv-contrib-python` is a strict superset (same core API plus extra modules) and an additional `opencv-python` entry would just get installed then silently overwritten in the shared `cv2/` namespace. `requirements.txt` pins `opencv-contrib-python` directly so it resolves deterministically instead of drifting to whatever's newest on PyPI; if you bump `mediapipe`, re-check this pin still matches a version it's compatible with.
- The webcam is opened with `cv2.VideoCapture(0)`, which uses the system's default camera. Change the index if you have multiple cameras and need a different one.
- `FingerImages/` only has icons for counts 1–6. With two hands the detector can report up to 10 fingers; counts of 0 or above 6 show the number overlay only (no image).

## Deployment notes

`FingerCounter.py` is a local, GUI-based (`cv2.imshow`) desktop script — it must be run on a machine with a display and an accessible webcam.

For deploying as a web app, use **`app.py`** instead: a Streamlit + [`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc) front end that streams the viewer's browser webcam to the server for processing, reusing `HandTrackingModule` and `FingerCounter.fingerCount` as-is. Run it locally with:

```bash
streamlit run app.py
```

When deploying to **Streamlit Community Cloud**, point it at `app.py`. The full (non-headless) `opencv-python` build that `FingerCounter.py` needs for `cv2.imshow` requires system-level graphics libraries (`libGL.so.1`, etc.) that Streamlit Cloud's minimal container doesn't include by default — without them the deploy fails with `ImportError: libGL.so.1: cannot open shared object file`. `packages.txt` at the repo root lists the apt package (`libgl1`) that Streamlit Cloud installs before the Python build, which fixes this without needing to swap to `opencv-python-headless` (which would break `cv2.imshow` for local desktop use).
