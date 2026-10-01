# SPIN — Sign-driven Player & INstrumental mixer

Live, webcam-driven hand-gesture control over a song's vocals and instrumental — volume, EQ, playback — using a self-trained hand-pose classifier running on top of pretrained hand/face tracking, mixing pre-separated stems in real time.

No physical controller. No touchscreen. Your hands (and, experimentally, your eyes) are the mixer.

## What this is

A deep learning lab project. The graded/original contribution is the **gesture classifier**: a small MLP trained from scratch on a self-collected dataset of hand landmarks, reaching ~95% accuracy across 7 pose classes. Everything else — hand tracking, face/iris tracking, and song source-separation — uses pretrained, off-the-shelf models, explicitly **not** part of the contribution (see Credits).

## Features

- **Mode-select gestures** (hold ~1s): switch between Vocals & Instrumental, Bass & Treble, and Player control modes
- **Vocals & Instrumental** — pinch distance per hand sets each stem's volume continuously
- **Bass & Treble** — wrist-tilt angle per hand drives real low-shelf/high-shelf EQ filters (±12dB)
- **Player mode** — look left/right and hold to skip back/forward (experimental gaze control via a second pretrained face-mesh model)
- **Universal gestures** (work in any mode): a trained "stop sign" pose toggles play/pause; a wide-armed "Absolute Cinema" pose opens a file picker to load and auto-separate a new song
- **GPU-accelerated** source separation (CUDA) — about 20x faster than CPU
- Smooth crossfading between songs, live playback progress bar, on-screen debug telemetry for every control

## Credits

This project builds on top of, and is deeply grateful to:

- **[hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)** by [Kazuhito Takahashi](https://github.com/Kazuhito00) — the original base repository this project started from (webcam landmark logging, the keypoint/point-history classifier scaffolding, and training notebooks). Licensed under Apache 2.0; see [LICENSE](LICENSE). The original README is preserved at [BASE_REPO_README.md](BASE_REPO_README.md).
- **[English translation](https://github.com/kinivi/hand-gesture-recognition-mediapipe)** by [Nikita Kiselov](https://github.com/kinivi).
- **[MediaPipe](https://developers.google.com/mediapipe)** (Google) — pretrained hand landmark and face mesh (iris) tracking. Not trained or fine-tuned here.
- **[Demucs](https://github.com/facebookresearch/demucs)** (Meta AI Research) — pretrained music source separation (`htdemucs`). Not trained here.

The gesture-pose classifier (dataset, preprocessing, architecture, training) and the entire live mixer application (`mixer_app.py`, `audio_engine.py`, `song_loader.py`) are original work for this project.

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install mediapipe==0.10.21 tensorflow opencv-python scikit-learn pandas matplotlib seaborn scipy sounddevice soundfile demucs torch
```

`mediapipe` must stay below `1.0.0` — the 1.x line replaced the Solutions API (`mp.solutions.hands`, `mp.solutions.face_mesh`) this project uses with a different Tasks API.

For GPU-accelerated song separation (optional but recommended — ~20x faster), install a CUDA build of PyTorch matching your GPU instead of the default CPU build; see [pytorch.org](https://pytorch.org/get-started/locally/).

## Usage

**Run the live mixer:**
```bash
python mixer_app.py
```
Picks a song via file dialog on launch (separates it automatically if it hasn't been used before — cached after that), then opens the gesture-controlled mixer.

**Record your own gesture data / retrain:**
```bash
python app.py
```
Press `k` to log static poses, `h` to log motion data; tap a number key to arm/disarm continuous logging for that class. See `keypoint_classification_EN.ipynb` for the training pipeline.

## Gesture reference

| Pose | Meaning |
|---|---|
| `one` (index finger) | Select Player mode |
| `two` (peace sign) | Select Vocals & Instrumental mode |
| `three` (3 fingers) | Select Bass & Treble mode |
| pinch open/closed | Set volume (Vocals & Instrumental mode) |
| neutral / thumbs-up, tilted | Set EQ boost/cut (Bass & Treble mode) |
| flat palm, fingers together ("stop sign") | Play/pause (any mode) |
| both arms wide, open hands ("Absolute Cinema") | Load a new song (any mode) |
| look left/right, hold | Skip back/forward (Player mode, `Space` to toggle) |

## Project structure

- `mixer_app.py` — the live gesture-controlled mixer application
- `audio_engine.py` — real-time stem mixing + EQ (sounddevice, scipy)
- `song_loader.py` — song picking, caching, Demucs separation
- `app.py` — gesture data recording tool (adapted from the base repo)
- `model/` — trained classifiers (TFLite) and datasets
- `keypoint_classification_EN.ipynb` — training notebook for the pose classifier
- `project_summary.json` — full write-up of dataset, architecture, metrics, and methodology for the report

## License

Apache 2.0, inherited from the base repository — see [LICENSE](LICENSE).
