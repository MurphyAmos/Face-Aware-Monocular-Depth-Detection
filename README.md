# Face-Aware-Monocular-Depth-Detection
Applies monocular depth estimation to detected faces in a video, overlaying a real-time depth heatmap onto each face rather than depth-mapping the entire frame.

## What it does

Instead of running depth estimation on every pixel of every frame, this project narrows the scope: detect faces first, then only run depth inference on those regions. Each face gets its own depth-mapped heatmap, blended back onto the original frame, so the rest of the scene stays untouched while the detected faces get a live depth overlay.

## How it works

1. **Frame capture & skipping.** OpenCV reads the source video, using `cap.grab()` to skip undecoded frames and `cap.read()` only on the frames actually being processed, avoiding the cost of decoding frames that get thrown away anyway.
2. **Face detection.** Each processed frame is passed through `face_recognition` to locate faces, and each detected face is cropped out individually.
3. **Depth inference.** Every cropped face is passed through a HuggingFace `depth-estimation` pipeline running [Depth-Anything V2 (Small)](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), producing a per-face depth map.
4. **Temporal smoothing.** Each face's depth map is smoothed against its previous frame using an exponential moving average, reducing the frame-to-frame flicker that comes from estimating depth independently every frame.
5. **Heatmap blending.** The smoothed depth map is converted to a JET colormap and alpha-blended back onto the original face region, rather than replacing the frame outright.
6. **Video encoding.** Blended frames are written to the output video at a framerate scaled to the source capture rate, adjusted for the frame-skip factor.

Note: this is near-real-time, not hard real-time. The processing is bound by inference speed, so if inference can't keep pace with the camera or source video, the live preview will lag slightly behind.

## Demo

<table>
  <tr>
    <td><img src="demos/ay5rgb.gif" width="400"/></td>
    <td><img src="demos/ay5rf9.gif" width="400"/></td>
  </tr>
  <tr>
    <td align="center">Original</td>
    <td align="center">Face-Depth Overlay Output</td>
  </tr>
</table>

## Tech stack

* Python
* OpenCV for video I/O, frame decoding/encoding, and frame skipping
* face_recognition for face detection
* HuggingFace Transformers for the depth-estimation model pipeline
* Depth-Anything V2 for monocular depth estimation
* NumPy for depth map smoothing and blending

## Setup

Clone the repository:

```bash
git clone https://github.com/MurphyAmos/Face-Aware-Monocular-Depth-Detection.git
cd Face-Aware-Monocular-Depth-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Hugging Face token:

```bash
export HF_TOKEN="your_token_here"      # macOS/Linux
setx HF_TOKEN "your_token_here"        # Windows
```

## Usage

Update the video source path to point to your own file, then run:

```bash
python main.py
```

Output is written to `test.mp4` in the working directory. A `test` flag is available for a lightweight demo mode that generates a faux depth map (grayscale-based) without running the real model, useful for quick visual checks without GPU/model overhead.

## Known limitations & Next Fixes
* **Face tracking is index-based, not identity-based.** Temporal smoothing is keyed to a face's position in the detection list each frame, not a persistent identity. If face order shifts between frames (someone leaves frame, new face detected in a different order), smoothing state can briefly blend across different faces instead of following the same one.
* **Only detected faces are depth-mapped.** The rest of the frame is left untouched by design, this is a face-focused overlay, not a full-frame depth pipeline.
* **No UI.** Currently there is no UI, fine for a quick demo, but a bit tedious for adjusting parameters like alpha, frame skip, or target resolution.

## Fixed & Updates
* **Frame-to-frame flicker (per-face).** Solved with exponential moving average smoothing per face, applied independently from the full-frame version of this fix in the original depth pipeline project.
* **Frame skipping efficiency.** Switched from decode-then-discard to `cap.grab()` for skipped frames, avoiding the cost of decoding frames that were never going to be used.

## Motivation

After solving frame-to-frame flicker in the full-frame depth pipeline, the natural next question was whether the same idea could be narrowed and applied more precisely, tracking depth on just the parts of a frame that actually matter, like faces, instead of the whole scene.

