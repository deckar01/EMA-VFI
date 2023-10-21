# Holocam Bilinear Interpolation

Bilinear time-then-space interpolation for jittery, sparse camera arrays.

## Strategy

1. Model the video frames from a sparse camera array as set of parallel timelines.
2. Interpolate the frame rate of each timelines.
3. Interpolate the camera angles between timelines.

```
  -- space -->
|
|    A
t    |       C
i    |       |
m    E----G--F
e    |       |
|    B       |
|            |
v            D
```

|      | | space    |        | | time     |        |
|-----:|-|----------|--------|-|----------|--------|
| ABCD | | linear   | sparse | | variable | sparse |
|   EF | | linear   | sparse | | uniform  | dense  |
|    G | | uniform  | dense  | | uniform  | dense  |

## Performance

Inference calls within shared edges are optimized by caching feature extraction "bones" between image pairs. Frames are produced in the order needed for streaming to a video encoder. Cached VRAM "bones" are freed as soon the interpolation edge is no longer part of the window containing the current frame. The VRAM buffer used on top of the pytorch model is approximately `|image + features| * (2 * n + 3)`, where `n` is the number of timelines (input cameras).

## Dependencies

1. Download the "ours_t.pkl" model into `ckpt` per [EMA-VFI]
2. Install dependencies `pip install -r requirements.txt` (modify [pytorch] per platform)

[EMA-VFI]: https://github.com/MCG-NJU/EMA-VFI#sunglassesplay-with-demos
[pytorch]: https://pytorch.org/get-started/locally/

## Usage

```py
python -m bilinear --help
```
