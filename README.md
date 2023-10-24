# Holocam Bilinear Interpolation

Bilinear time-then-space interpolation for jittery, sparse camera arrays.

## Strategy

1. Model the video frames from a sparse camera array as a set of parallel timelines.
2. Interpolate the frame rate of each timeline.
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

### Compute

Inference calls within shared edges are optimized by caching and reusing motion features between image pairs.

### VRAM

When the four points bounding the sample point change, the corresponding memory is freed. The buffer used on top of the pytorch model is approximately `|image + features| * (2 * n + 3)`, where `n` is the number of timelines (input cameras).

### IO

Image decoding and video encoding run in separate processes parallel to inference using pytorch multiprocessing. Data is pipelined through CPU tensor queues.

### RAM

Frames are produced in the order needed for streaming hologram quilts to a video encoder. The input queue is limited to 8 images, which prevents IO from blocking compute without unnecissary back pressure.

## Dependencies

1. Download the "ours_t.pkl" model into `ckpt` per [EMA-VFI]
2. Install dependencies `pip install -r requirements.txt` (modify [pytorch] per platform)

[EMA-VFI]: https://github.com/MCG-NJU/EMA-VFI#sunglassesplay-with-demos
[pytorch]: https://pytorch.org/get-started/locally/

## Usage

```py
python -m bilinear --help
```
