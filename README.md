# Holocam Bilinear Interpolation

Bilinear time-then-space interpolation for jittery, sparse camera arrays.

## Strategy

1. Model the video frames from a sparse camera array as a 2D mesh.
2. Interpolate the frame rate of each camera angle as a 2D grid.
3. Interpolate the camera angles between syntetic frames.

- (space, time) (uniform:sparse, variable:sparse)
  1. (uniform:sparse, uniform:dense)
  2. (uniform:dense, uniform:dense)

## Dependencies

1. Download a trained interpolation model into `ckpt` per [EMA-VFI]
2. Install dependencies `pip install -r requirements.txt` (modify [pytorch] per platform)

[EMA-VFI]: https://github.com/MCG-NJU/EMA-VFI#sunglassesplay-with-demos
[pytorch]: https://pytorch.org/get-started/locally/

## Usage

```py
python bilinear.py --help
```
