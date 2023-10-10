"""
Bilinear time-then-space interpolation for jittery, sparse camera arrays.

Image names must use the "{angle:int}-{time:float}.{ext}" name format.
Ascending angles must be right-to-left unless --reverse.
Cameras must be uniformly spaced. (TODO: Allow non-linear angles in batch2)
"""
from glob import glob
import math
import os
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image

import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

show_defaults = argparse.ArgumentDefaultsHelpFormatter
pre_wrap = argparse.RawTextHelpFormatter


class Format(show_defaults, pre_wrap):
    pass


parser = argparse.ArgumentParser(description=__doc__, formatter_class=Format)
parser.add_argument("--model", type=str, default="ours_t", help="Model name (in ckpt/)")
parser.add_argument("--name", type=str, default="quilt", help="Quilt video prefix")
parser.add_argument("--input", type=str, default="mesh", help="Input image folder")
parser.add_argument("--ext", type=str, default="png", help="Input image extension")
parser.add_argument("--fps", type=int, default=24, help="Temporal resolution")
parser.add_argument("--dim", type=int, default=48, help="Spatial resolution")
parser.add_argument("--qw", type=int, default=8, help="Quilt width")
parser.add_argument("--reverse", action="store_true", help="Left to right angles")
args = parser.parse_args()
assert args.model in ["ours_t", "ours_small_t"], "Model not exists!"

# Model setting
TTA = True
if args.model == "ours_small_t":
    TTA = False
    cfg.MODEL_CONFIG["LOGNAME"] = "ours_small_t"
    cfg.MODEL_CONFIG["MODEL_ARCH"] = cfg.init_model_config(F=16, depth=[2, 2, 2, 2, 2])
else:
    cfg.MODEL_CONFIG["LOGNAME"] = "ours_t"
    cfg.MODEL_CONFIG["MODEL_ARCH"] = cfg.init_model_config(F=32, depth=[2, 2, 2, 4, 4])

model = Model(-1)
model.load_model()
model.eval()
model.device()

# Sort images into a space-time mesh
names = glob(f"{args.input}/*.{args.ext}")
mesh = [
    (int(parts[0]), float(parts[1][:-4]), name)
    for name in names
    if (parts := os.path.basename(name).split("-"))
]
angles = list(sorted(set(x for x, *_ in mesh), reverse=args.reverse))
mesh = [list(sorted((t, n) for i, t, n in mesh if i == x)) for x in angles]

# Determine the time range covered by all nodes
t0 = max(T[0][0] for T in mesh)
t1 = min(T[-1][0] for T in mesh)

# Calculate the interpolations for the desired frame rate
batch1 = []
for x, T in zip(angles, mesh):
    step, i, t = 0, 0, t0
    (ta, na), (tb, nb) = T[i : i + 2]
    D = []
    while t <= t1:
        while T[i + 1][0] < t:
            if D:
                batch1.append((na, nb, D, x))
                D = []
            i += 1
            (ta, na), (tb, nb) = T[i : i + 2]
        dt = (t - ta) / (tb - ta)
        D.append(dt)
        step += 1
        t = step / args.fps + t0
    if D:
        batch1.append((na, nb, D, x))

batch1_size = sum(len(b[2]) for b in batch1)


def img_to_gpu(img):
    tensor = torch.tensor(img.transpose(2, 0, 1)).cuda()
    tensor = (tensor / 255.0).unsqueeze(0)
    return tensor


padder = InputPadder(img_to_gpu(cv2.imread(names[0])).shape, divisor=32)

# Interpolate the frame rate
sparse = {x: [] for x in angles}
with tqdm(desc="Interpolating time", total=batch1_size) as pbar:
    for img0, img1, D, x in batch1:
        x0 = padder.pad(img_to_gpu(cv2.imread(img0)))[0]
        x1 = padder.pad(img_to_gpu(cv2.imread(img1)))[0]

        with torch.no_grad():
            preds: list[torch.Tensor] = model.multi_inference(x0, x1, time_list=D)

        for pred in preds:
            tensor = padder.unpad(pred).detach().cpu()
            mat = tensor.numpy()
            image = (mat.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            sparse[x].append(image)

        pbar.update(len(D))

# Calculate the interpolations for the desired quilt size
batch2 = []
duration = len(list(sparse.values())[0])
for t in range(duration):
    ca = 0
    D = []
    for step in range(args.dim):
        x = (len(angles) - 1) * step / (args.dim - 1)
        a = int(x)
        if a == x and a > 0:
            a -= 1
        if a != ca:
            batch2.append((sparse[ca][t], sparse[ca + 1][t], D))
            D = []
            ca = a
        dt = x - a
        D.append(dt)
        step += 1
        x = step / args.dim + t0
    if D:
        batch2.append((sparse[ca][t], sparse[ca + 1][t], D))

batch2_size = sum(len(b[2]) for b in batch2)

# Compute the quilt dimensions
W, H = padder.wd, padder.ht
AR = W / H
QW = args.qw
QH = math.ceil(args.dim / QW)
QPW = QW * W
QPH = QH * H

# Setup the video encoder
quilt = Image.new("RGB", size=(QPW, QPH))
name = f"{args.name}_qs{QW}x{QH}a{AR:.3f}.mp4"
fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
writer = cv2.VideoWriter(name, fourcc, args.fps, (QPW, QPH))

# Interpolate the camera angles
x = 0
with tqdm(desc="Interpolating space", total=batch2_size) as pbar:
    for img0, img1, D in batch2:
        x0 = padder.pad(img_to_gpu(img0))[0]
        x1 = padder.pad(img_to_gpu(img1))[0]

        with torch.no_grad():
            preds: list[torch.Tensor] = model.multi_inference(x0, x1, time_list=D)

        for pred in preds:
            tensor = padder.unpad(pred).detach().cpu()
            mat = tensor.numpy()
            image = (mat.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            frame = Image.fromarray(image, "RGB")

            # Paste interpolated frames into the quilt
            px = (QW - 1 - x % QW) * W
            py = (x // QW) * H
            quilt.paste(frame, box=(px, py))

            # Flush completed quilts to disk
            x = (x + 1) % args.dim
            if x == 0:
                writer.write(np.array(quilt))

        pbar.update(len(D))

writer.release()
