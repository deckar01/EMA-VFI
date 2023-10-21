from glob import glob
import os

import cv2
import numpy as np
from tqdm import tqdm

from .cli import args
from .core import BilinearTimeSpace, Vertex, Space, Time
from .gpu import Marshal
from .quilt import Quilt


names = glob(f"{args.input}/*.{args.ext}")
points = [
    Vertex(Space(parts[0]), Time(parts[1][: -len(args.ext) - 1]), name)
    for name in names
    if (parts := os.path.basename(name).split("-"))
]

marshal = Marshal(points[0].f)
mesh = BilinearTimeSpace(args.dim, args.fps, points, marshal)

quilt = Quilt(marshal.width, marshal.height, args.qw, args.dim)
name = f"quilts/{args.name}{quilt.suffix}.mp4"
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
video = cv2.VideoWriter(name, fourcc, args.fps, quilt.size)

try:
    progress = tqdm(desc="Quilting", total=mesh.t_steps, smoothing=0)
    frames = mesh.interpolate()
    for i, frame in enumerate(frames):
        quilt.add(i, frame)
        if (i + 1) % args.dim == 0:
            video.write(np.array(quilt.image))
            progress.update()
finally:
    video.release()
