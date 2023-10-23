import math
from queue import Queue

import PIL.Image
import cv2
import numpy as np
from torch import Tensor
from tqdm import tqdm

from .model import Blender, InputPadder


class Quilt:
    def __init__(self, padder: InputPadder, quilt_width: int, dim: int):
        quilt_height = math.ceil(dim / quilt_width)

        self.dim = dim
        self.width = quilt_width
        self.padder = padder

        self.height = quilt_height
        self.image_width = padder.wd
        self.image_height = padder.ht

        ratio = self.image_width / self.image_height
        self.suffix = f"_qs{self.width}x{self.height}a{ratio:.3f}"

        self.size = (quilt_width * self.image_width, quilt_height * self.image_height)
        self.image = PIL.Image.new("RGB", self.size)

    def add(self, index: int, img: PIL.Image.Image) -> None:
        x = index % self.dim
        px = (self.width - 1 - x % self.width) * self.image_width
        py = (x // self.width) * self.image_height
        self.image.paste(img, box=(px, py))

    def dump(self, ostream: Queue[Tensor], total: int, fps: int, name: str) -> None:
        name = f"quilts/{name}{self.suffix}.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video = cv2.VideoWriter(name, fourcc, fps, self.size)

        try:
            progress = tqdm(desc="Interpolating", total=total, unit="quilt")
            i = 0
            while (tensor := ostream.get()).size(dim=0):
                frame = Blender.dump(tensor, self.padder)
                self.add(i, frame)
                if (i + 1) % self.dim == 0:
                    video.write(np.array(self.image))
                    progress.update()
                i += 1
        finally:
            video.release()
