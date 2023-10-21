import math

import PIL.Image


class Quilt:
    image: PIL.Image.Image

    def __init__(self, width: int, height: int, quilt_width: int, dim: int):
        quilt_height = math.ceil(dim / quilt_width)

        self.dim = dim
        self.width = quilt_width
        self.height = quilt_height
        self.image_width = width
        self.image_height = height

        ratio = width / height
        self.suffix = f"_qs{self.width}x{self.height}a{ratio:.3f}"

        self.size = (quilt_width * width, quilt_height * height)
        self.image = PIL.Image.new("RGB", self.size)

    def add(self, index: int, img: PIL.Image.Image) -> None:
        x = index % self.dim
        px = (self.width - 1 - x % self.width) * self.image_width
        py = (x // self.width) * self.image_height
        self.image.paste(img, box=(px, py))
