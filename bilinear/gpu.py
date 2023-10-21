import cv2
import PIL.Image
import numpy
import torch

from ema_vfi.padder import InputPadder


class Marshal:
    def __init__(self, proto: str):
        tensor = self._load(proto)
        self.padder = InputPadder(tensor.shape, divisor=32)
        self.width = self.padder.wd
        self.height = self.padder.ht

    @staticmethod
    def _load(file: str) -> torch.Tensor:
        img = cv2.imread(file)
        mat = img.transpose(2, 0, 1)
        tensor = torch.tensor(mat).cuda(non_blocking=True)
        return (tensor / 255.0).unsqueeze(0)

    def load(self, file: str) -> torch.Tensor:
        return self.padder.pad(self._load(file))[0]

    def dump(self, tensor: torch.Tensor) -> PIL.Image.Image:
        tensor = self.padder.unpad(tensor).squeeze(0).cpu()
        mat = tensor.numpy()
        mat = (mat.transpose(1, 2, 0) * 255.0).astype(numpy.uint8)
        img = PIL.Image.fromarray(mat, "RGB")
        return img
