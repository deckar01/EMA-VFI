from functools import partial
import PIL.Image
import numpy
import torch
import torchvision.io

from ema_vfi.model.flow_estimation import MultiScaleFlow
from ema_vfi.model.feature_extractor import MotionFormer
from ema_vfi.padder import InputPadder


F = 32
W = 7
depth = [2, 2, 2, 4, 4]

backbone = MotionFormer(
    embed_dims=[F, 2 * F, 4 * F, 8 * F, 16 * F],
    motion_dims=[0, 0, 0, 8 * F // depth[-2], 16 * F // depth[-1]],
    num_heads=[8 * F // 32, 16 * F // 32],
    mlp_ratios=[4, 4],
    qkv_bias=True,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    depths=depth,
    window_sizes=[W, W],
)

net = MultiScaleFlow(
    backbone,
    embed_dims=[F, 2 * F, 4 * F, 8 * F, 16 * F],
    motion_dims=[0, 0, 0, 8 * F // depth[-2], 16 * F // depth[-1]],
    depths=depth,
    num_heads=[8 * F // 32, 16 * F // 32],
    window_sizes=[W, W],
    scales=[4, 8, 16],
    hidden_dims=[4 * F, 4 * F],
    c=F,
)

param = torch.load("ckpt/ours_t.pkl")
net.load_state_dict(
    {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k
        if "attn_mask" not in k
        if "HW" not in k
    }
)

net.eval()
net.to(torch.device("cuda"), non_blocking=True)
torch.set_grad_enabled(False)


class Blender:
    A: torch.Tensor
    B: torch.Tensor

    @staticmethod
    def padder(proto: str) -> InputPadder:
        return InputPadder(Blender._read(proto).shape, divisor=32)
    
    @staticmethod
    def _read(n: str) -> torch.Tensor:
        tensor = torchvision.io.read_image(n).cuda()
        return (tensor / 255.0).unsqueeze(0)

    @staticmethod
    def read(a: str, b:str, padder: InputPadder) -> "Blender":
        A, = padder.pad(Blender._read(a))
        B, = padder.pad(Blender._read(b))
        return Blender(A, B)

    @staticmethod
    def dump(tensor: torch.Tensor, padder: InputPadder) -> PIL.Image.Image:
        tensor = padder.unpad(tensor).squeeze(0).cpu()
        mat = (tensor.numpy().transpose(1, 2, 0) * 255.0).astype(numpy.uint8)
        return PIL.Image.fromarray(mat[:,:,::-1], "RGB")

    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        self.A = A
        self.B = B
        self.af, self.mf = net.feature_bone(A, B)

    def sample(self, t: float) -> torch.Tensor:
        flow, mask = net.calculate_flow(self.A, self.B, t, self.af, self.mf)
        return net.coraseWarp_and_Refine(self.A, self.B, self.af, flow, mask)
