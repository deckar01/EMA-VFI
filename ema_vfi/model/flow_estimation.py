import torch
import torch.nn as nn
import torch.nn.functional as F

from .warplayer import warp
from .refine import conv, Unet


class Head(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head, self).__init__()
        self.upsample = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.scale = scale
        self.conv = nn.Sequential(
            conv(in_planes * 2 // (4 * 4) + in_else, c),
            conv(c, c),
            conv(c, 5),
        )

    def forward(self, motion_feature, x, flow):  # /16 /8 /4
        motion_feature = self.upsample(motion_feature)  # /4 /2 /1
        if self.scale != 4:
            x = F.interpolate(
                x, scale_factor=4.0 / self.scale, mode="bilinear", align_corners=False
            )
        if flow is not None:
            if self.scale != 4:
                flow = (
                    F.interpolate(
                        flow,
                        scale_factor=4.0 / self.scale,
                        mode="bilinear",
                        align_corners=False,
                    )
                    * 4.0
                    / self.scale
                )
            x = torch.cat((x, flow), 1)
        x = self.conv(torch.cat([motion_feature, x], 1))
        x = F.interpolate(
            x, scale_factor=self.scale // 4, mode="bilinear", align_corners=False
        )
        flow = x[:, :4] * (self.scale // 4)
        mask = x[:, 4:5]
        return flow, mask


class MultiScaleFlow(nn.Module):
    def __init__(self, backbone, **kargs):
        super(MultiScaleFlow, self).__init__()
        self.flow_num_stage = len(kargs["hidden_dims"])
        self.feature_bone = backbone
        self.block = nn.ModuleList(
            [
                Head(
                    kargs["motion_dims"][-1 - i] * kargs["depths"][-1 - i]
                    + kargs["embed_dims"][-1 - i],
                    kargs["scales"][-1 - i],
                    kargs["hidden_dims"][-1 - i],
                    6 if i == 0 else 17,
                )
                for i in range(self.flow_num_stage)
            ]
        )
        self.unet = Unet(kargs["c"] * 2)

    def warp_features(self, xs, flow):
        y0 = []
        y1 = []
        B = xs[0].size(0) // 2
        for x in xs:
            y0.append(warp(x[:B], flow[:, 0:2]))
            y1.append(warp(x[B:], flow[:, 2:4]))
            flow = (
                F.interpolate(
                    flow,
                    scale_factor=0.5,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=False,
                )
                * 0.5
            )
        return y0, y1

    def calculate_flow(self, img0, img1, timestep, af=None, mf=None):
        B = img0.size(0)
        flow, mask = None, None
        for i in range(self.flow_num_stage):
            t = torch.full(mf[-1 - i][:B].shape, timestep, dtype=torch.float).cuda()
            if flow != None:
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                flow_, mask_ = self.block[i](
                    torch.cat(
                        [
                            t * mf[-1 - i][:B],
                            (1 - t) * mf[-1 - i][B:],
                            af[-1 - i][:B],
                            af[-1 - i][B:],
                        ],
                        1,
                    ),
                    torch.cat((img0, img1, warped_img0, warped_img1, mask), 1),
                    flow,
                )
                flow = flow + flow_
                mask = mask + mask_
            else:
                flow, mask = self.block[i](
                    torch.cat(
                        [
                            t * mf[-1 - i][:B],
                            (1 - t) * mf[-1 - i][B:],
                            af[-1 - i][:B],
                            af[-1 - i][B:],
                        ],
                        1,
                    ),
                    torch.cat((img0, img1), 1),
                    None,
                )

        return flow, mask

    def coraseWarp_and_Refine(self, img0, img1, af, flow, mask):
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        c0, c1 = self.warp_features(af, flow)
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        mask_ = torch.sigmoid(mask)
        merged = warped_img0 * mask_ + warped_img1 * (1 - mask_)
        pred = torch.clamp(merged + res, 0, 1)
        return pred
