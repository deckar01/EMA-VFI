from bisect import insort
from collections import defaultdict
from dataclasses import dataclass
from typing import Generator

from PIL.Image import Image

from .model import Blender

Space = float
Time = float
ImageName = str


@dataclass(order=True, frozen=True)
class Vertex:
    x: Space
    t: Time
    f: ImageName

    def __repr__(self):
        return f"(x:{self.x:.2f}, t:{self.t:.2f})"


class BilinearTimeSpace:
    def __init__(self, n: int, fps: int, points: list[Vertex]):
        self.n = n
        self.fps = fps
        self.points = points
        self.padder = Blender.padder(points[0].f)

        # Partition the points into timelines
        self.TL: defaultdict[Space, list[Vertex]] = defaultdict(list)
        for v in self.points:
            insort(self.TL[v.x], v)

        # Find the time range covered by all timelines
        self.t0 = max(T[0].t for T in self.TL.values())
        self.t1 = min(T[-1].t for T in self.TL.values())
        self.t_steps = int((self.t1 - self.t0) * self.fps) + 1

        # Calculate the space range
        self.X = list(sorted(self.TL.keys()))
        self.x0 = self.X[0]
        self.x1 = self.X[-1]
        self.x_steps = self.n

    def interpolate(self) -> Generator[Image, None, None]:
        # Cache the model bones for each timeline
        Tm: list[Blender | None] = [None] * len(self.X)

        # Interpolate the frame rate
        i = [1] * len(self.X)
        for t_step in range(self.t_steps):
            t = self.t0 + t_step / self.fps

            # Cache the model bone for the current space walk
            Xm: Blender | None = None

            # Interpolate the camera spacing
            j = 0
            k = 1
            for x_step in range(self.x_steps):
                x = self.x0 + (self.x1 - self.x0) * x_step / (self.x_steps - 1)

                # Advance the space window
                while self.X[k] < x:
                    j += 1
                    k += 1
                    Xm = None

                L = self.X[j]
                R = self.X[k]
                Xr = (x - L) / (R - L)

                # Advance the left time window
                while self.TL[L][i[j]].t < t:
                    i[j] += 1
                    Tm[j] = None
                    Xm = None

                La = self.TL[L][i[j] - 1]
                Lb = self.TL[L][i[j]]
                Lr = (t - La.t) / (Lb.t - La.t)

                # Advance the right time window
                while self.TL[R][i[k]].t < t:
                    i[k] += 1
                    Tm[k] = None
                    Xm = None

                Ra = self.TL[R][i[k] - 1]
                Rb = self.TL[R][i[k]]
                Rr = (t - Ra.t) / (Rb.t - Ra.t)

                # Blend new windows into the cache
                if Tm[j] is None:
                    Tm[j] = Blender.read(La.f, Lb.f, self.padder)

                if Tm[k] is None:
                    Tm[k] = Blender.read(Ra.f, Rb.f, self.padder)

                if Xm is None:
                    Xm = Blender(Tm[j].sample(Lr), Tm[k].sample(Rr))  # type: ignore[union-attr]

                # Interpolate the current space-time point
                yield Blender.dump(Xm.sample(Xr), self.padder)
