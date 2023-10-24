from bisect import insort
from collections import defaultdict
from dataclasses import dataclass
from queue import Queue

from torch import Tensor

from ema_vfi.model.device import device
from .model import Blender, InputPadder

Space = float
Time = float


@dataclass(order=True, frozen=True)
class Vertex:
    x: Space
    t: Time
    file: str


class BilinearTimeSpace:
    def __init__(self, n: int, fps: int, points: list[Vertex]):
        self.n = n
        self.fps = fps
        self.points = points

        # Partition the points into timelines
        self.timelines: defaultdict[Space, list[Vertex]] = defaultdict(list)
        for v in self.points:
            insort(self.timelines[v.x], v)

        # Find the time range covered by all timelines
        self.t0 = max(T[0].t for T in self.timelines.values())
        self.t1 = min(T[-1].t for T in self.timelines.values())
        self.t_steps = int((self.t1 - self.t0) * self.fps) + 1

        # Calculate the space range
        self.X = list(sorted(self.timelines.keys()))
        self.x0 = self.X[0]
        self.x1 = self.X[-1]
        self.x_steps = self.n

    def walk(self) -> None:
        # Cache the window for each timeline
        time_windows: list[bool] = [False] * len(self.X)

        # Interpolate the frame rate
        i = [1] * len(self.X)
        for t_step in range(self.t_steps):
            t = self.t0 + t_step / self.fps

            # Cache the window for the current space walk
            space_window: bool = False

            # Interpolate the camera spacing
            j = 0
            k = 1
            for x_step in range(self.x_steps):
                x = self.x0 + (self.x1 - self.x0) * x_step / (self.x_steps - 1)

                # Advance the space window
                while self.X[k] < x:
                    j += 1
                    k += 1
                    space_window = False

                ABx = self.X[j]
                CDx = self.X[k]
                EFd = (x - ABx) / (CDx - ABx)

                # Advance the left time window
                while self.timelines[ABx][i[j]].t < t:
                    i[j] += 1
                    time_windows[j] = False
                    space_window = False

                A = self.timelines[ABx][i[j] - 1]
                B = self.timelines[ABx][i[j]]
                ABt = (t - A.t) / (B.t - A.t)

                # Cache the left time window
                if not time_windows[j]:
                    self.step_time(j, A, B)
                    time_windows[j] = True

                # Advance the right time window
                while self.timelines[CDx][i[k]].t < t:
                    i[k] += 1
                    time_windows[k] = False
                    space_window = False

                C = self.timelines[CDx][i[k] - 1]
                D = self.timelines[CDx][i[k]]
                CDt = (t - C.t) / (D.t - C.t)

                # Cache the right time window
                if not time_windows[k]:
                    self.step_time(k, C, D)
                    time_windows[k] = True

                # Cache the space window
                if not space_window:
                    self.step_space(j, ABt, k, CDt)
                    space_window = True

                # Interpolate the space-time point
                self.sample(EFd)

    def step_time(self, i: int, start: Vertex, end: Vertex) -> None:
        pass

    def step_space(self, j: int, ABt: float, k: int, CDt: float) -> None:
        pass

    def sample(self, EFd: float) -> None:
        pass


class Compute(BilinearTimeSpace):
    def run(self, istream: Queue[Tensor], ostream: Queue[Tensor]) -> None:
        self.istream = istream
        self.ostream = ostream

        self.time_windows: dict[int, Blender] = {}
        self.space_windows: dict[int, Blender] = {}

        super().walk()

        ostream.put(Tensor())

    def step_time(self, i: int, start: Vertex, end: Vertex) -> None:
        """Cache a time blend"""
        start_vertex = self.istream.get().to(device)
        end_vertex = self.istream.get().to(device)
        self.time_windows[i] = Blender(start_vertex, end_vertex)

    def step_space(self, j: int, ABt: float, k: int, CDt: float) -> None:
        """Cache a space blend"""
        AB = self.time_windows[j]
        CD = self.time_windows[k]
        self.space_windows[0] = Blender(AB.sample(ABt), CD.sample(CDt))

    def sample(self, EFd: float) -> None:
        """Sample the space blend"""
        EF = self.space_windows[0]
        self.ostream.put(EF.sample(EFd).cpu())


class Prefetch(BilinearTimeSpace):
    def fill(self, istream: Queue[Tensor], padder: InputPadder) -> None:
        self.istream = istream
        self.padder = padder
        super().walk()

    def step_time(self, i: int, start: Vertex, end: Vertex) -> None:
        """Prepare a time window for blending"""
        self.istream.put(Blender.read(start.file, self.padder).cpu())
        self.istream.put(Blender.read(end.file, self.padder).cpu())
