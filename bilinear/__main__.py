from glob import glob
from queue import Queue
import os

from torch import Tensor
from torch.multiprocessing import Manager, Process

from .cli import args
from .core import Compute, Prefetch, Vertex, Space, Time
from .model import Blender
from .quilt import Quilt


names = glob(f"{args.input}/*.{args.ext}")
points = [
    Vertex(Space(parts[0]), Time(parts[1][: -len(args.ext) - 1]), name)
    for name in names
    if (parts := os.path.basename(name).split("-"))
]

manager = Manager()
istream: Queue[Tensor] = manager.Queue(8)
ostream: Queue[Tensor] = manager.Queue()

padder = Blender.padder(points[0].file)

prefetch = Prefetch(args.dim, args.fps, points)
prefetch_process = Process(target=prefetch.fill, args=(istream, padder))

compute = Compute(args.dim, args.fps, points)

quilt = Quilt(padder, args.qw, args.dim)
quilt_args = (ostream, compute.t_steps, args.fps, args.name)
quilt_process = Process(target=quilt.dump, args=quilt_args)

prefetch_process.start()
quilt_process.start()

compute.run(istream, ostream)

prefetch_process.join()
quilt_process.join()
