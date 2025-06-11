import pickle
from abc import ABC
import copy
import torch
import sys

from torch.autograd.profiler import record_function

from paths import CGSGAN_SOURCE_PATH

sys.path.append(CGSGAN_SOURCE_PATH)

import dnnlib  # type: ignore
import legacy as legacy  # type: ignore


class Generator(ABC):
    def __init__(self, device):
        self.device = device
        self.G = None

    def generate(self, w, cam, grad=False):
        if grad:
            generated_images = self.G.synthesis(
                ws=w,
                c=cam,
                noise_mode="const",
                gs_params=None,
                random_bg=False,
            )
        else:
            with torch.no_grad():
                generated_images = self.G.synthesis(
                    ws=w,
                    c=cam,
                    noise_mode="const",
                    gs_params=None,
                    random_bg=False,
                )
        generated_images = generated_images["image"]
        return generated_images


class CGSGANGenerator(Generator):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(device)
        self.G = self.initialize_renderer(model_path)

    def initialize_renderer(self, model_path):
        with open(model_path, "rb") as file:
            save_file = pickle.load(file)
        G = save_file["G_ema"].to(self.device)
        G = copy.deepcopy(G).eval().requires_grad_(True).to(self.device)
        return G


class AttachedGSGANGenerator(Generator):

    def __init__(self, model, device="cuda:0"):
        super().__init__(device)
        self.G = model
