from dataclasses import dataclass
from typing import Dict


@dataclass
class LossWeights:
    l2: float
    lpips: float
    id: float


@dataclass
class InversionSettings:
    image_path: str
    max_inversion_steps: int
    max_tuning_steps: int
    one_w_for_all: bool
    loss_weights: Dict[str, LossWeights]
    optimize_cam: bool
    device: str
