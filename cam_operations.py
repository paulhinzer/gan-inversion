import torch
from paths import CGSGAN_SOURCE_PATH
import sys

sys.path.append(CGSGAN_SOURCE_PATH)
from camera_utils import LookAtPoseSampler  # type: ignore


def get_neutral_camera():
    return get_cam((0.5, 0.5))


def get_random_camera(num=1, stdev=0.2):
    return get_cam((0.5, 0.5), stdev=stdev, num=num)


def get_cam(angles, stdev=0.0, num=1, device="cuda:0", radius=2.7):
    focal_length = 4.2647
    intrinsics = torch.tensor(
        [[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device="cuda:0"
    )
    intrinsics = intrinsics.reshape(-1, 9)
    intrinsics = intrinsics.repeat(num, 1)
    point = [0, 0, 0]
    camera_lookat_point = torch.tensor(point, device=device)
    cam2world_pose = LookAtPoseSampler.sample(
        3.14 * angles[0],
        3.14 * angles[1],
        camera_lookat_point,
        horizontal_stddev=stdev,
        vertical_stddev=stdev,
        batch_size=num,
        radius=radius,
        device=device,
    )
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    return c
