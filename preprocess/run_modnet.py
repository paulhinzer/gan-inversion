import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from MODNet.src.models.modnet import MODNet

im_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def get_model(
    checkpoint_path="MODNet/pretrained/modnet_photographic_portrait_matting.ckpt",
):
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet).cuda()
    weights = torch.load(checkpoint_path)
    modnet.load_state_dict(weights)
    modnet.eval()
    return modnet


def resize_for_input(img: torch.Tensor, ref_size=512):
    _, _, im_h, im_w = img.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    img = F.interpolate(img, size=(im_rh, im_rw), mode="area")
    return img


def unify_channels(img: np.ndarray):
    if len(img.shape) == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, 0:3]
    return img


if __name__ == "__main__":
    input_path = None
    mask_output_path = "."
    os.makedirs(mask_output_path, exist_ok=True)

    modnet = get_model()

    for im_name in tqdm(os.listdir(input_path)):
        mask_name = im_name.split(".")[0] + ".png"
        mask_path = os.path.join(mask_output_path, mask_name)

        # white_bg_name = im_name.split(".")[0] + ".png"
        # white_bg_path = os.path.join(output_path, white_bg_name)

        if os.path.exists(mask_path):
            continue
        if not (im_name.endswith(".png") or im_name.endswith(".jpg")):
            continue

        img = Image.open(os.path.join(input_path, im_name))
        img = np.asarray(img)
        img = unify_channels(img)
        img = Image.fromarray(img)
        img = im_transform(img)
        img = img[None, ...]

        img_resized = resize_for_input(img)
        pred_semantic, pred_detail, pred_mask = modnet(img_resized.cuda(), True)
        im_b, im_c, im_h, im_w = img.shape
        mask = F.interpolate(pred_mask, size=(im_h, im_w), mode="area")
        mask = mask.detach().cpu().numpy()
        img = img.detach().cpu().numpy()

        # save mask
        Image.fromarray(((mask[0][0] * 255).astype(np.uint8)), mode="L").save(mask_path)

        # save image with white background
        # white_bg = mask * img + (1 - mask) * np.ones_like(img)
        # white_bg = (255 * (white_bg + 1) / 2).astype(np.uint8)
        # Image.fromarray(np.transpose(white_bg[0], [1, 2, 0]), mode="RGB").save(white_bg_path)
