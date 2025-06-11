from DataInterface import InversionSettings, LossWeights
import os
import imageio
import lpips
from torch import Tensor
import numpy as np
import torch
from tqdm import tqdm
import sys

import cv2
from Generator import CGSGANGenerator

from paths import CGSGAN_MODEL_PATH, CGSGAN_SOURCE_PATH, DEBUG_IMG_PATH

sys.path.append(CGSGAN_SOURCE_PATH)
from preprocess.Preprocess import Preprocessor
from training.gs_generator import GSGenerator  # type: ignore
import legacy as legacy  # type: ignore
from losses.id_loss.id_loss import IDLoss


class PTI:
    def __init__(self, generator, settings):
        self.preprocessor = Preprocessor()
        self.settings = settings
        self.device = settings.device
        self.image_size = 512
        self.generator = generator
        self.images = None
        self.loss_L2 = torch.nn.MSELoss(reduction="sum").to(self.device)
        self.loss_percept = lpips.LPIPS(net="alex").to(self.device)
        self.ID_loss = IDLoss()

    def mask_image(self, image, mask_image):
        bg = np.ones_like(image) * 255
        image = ((mask_image / 255) * image + (1 - mask_image / 255) * bg).astype(
            np.uint8
        )
        image = torch.from_numpy(image).to("cuda:0") / 127.5 - 1
        return image

    def get_random_w(self, num):
        z_samples = np.random.RandomState(123).randn(num, self.generator.G.z_dim)
        w_samples = self.generator.G.mapping(
            torch.from_numpy(z_samples).to(self.device), self.cam[0].repeat((num, 1))
        )
        return w_samples

    def calc_average_w(self):
        w_samples = self.get_random_w(10000)
        w_avg = (
            torch.mean(w_samples, dim=0)
            .unsqueeze(dim=0)
            .clone()
            .detach()
            .requires_grad_(True)
        )
        return w_avg

    def calc_loss(self, gen_image, gt_image, weights):
        loss_l2 = self.loss_L2(gen_image, gt_image)
        return loss_l2
        # loss_lpips = self.loss_percept(gen_image, gt_image)
        # loss_lpips = torch.mean(loss_lpips)
        # loss_id = self.ID_loss(gen_image, gt_image)
        # loss = loss_l2 * weights.l2 + loss_lpips * weights.lpips + loss_id * weights.id
        # return loss

    def pre_tuning(self):
        # self.logger.mode = "tuning"
        optimizer = torch.optim.Adam(params=self.generator.G.parameters(), lr=0.001)
        return optimizer

    def pre_inversion(self):
        avg_w: Tensor = self.calc_average_w()
        if self.settings.one_w_for_all:
            w = avg_w.clone().detach().requires_grad_(True).to(self.device)
        else:
            w = (
                avg_w.repeat((self.num_images, 1, 1))
                .clone()
                .detach()
                .requires_grad_(True)
                .to(self.device)
            )
        trainable = [{"params": w}]
        if self.settings.optimize_cam:
            trainable += [{"params": self.cam}]
        optimizer = torch.optim.Adam(params=trainable, lr=0.001)
        return optimizer, w

    def shape_w(self, w):
        if w.shape[0] == 1:
            return w.repeat(self.num_images, 1, 1)
        return w

    def invert(self):
        optim, w = self.pre_inversion()
        weights = self.settings.loss_weights["inversion"]
        for i in tqdm(range(self.settings.max_inversion_steps)):
            optim.zero_grad()
            gen_w = w
            # generated_images = self.generator.generate(gen_w, self.cam, grad=True)
            generated_images = self.generator.G.synthesis(
                ws=w,
                c=self.cam,
                noise_mode="const",
                gs_params=None,
                random_bg=False,
            )
            generated_images = generated_images["image"].requires_grad_(True)

            # save_debug_image(generated_images[0], name="gen")
            loss = self.calc_loss(generated_images, self.images, weights)
            loss.backward()
            optim.step()
        return self.shape_w(w)

    def tune(self, w_pivot):
        optim = self.pre_tuning()
        weights = self.settings.loss_weights["tuning"]
        for i in tqdm(range(self.settings.max_tuning_steps)):
            generated_images = self.generator.generate(w_pivot, self.cam, grad=True)
            loss = self.calc_loss(generated_images, self.images, weights)
            optim.zero_grad()
            loss.backward()
            optim.step()

    def preprocess(self, images):
        masks, cams, images = self.preprocessor(images)
        cams_tensors = []
        masked_images = []

        for image, mask, c in zip(images, masks, cams):
            if mask.ndim == 2:
                mask = mask[:, :, np.newaxis]  # HW => HWC
            bg = np.ones_like(image) * 255
            image = ((mask / 255) * image + (1 - mask / 255) * bg).astype(np.uint8)
            image = torch.from_numpy(image).to("cuda:0") / 127.5 - 1
            image = image.permute(2, 0, 1)
            # image = image[[2, 1, 0], :, :]
            masked_images.append(image)
            c = torch.tensor([float(s) for s in c]).unsqueeze(dim=0)
            cams_tensors.append(c)
        self.images = (
            torch.stack(masked_images, dim=0)
            .clone()
            .detach()
            .to(self.device)
            .to(torch.float32)
        )
        save_debug_image(self.images[0], name="masked")
        self.cam = (
            torch.cat(cams_tensors, dim=0)
            .clone()
            .to(self.device)
            .to(torch.float32)
            .detach()
        )
        self.num_images = self.images.shape[0]

    def train(self, images):
        self.preprocess(images)
        w_pivot = self.invert()
        self.tune(w_pivot)


def tensor_to_image(tensor, normalize=True):
    if torch.is_tensor(tensor):
        image = tensor.detach().cpu().numpy().squeeze()
    else:
        image = tensor

    if normalize:
        image = 255 * ((image + 1) / 2)
        image = image.clip(0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        image = image.transpose(1, 2, 0)
    elif len(image.shape) == 4:
        image = image.transpose(0, 2, 3, 1)
    return image


def save_debug_image(img, step=None, name="debug", from_tensor=True, save_path=None):
    if save_path == None:
        save_path = DEBUG_IMG_PATH
    if step is None:
        full_name = name
    else:
        index = str(step).zfill(4)
        full_name = f"{name}_{index}"
    if from_tensor:
        image = tensor_to_image(img)
    else:
        image = img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type:ignore
    cv2.imwrite(  # type:ignore
        f"{save_path}/{full_name}.png",
        image,
    )


if __name__ == "__main__":
    settings = InversionSettings(
        image_path="foo",
        max_inversion_steps=1000,
        max_tuning_steps=600,
        one_w_for_all=True,
        loss_weights={
            "inversion": LossWeights(l2=1, lpips=1, id=1),
            "tuning": LossWeights(l2=0.001, lpips=0.001, id=0.001),
        },
        optimize_cam=False,
        device="cuda:0",
    )
    generator = CGSGANGenerator(model_path=CGSGAN_MODEL_PATH, device=settings.device)
    pti = PTI(generator, settings)
    paths = [
        "~/Desktop/cgs_gan/inversion_examples/images_512/69992.jpg",
    ]
    images = [imageio.imread(p) for p in paths]
    save_debug_image(images[0], name="img_orig", from_tensor=False)
    pti.train(images)
