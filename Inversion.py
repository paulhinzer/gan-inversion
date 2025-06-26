import cv2
import os
from root import get_project_path
import lpips
from torch import Tensor
import numpy as np
import torch
from tqdm import tqdm
from preprocess.Preprocess import Preprocessor
from losses.id_loss.id_loss import IDLoss


class Inversion:
    def __init__(self, generator, device):
        self.preprocessor = Preprocessor()
        self.device = device
        self.generator = generator
        self._images = torch.zeros(1)
        self.loss_L2 = torch.nn.MSELoss(reduction="mean").to(self.device)
        self.loss_percept = lpips.LPIPS(net="vgg").to(self.device)
        self.ID_loss = IDLoss()
        self._w = None
        self._optim = {"optimizer": None, "state": "initial"}

    def mask_image(self, image, mask_image):
        bg = np.ones_like(image) * 255
        image = ((mask_image / 255) * image + (1 - mask_image / 255) * bg).astype(
            np.uint8
        )
        image = torch.from_numpy(image).to("cuda:0") / 127.5 - 1
        return image

    def get_random_w(self, num):
        z_samples = np.random.RandomState(123).randn(num, self.generator.z_dim)
        w_samples = self.generator.mapping(
            torch.from_numpy(z_samples).to(self.device), self.cam[0].repeat((num, 1))
        )
        return w_samples

    def calc_average_w(self):
        w_samples = self.get_random_w(100_000)
        w_avg = (
            torch.mean(w_samples, dim=0)
            .unsqueeze(dim=0)
            .clone()
            .detach()
            .requires_grad_(True)
        )
        return w_avg

    def calc_loss(self, gen_image, gt_image, weights):
        losses = {}
        losses["l2"] = self.loss_L2(gen_image, gt_image)
        loss_lpips = self.loss_percept(gen_image, gt_image)
        losses["lpips"] = torch.mean(loss_lpips)
        losses["id"] = self.ID_loss(gen_image, gt_image)
        losses["full"] = (
            losses["l2"] * weights["mse_loss"]
            + losses["lpips"] * weights["lpips_loss"]
            + losses["id"] * weights["id_loss"]
        )
        return losses

    def pre_tuning(self):
        torch.set_grad_enabled(True)
        if self._w is None:
            self.reset_w()
        self._optim = {
            "optimizer": torch.optim.Adam(params=self.generator.parameters(), lr=0.000),
            "state": "tune",
        }

    def pre_inversion(self):
        torch.set_grad_enabled(True)
        if self._w is None:
            self.reset_w()
        trainable = [{"params": self._w}]
        self._optim = {
            "optimizer": torch.optim.Adam(params=trainable, lr=0.000),
            "state": "invert",
        }

    def reset_w(self):
        avg_w = self.calc_average_w()
        self._w = avg_w.clone().detach().requires_grad_(True).to(self.device)

    def shape_w(self, w, num_images):
        if w.shape[0] == 1:
            return w.expand(num_images, 1, 512)
        return w

    def invert(self):
        self.pre_inversion()
        weights = self.settings["loss_weights"]["inversion"]
        for _ in tqdm(range(self.settings["max_inversion_steps"])):
            self.inversion_step(weights)
        return self.shape_w(self._w, self.num_images)

    def step(self, hyperparams) -> dict:
        self.update_optimizer_lr(hyperparams["lr"])
        batch_size = int(hyperparams["batch_size"])
        self.get_optim().zero_grad()

        select_indices = np.arange(self.num_images, dtype=int)
        if batch_size != self.num_images:
            select_indices = np.random.choice(select_indices, batch_size)
        ground_truth = self._images[select_indices]
        cam = self.cam[select_indices]

        gen_w = self.shape_w(self._w, batch_size)
        generated_images = self.generator.synthesis(ws=gen_w, c=cam, random_bg=False)[
            "image"
        ]
        loss = self.calc_loss(generated_images, ground_truth, hyperparams)
        loss["full"].backward()
        self.get_optim().step()
        return loss

    def inversion_step(self, hyperparams):
        if self.get_state() == "initial" or self.get_state() == "tune":
            self.pre_inversion()
        loss = self.step(hyperparams)
        return self.get_current_w_pivot(), loss

    def get_current_w_pivot(self):
        if self._w is None:
            raise AttributeError("w pivot is not yet initialized")
        return self._w.clone().detach()

    def update_optimizer_lr(self, lr):
        for param_group in self.get_optim().param_groups:
            param_group["lr"] = lr

    def get_optim(self) -> torch.optim.Optimizer:
        if self._optim["optimizer"] is None:
            raise AttributeError("Optimizer is not initialized")
        return self._optim["optimizer"]

    def set_state(self, state):
        assert state in ["initial", "invert", "tune"]
        self._optim["state"] = state

    def get_state(self):
        return self._optim["state"]

    def tune(self):
        self.pre_tuning()
        weights = self.settings["loss_weights"]["tuning"]
        for _ in tqdm(range(self.settings["max_tuning_steps"])):
            _, _ = self.tuning_step(weights)

    def tuning_step(self, hyperparams):
        if self.get_state() == "initial":
            self.pre_inversion()
        if self.get_state() == "invert":
            self.pre_tuning()
        if self.get_state() != "tune":
            raise AttributeError("Model is not currently tuning.")
        loss = self.step(hyperparams)
        return self.get_current_w_pivot(), loss

    def preprocess(self, images, target_size):
        masks, cams, images = self.preprocessor(images, target_size=target_size)
        cams_tensors = []
        masked_images = []

        for image, mask, camera in tqdm(zip(images, masks, cams), desc="preprocessing"):
            if mask.ndim == 2:
                mask = mask[:, :, np.newaxis]
            bg = np.ones_like(image) * 255
            image = ((mask / 255) * image + (1 - mask / 255) * bg).astype(np.uint8)
            image = torch.from_numpy(image).to(self.device) / 127.5 - 1
            image = image.permute(2, 0, 1)
            masked_images.append(image.clone())
            c = torch.tensor([float(s) for s in camera]).unsqueeze(dim=0)
            cams_tensors.append(c.clone())
        self._images = (
            torch.stack(masked_images, dim=0)
            .clone()
            .detach()
            .to(self.device)
            .to(torch.float32)
        )
        self.cam = (
            torch.cat(cams_tensors, dim=0)
            .clone()
            .to(self.device)
            .to(torch.float32)
            .detach()
        )
        self.num_images = self._images.shape[0]
        return_images = tensor_to_image(self._images)
        if self.num_images == 1:
            return_images = np.expand_dims(return_images, axis=0)
        return return_images

    def train(self, images):
        self.preprocess(images)
        _ = self.invert()
        self.tune()

    def render_w(self, w=None, cam=None):
        if w is None:
            w = self.get_current_w_pivot()
        if cam is None:
            cam = self.cam[0].unsqueeze(dim=0)

        image = self.generator.synthesis(ws=w, c=cam, random_bg=False)["image"]
        return image


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


def save_image(img, step=None, name="debug", from_tensor=True, save_path=None):
    if save_path == None:
        save_path = f"{get_project_path()}/examples/out"
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
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(  # type:ignore
        f"{save_path}/{full_name}.png",
        image,
    )
