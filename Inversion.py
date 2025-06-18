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
        self.settings = {
            "one_w_for_all": True,
            "optimize_cam": False,
            "device": device,
        }
        self.device = device
        self.generator = generator
        self._images = None
        self.loss_L2 = torch.nn.MSELoss(reduction="mean").to(self.device)
        self.loss_percept = lpips.LPIPS(net="alex").to(self.device)
        self.ID_loss = IDLoss()
        self._w = None
        self._optim = {"optimizer": None, "state": None}
        self.pre_inversion_done = False

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
        assert self._w is not None
        self._optim = {
            "optimizer": torch.optim.Adam(
                params=self.generator.G.parameters(), lr=0.001
            ),
            "state": "tune",
        }

    def pre_inversion(self):
        torch.set_grad_enabled(True)
        avg_w: Tensor = self.calc_average_w()
        if self.settings["one_w_for_all"]:
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
        if self.settings["optimize_cam"]:
            trainable += [{"params": self.cam}]
        self._optim = {
            "optimizer": torch.optim.Adam(params=trainable, lr=0.001),
            "state": "invert",
        }
        self._w = w

    def shape_w(self, w):
        if w.shape[0] == 1:
            return w.expand(self.num_images, 1, 512)
        return w

    def invert(self):
        self.pre_inversion()
        weights = self.settings["loss_weights"]["inversion"]
        for _ in tqdm(range(self.settings["max_inversion_steps"])):
            self.inversion_step(weights)
        return self.shape_w(self._w)

    def inversion_step(self, loss_weights):
        lr = loss_weights["lr"]
        if not self.pre_inversion_done:
            self.pre_inversion()
            self.pre_inversion_done = True
        if self.get_state() != "invert":
            raise AttributeError("Model is not currently inverting.")
        self.update_optimizer_lr(lr)
        self.get_optim().zero_grad()
        gen_w = self.shape_w(self._w)
        generated_images = self.generator.generate(gen_w, self.cam, grad=True)
        loss = self.calc_loss(generated_images, self._images, loss_weights)
        loss["full"].backward()
        self.get_optim().step()
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
        assert state in ["invert", "tune", "finished"]
        self._optim["state"] = state

    def get_state(self):
        return self._optim["state"]

    def tune(self):
        self.pre_tuning()
        weights = self.settings["loss_weights"]["tuning"]
        for _ in tqdm(range(self.settings["max_tuning_steps"])):
            _, _ = self.tuning_step(weights)

    def tuning_step(self, loss_weights, lr=0.001):
        self.update_optimizer_lr(lr)
        if self.get_state() != "tune":
            raise AttributeError("Model is not currently tuning.")
        generated_images = self.generator.generate(self._w, self.cam, grad=True)
        loss = self.calc_loss(generated_images, self._images, loss_weights)
        self.get_optim().zero_grad()
        loss["full"].backward()
        self.get_optim().step()
        return self.get_current_w_pivot(), loss

    def preprocess(self, images):
        masks, cams, images = self.preprocessor(images)
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
