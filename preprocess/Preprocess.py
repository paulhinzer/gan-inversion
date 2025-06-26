from PIL import Image
import sys
import dlib
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from root import get_project_path
from insightface import app

sys.path.append(f"{get_project_path()}/preprocess/3DDFA_V2/")
from FaceBoxes import FaceBoxes  # type: ignore
from TDDFA import TDDFA  # type: ignore
from utils_3ddfa.pose import P2sRt  # type: ignore
from utils_3ddfa.pose import matrix2angle  # type: ignore

from preprocess.MODNet.src.models.modnet import MODNet
from preprocess.crop_utils import (
    get_crop_bound,
    crop_image,
    crop_final,
    find_center_bbox,
    eg3dcamparams,
)


class KeypointDetectorInsightface:
    def __init__(self):
        self.app = app.FaceAnalysis(name="buffalo_l")  # enable detection model only
        self.app.prepare(ctx_id=0)

    def __call__(self, images):
        landmarks = []
        for img in images:
            detection = self.app.get(img)
            if len(detection) > 0:
                landmarks.append(detection[0].landmark_2d_106)
            else:
                landmarks.append([])
        return landmarks


class KeypointDetectorDlib:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            f"{get_project_path()}/models/shape_predictor_68_face_landmarks.dat"
        )

    def __call__(self, images):
        landmarks = []
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            if len(rects) == 0:
                landmarks.append(-1)
                continue
            shape = predictor(gray, rects[0])
            landmarks.append([np.array([p.x, p.y]) for p in shape.parts()])
        return landmarks


class FaceCropper:
    def __init__(self):
        cfg = {
            "arch": "mobilenet",
            "widen_factor": 1.0,
            "checkpoint_fp": f"{get_project_path()}/models/mb1_120x120.pth",
            "bfm_fp": f"{get_project_path()}/models/bfm_noneck_v3_slim.pkl",
            "param_mean_std_fp": f"{get_project_path()}/models/param_mean_std_62d_120x120.pkl",
            "size": 120,
            "num_params": 62,
        }

        self.tddfa = TDDFA(gpu_mode=True, **cfg)
        self.face_boxes = FaceBoxes()

    def __call__(self, images, lmx, size=512):
        results_meta = []
        cropped_images = []
        for i, item in enumerate(zip(images, lmx)):
            img_orig, landmarks = item
            quad, quad_c, quad_x, quad_y = get_crop_bound(landmarks)

            bound = np.array(
                [[0, 0], [0, size - 1], [size - 1, size - 1], [size - 1, 0]],
                dtype=np.float32,
            )
            mat = cv2.getAffineTransform(quad[:3], bound[:3])
            img = crop_image(img_orig, mat, size, size)
            h, w = img.shape[:2]

            # Detect faces, get 3DMM params and roi boxes
            boxes = self.face_boxes(img)
            if len(boxes) == 0:
                print(f"No face detected")
                continue

            param_lst, roi_box_lst = self.tddfa(img, boxes)
            box_idx = find_center_bbox(roi_box_lst, w, h)

            param = param_lst[box_idx]
            P = param[:12].reshape(3, -1)  # camera matrix
            s_relative, R, t3d = P2sRt(P)

            # Adjust z-translation in object space
            R_ = param[:12].reshape(3, -1)[:, :3]
            u = self.tddfa.bfm.u.reshape(3, -1, order="F")
            trans_z = np.array([0, 0, 0.5 * u[2].mean()])  # Adjust the object center
            trans = np.matmul(R_, trans_z.reshape(3, 1))
            t3d += trans.reshape(3)

            """ Camera extrinsic estimation for GAN training """
            # Normalize P to fit in the original image (before 3DDFA cropping)
            sx, sy, ex, ey = roi_box_lst[0]
            scale_x = (ex - sx) / self.tddfa.size
            scale_y = (ey - sy) / self.tddfa.size
            t3d[0] = (t3d[0] - 1) * scale_x + sx
            t3d[1] = (self.tddfa.size - t3d[1]) * scale_y + sy
            t3d[0] = (t3d[0] - 0.5 * (w - 1)) / (0.5 * (w - 1))  # Normalize to [-1,1]
            t3d[1] = (t3d[1] - 0.5 * (h - 1)) / (
                0.5 * (h - 1)
            )  # Normalize to [-1,1], y is flipped for image space
            t3d[1] *= -1
            t3d[2] = 0
            # orthogonal camera is agnostic to Z (the model always outputs 66.67)

            s_relative = s_relative * 2000
            scale_x = (ex - sx) / (w - 1)
            scale_y = (ey - sy) / (h - 1)
            s = (scale_x + scale_y) / 2 * s_relative

            quad_c = quad_c + quad_x * t3d[0]
            quad_c = quad_c - quad_y * t3d[1]
            quad_x = quad_x * s
            quad_y = quad_y * s
            c, x, y = quad_c, quad_x, quad_y
            quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(
                np.float32
            )

            # final projection matrix
            s = 1
            t3d = 0 * t3d
            R[:, :3] = R[:, :3] * s
            P = np.concatenate([R, t3d[:, None]], 1)
            P = np.concatenate([P, np.array([[0, 0, 0, 1.0]])], 0)
            results_meta.append(eg3dcamparams(P.flatten()))

            # Save cropped images
            cropped_img = crop_final(img_orig, size=size, quad=quad)
            cropped_images.append(cv2.resize(cropped_img, (size, size)))

        return cropped_images, results_meta


class Masking:
    def __init__(self):
        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.modnet = MODNet(backbone_pretrained=False)
        self.modnet = nn.DataParallel(self.modnet).cuda()
        self.modnet.load_state_dict(
            torch.load(
                f"{get_project_path()}/models/modnet_photographic_portrait_matting.ckpt"
            )
        )
        self.modnet.eval()

    def resize_for_input(self, img: torch.Tensor, ref_size=512):
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

    def unify_channels(self, img: np.ndarray):
        if len(img.shape) == 2:
            img = img[:, :, None]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, 0:3]
        return img

    def __call__(self, images):
        masks = []
        for img in images:

            img_h = img.shape[0]
            img_w = img.shape[1]

            img = self.unify_channels(img)
            img = Image.fromarray(img)
            img = self.im_transform(img)
            img = img[None, ...]

            img_resized = self.resize_for_input(img)
            _, _, pred_mask = self.modnet(img_resized.cuda(), True)
            _, _, im_h, im_w = img.shape
            mask = F.interpolate(pred_mask, size=(im_h, im_w), mode="area")
            mask = mask.detach().cpu().numpy()

            masks.append(
                cv2.resize((mask[0][0] * 255).astype(np.uint8), [img_w, img_h])
            )

        return masks


class Preprocessor:
    def __init__(self):
        self.keypoint_detector = KeypointDetectorInsightface()
        self.face_cropper = FaceCropper()
        self.masking = Masking()

    def filter_valid(self, filtered, filterer):
        new_filtered = []
        new_filterer = []
        for img, kp in zip(filtered, filterer):
            if not isinstance(kp, int) or kp != -1:
                new_filtered.append(img)
                new_filterer.append(kp)

        return new_filtered, new_filterer

    @staticmethod
    def mask_image(image, mask_image):
        if mask_image.ndim == 2:
            mask_image = mask_image[:, :, np.newaxis]
        bg = np.ones_like(image) * 255
        image = ((mask_image / 255) * image + (1 - mask_image / 255) * bg).astype(
            np.uint8
        )
        image = torch.from_numpy(image).to("cuda:0") / 127.5 - 1
        return image

    @staticmethod
    def mask_all_images(images, masks):
        masked_images = []
        for image, mask in zip(images, masks):
            masked_images.append(Preprocessor.mask_image(image, mask))
        return masked_images

    def __call__(self, images, target_size):
        num_images = len(images)

        if num_images == 0:
            raise ValueError("No images given")

        keypoints = self.keypoint_detector(images)
        if len(keypoints) != num_images:
            raise ValueError("No image keypoints detected")

        images, keypoints = self.filter_valid(images, keypoints)
        num_images = len(images)

        cropped_images, cam = self.face_cropper(images, keypoints, target_size)
        cam, images = self.filter_valid(cam, cropped_images)
        masks = self.masking(cropped_images)
        torch.set_grad_enabled(True)
        return masks, cam, cropped_images
