from PIL import Image

# from MODNet.src.models.modnet import MODNet
from preprocess.MODNet.src.models.modnet import MODNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import dlib
import numpy as np
import cv2
import yaml
import numpy as np
from preprocess.crop_utils import (
    get_crop_bound,
    crop_image,
    crop_final,
    find_center_bbox,
    eg3dcamparams,
)
import sys
from root import get_project_path


class KeypointDetector:
    def __call__(self, images):
        landmarks = []
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            f"{get_project_path()}/models/shape_predictor_68_face_landmarks.dat"
        )
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            for i, rect in enumerate(rects):
                shape = predictor(gray, rect)
                landmarks.append([np.array([p.x, p.y]) for p in shape.parts()])
        return landmarks


class FaceCropper:
    def __call__(self, images, lmx):
        sys.path.append(f"{get_project_path()}/preprocess/3DDFA_V2/")
        from FaceBoxes import FaceBoxes  # type: ignore
        from TDDFA import TDDFA  # type: ignore
        from utils.pose import P2sRt  # type: ignore
        from utils.pose import matrix2angle  # type: ignore

        cfg = {
            "arch": "mobilenet",
            "widen_factor": 1.0,
            "checkpoint_fp": f"{get_project_path()}/preprocess/3DDFA_V2/weights/mb1_120x120.pth",
            "bfm_fp": f"{get_project_path()}/preprocess/3DDFA_V2/configs/bfm_noneck_v3.pkl",
            "size": 120,
            "num_params": 62,
        }

        # cfg = yaml.load(
        #     open(f"{get_project_path()}/preprocess/3DDFA_V2/configs/mb1_120x120.yml"),
        #     Loader=yaml.SafeLoader,
        # )
        gpu_mode = True
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()
        size = 512
        results_quad = {}
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
            boxes = face_boxes(img)
            if len(boxes) == 0:
                print(f"No face detected")

            param_lst, roi_box_lst = tddfa(img, boxes)
            box_idx = find_center_bbox(roi_box_lst, w, h)

            param = param_lst[box_idx]
            P = param[:12].reshape(3, -1)  # camera matrix
            s_relative, R, t3d = P2sRt(P)
            pose = matrix2angle(R)
            pose = [p * 180 / np.pi for p in pose]

            # Adjust z-translation in object space
            R_ = param[:12].reshape(3, -1)[:, :3]
            u = tddfa.bfm.u.reshape(3, -1, order="F")
            trans_z = np.array([0, 0, 0.5 * u[2].mean()])  # Adjust the object center
            trans = np.matmul(R_, trans_z.reshape(3, 1))
            t3d += trans.reshape(3)

            """ Camera extrinsic estimation for GAN training """
            # Normalize P to fit in the original image (before 3DDFA cropping)
            sx, sy, ex, ey = roi_box_lst[0]
            scale_x = (ex - sx) / tddfa.size
            scale_y = (ey - sy) / tddfa.size
            t3d[0] = (t3d[0] - 1) * scale_x + sx
            t3d[1] = (tddfa.size - t3d[1]) * scale_y + sy
            t3d[0] = (t3d[0] - 0.5 * (w - 1)) / (0.5 * (w - 1))  # Normalize to [-1,1]
            t3d[1] = (t3d[1] - 0.5 * (h - 1)) / (
                0.5 * (h - 1)
            )  # Normalize to [-1,1], y is flipped for image space
            t3d[1] *= -1
            t3d[2] = (
                0  # orthogonal camera is agnostic to Z (the model always outputs 66.67)
            )

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
            # results_quad[img_path] = quad

            # Save cropped images
            cropped_img = crop_final(img_orig, size=size, quad=quad)
            # cropped_img = cropped_img[:, :, ::-1] # BGR TO RGB
            cropped_images.append(cv2.resize(cropped_img, (512, 512)))

        results_new = []
        # for img, P in results_meta.items():
        #     img = os.path.basename(img)
        #     res = [format(r, ".6f") for r in P]
        #     results_new.append((img, res))
        return cropped_images, results_meta


class Masking:
    def __init__(self):
        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def get_model(
        self,
        checkpoint_path=f"{get_project_path()}/models/modnet_photographic_portrait_matting.ckpt",
    ):
        modnet = MODNet(backbone_pretrained=False)
        modnet = nn.DataParallel(modnet).cuda()
        weights = torch.load(checkpoint_path)
        modnet.load_state_dict(weights)
        modnet.eval()
        return modnet

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
        modnet = self.get_model()
        masks = []
        images_resized = []

        for img in images:
            img = self.unify_channels(img)
            img = Image.fromarray(img)
            img = self.im_transform(img)
            img = img[None, ...]

            img_resized = self.resize_for_input(img)
            _, _, pred_mask = modnet(img_resized.cuda(), True)
            _, _, im_h, im_w = img.shape
            mask = F.interpolate(pred_mask, size=(im_h, im_w), mode="area")
            mask = mask.detach().cpu().numpy()

            masks.append((mask[0][0] * 255).astype(np.uint8))
            images_resized.append(img_resized)
        return images_resized, masks


class Preprocessor:
    def __init__(self):
        self.keypoint_detector = KeypointDetector()
        self.face_cropper = FaceCropper()
        self.masking = Masking()

    def __call__(self, images):
        keypoints = self.keypoint_detector(images)
        cropped_images, cam = self.face_cropper(images, keypoints)
        _, masks = self.masking(cropped_images)
        torch.set_grad_enabled(True)
        return masks, cam, cropped_images
