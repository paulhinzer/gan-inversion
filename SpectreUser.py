from root import get_project_path
import cv2
import torchvision.transforms as transforms
from spectre.src.utils.util import tensor2video, tensor2image
from skimage.transform import estimate_transform, warp
from spectre.config import cfg as spectre_cfg
import numpy as np
from SpectreVive import SPECTRE
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, List, Dict


class SpectreUser:
    def __init__(self, device="cuda:0"):
        spectre_cfg.pretrained_modelpath = (
            f"{get_project_path()}/spectre/pretrained/spectre_model.tar"
        )
        spectre_cfg.model.use_tex = False
        spectre_cfg.model.flame_model_path = (
            f"{get_project_path()}/models/generic_flame_model.pkl"
        )
        self.device = device
        self.spectre: SPECTRE = SPECTRE(spectre_cfg, self.device)
        self.spectre.eval()
        self.frame_idx = 0
        self.overlapping_indices = 0

    def get_encoding_no_crop(
        self, frames: Union[Tensor, List[Tensor]], debug=False
    ) -> List[Dict]:
        original_video_length = frames.shape[0]
        self.frame_idx = list(range(0, original_video_length))

        self.frame_idx.insert(0, self.frame_idx[0])
        self.frame_idx.insert(0, self.frame_idx[0])
        self.frame_idx.append(self.frame_idx[-1])
        self.frame_idx.append(self.frame_idx[-1])
        self.frame_idx = np.array(self.frame_idx)
        L = 20  # chunk size

        # create lists of overlapping indices
        indices = list(range(len(self.frame_idx)))
        self.overlapping_indices = [
            indices[i : i + L] for i in range(0, len(indices), L - 4)
        ]

        if len(self.overlapping_indices[-1]) < 5:
            # if the last chunk has less than 5 frames, pad it with the semilast frame
            self.overlapping_indices[-2] = (
                self.overlapping_indices[-2] + self.overlapping_indices[-1]
            )
            self.overlapping_indices[-2] = np.unique(
                self.overlapping_indices[-2]
            ).tolist()
            self.overlapping_indices = self.overlapping_indices[:-1]

        self.overlapping_indices = np.array(self.overlapping_indices)
        code_dicts = []
        all_images = []
        all_shape_images = []
        for chunk_id, chunk_indices in enumerate(self.overlapping_indices):
            indices_current_chunk = self.frame_idx[chunk_indices]
            images_array = frames[indices_current_chunk, :, :, :]
            # images_list = []
            #
            # """ load each image and crop it around the face if necessary """
            # for index_in_chunk in indices_current_chunk:
            #     frame = frames[index_in_chunk]
            #     lmk = landmarks[index_in_chunk]
            #
            #     tform = self.crop_face(frame,lmk,scale=1.6)
            #     cropped_image = warp(frame, tform.inverse, output_shape=(224, 224))
            #     image = 255 * ((cropped_image + 1) / 2)
            #     image = image.clip(0, 255).astype(np.uint8)
            #     images_list.append(cropped_image.transpose(2,0,1))
            #
            # images_array = torch.from_numpy(np.array(images_list)).type(dtype = torch.float32).to(self.device) #K,224,224,3 <- I feel like that's wrong!
            # images_array = torch.cat(images_list).type(dtype = torch.f# Assuming `frames` is your input tensor with shape [9, 3, 512, 512]
            # Step 1: Scale values from [-1, 1] to [0, 255]
            images_array = (images_array + 1) * 0.5
            # images_array = images_array.clamp(0, 255).byte()  # Ensure values are in the correct range and convert to byte

            # Step 2: Resize images to 224x224
            resize_transform = transforms.Resize((224, 224))
            images_array = torch.stack(
                [resize_transform(frame) for frame in images_array]
            )

            # Step 3: Reorder the dimensions to [9, 224, 224, 3]
            # images_array = images_array.permute(0, 2, 3, 1)
            codedict, initial_deca_exp, initial_deca_jaw = self.spectre.encode(
                images_array, requires_grad=True
            )
            codedict["exp"] = codedict["exp"] + initial_deca_exp
            codedict["pose"][..., 3:] = codedict["pose"][..., 3:] + initial_deca_jaw
            for key in codedict.keys():
                """filter out invalid indices - see explanation at the top of the function"""

                if chunk_id == 0 and chunk_id == len(self.overlapping_indices) - 1:
                    pass
                elif chunk_id == 0:
                    codedict[key] = codedict[key][:-2]
                elif chunk_id == len(self.overlapping_indices) - 1:
                    codedict[key] = codedict[key][2:]
                else:
                    codedict[key] = codedict[key][2:-2]
            if debug:
                opdict, visdict = self.spectre.decode(
                    codedict, rendering=True, vis_lmk=False, return_vis=True
                )
                all_shape_images.append(visdict["shape_images"].detach().cpu())
                all_images.append(codedict["images"].detach().cpu())
            code_dicts.append(codedict)
        code_dicts = self.join_dicts(code_dicts)
        if debug:
            self.get_debug_images(all_shape_images, all_images)
        return code_dicts

    def get_encoding(
        self, frames: Union[Tensor, List[Tensor]], landmarks: Tensor, debug=False
    ) -> List[Dict]:
        original_video_length = len(frames)
        self.frame_idx = list(range(0, original_video_length))

        self.frame_idx.insert(0, self.frame_idx[0])
        self.frame_idx.insert(0, self.frame_idx[0])
        self.frame_idx.append(self.frame_idx[-1])
        self.frame_idx.append(self.frame_idx[-1])
        self.frame_idx = np.array(self.frame_idx)
        L = 10  # chunk size

        # create lists of overlapping indices
        indices = list(range(len(self.frame_idx)))
        self.overlapping_indices = [
            indices[i : i + L] for i in range(0, len(indices), L - 4)
        ]

        if len(self.overlapping_indices[-1]) < 5:
            # if the last chunk has less than 5 frames, pad it with the semilast frame
            self.overlapping_indices[-2] = (
                self.overlapping_indices[-2] + self.overlapping_indices[-1]
            )
            self.overlapping_indices[-2] = np.unique(
                self.overlapping_indices[-2]
            ).tolist()
            self.overlapping_indices = self.overlapping_indices[:-1]

        self.overlapping_indices = np.array(self.overlapping_indices)
        code_dicts = []
        all_images = []
        all_shape_images = []
        for chunk_id, chunk_indices in enumerate(self.overlapping_indices):
            indices_current_chunk = self.frame_idx[chunk_indices]
            images_list = []

            """ load each image and crop it around the face if necessary """
            for index_in_chunk in indices_current_chunk:
                frame = frames[index_in_chunk]
                lmk = landmarks[index_in_chunk]

                tform = self.crop_face(frame, lmk, scale=1.6)
                cropped_image = warp(frame, tform.inverse, output_shape=(224, 224))
                images_list.append(cropped_image.transpose(2, 0, 1))
                # image = 255 * ((cropped_image + 1) / 2)
                # image = image.clip(0, 255).astype(np.uint8)

            images_array = (
                torch.from_numpy(np.array(images_list))
                .type(dtype=torch.float32)
                .to(self.device)
            )  # K,224,224,3 <- I feel like that's wrong!
            # images_array = torch.cat(images_list).type(dtype = torch.float32)
            codedict, initial_deca_exp, initial_deca_jaw = self.spectre.encode(
                images_array, requires_grad=True
            )
            codedict["exp"] = codedict["exp"] + initial_deca_exp
            codedict["pose"][..., 3:] = codedict["pose"][..., 3:] + initial_deca_jaw
            for key in codedict.keys():
                """filter out invalid indices - see explanation at the top of the function"""

                if chunk_id == 0 and chunk_id == len(self.overlapping_indices) - 1:
                    pass
                elif chunk_id == 0:
                    codedict[key] = codedict[key][:-2]
                elif chunk_id == len(self.overlapping_indices) - 1:
                    codedict[key] = codedict[key][2:]
                else:
                    codedict[key] = codedict[key][2:-2]
            if debug:
                opdict, visdict = self.spectre.decode(
                    codedict, rendering=True, vis_lmk=False, return_vis=True
                )
                all_shape_images.append(visdict["shape_images"].detach().cpu())
                all_images.append(codedict["images"].detach().cpu())
            code_dicts.append(codedict)
        code_dicts = self.join_dicts(code_dicts)
        if debug:
            self.get_debug_images(all_shape_images, all_images)
        return code_dicts

    def get_encoding_no_grad(
        self, frames: Union[Tensor, List[Tensor]], landmarks: Tensor, debug=False
    ) -> List[Dict]:
        original_video_length = len(frames)
        self.frame_idx = list(range(0, original_video_length))

        """ SPECTRE uses a temporal convolution of size 5. 
        Thus, in order to predict the parameters for a contiguous video with need to 
        process the video in chunks of overlap 2, dropping values which were computed from the 
        temporal kernel which uses pad 'same'. For the start and end of the video we
        pad using the first and last frame of the video. 
        e.g., consider a video of size 48 frames and we want to predict it in chunks of 20 frames 
        (due to memory limitations). We first pad the video two frames at the start and end using
        the first and last frames correspondingly, making the video 52 frames length.
        
        Then we process independently the following chunks:
        [[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
         [16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35]
         [32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51]]
         
         In the first chunk, after computing the 3DMM params we drop 0,1 and 18,19, since they were computed 
         from the temporal kernel with padding (we followed the same procedure in training and computed loss 
         only from valid outputs of the temporal kernel) In the second chunk, we drop 16,17 and 34,35, and in 
         the last chunk we drop 32,33 and 50,51. As a result we get:
         [2..17], [18..33], [34..49] (end included) which correspond to all frames of the original video 
         (removing the initial padding).     
        """

        # pad
        self.frame_idx.insert(0, self.frame_idx[0])
        self.frame_idx.insert(0, self.frame_idx[0])
        self.frame_idx.append(self.frame_idx[-1])
        self.frame_idx.append(self.frame_idx[-1])
        self.frame_idx = np.array(self.frame_idx)
        L = 20  # chunk size

        # create lists of overlapping indices
        indices = list(range(len(self.frame_idx)))
        self.overlapping_indices = [
            indices[i : i + L] for i in range(0, len(indices), L - 4)
        ]

        if len(self.overlapping_indices[-1]) < 5:
            # if the last chunk has less than 5 frames, pad it with the semilast frame
            self.overlapping_indices[-2] = (
                self.overlapping_indices[-2] + self.overlapping_indices[-1]
            )
            self.overlapping_indices[-2] = np.unique(
                self.overlapping_indices[-2]
            ).tolist()
            self.overlapping_indices = self.overlapping_indices[:-1]

        self.overlapping_indices = np.array(self.overlapping_indices)
        code_dicts = []
        all_images = []
        all_shape_images = []
        with torch.no_grad():
            for chunk_id, chunk_indices in enumerate(self.overlapping_indices):
                indices_current_chunk = self.frame_idx[chunk_indices]
                images_list = []

                """ load each image and crop it around the face if necessary """
                for index_in_chunk in indices_current_chunk:
                    frame = frames[index_in_chunk]
                    lmk = landmarks[index_in_chunk]

                    tform = self.crop_face(frame, lmk, scale=1.6)
                    cropped_image = warp(frame, tform.inverse, output_shape=(224, 224))

                    images_list.append(cropped_image.transpose(2, 0, 1))

                images_array = (
                    torch.from_numpy(np.array(images_list))
                    .type(dtype=torch.float32)
                    .to(self.device)
                )  # K,224,224,3 <- I feel like that's wrong!
                # images_array = torch.cat(images_list).type(dtype = torch.float32)
                codedict, initial_deca_exp, initial_deca_jaw = self.spectre.encode(
                    images_array, requires_grad=True
                )
                codedict["exp"] = codedict["exp"] + initial_deca_exp
                codedict["pose"][..., 3:] = codedict["pose"][..., 3:] + initial_deca_jaw
                for key in codedict.keys():
                    """filter out invalid indices - see explanation at the top of the function"""

                    if chunk_id == 0 and chunk_id == len(self.overlapping_indices) - 1:
                        pass
                    elif chunk_id == 0:
                        codedict[key] = codedict[key][:-2]
                    elif chunk_id == len(self.overlapping_indices) - 1:
                        codedict[key] = codedict[key][2:]
                    else:
                        codedict[key] = codedict[key][2:-2]
                if debug:
                    opdict, visdict = self.spectre.decode(
                        codedict, rendering=True, vis_lmk=False, return_vis=True
                    )
                    all_shape_images.append(visdict["shape_images"].detach().cpu())
                    all_images.append(codedict["images"].detach().cpu())
                code_dicts.append(codedict)
        code_dicts = self.join_dicts(code_dicts)
        if debug:
            self.get_debug_images(all_shape_images, all_images)
        return code_dicts

    def join_dicts(self, dicts: List[Dict]):
        temp_dict = {}
        for d in dicts:
            for key, value in d.items():
                if key in temp_dict:
                    temp_dict[key].append(value)
                else:
                    temp_dict[key] = [value]

        result = {key: torch.cat(value_list) for key, value_list in temp_dict.items()}

        for key, value in result.items():
            result[key] = value[2:-2, :]

        return result

    def crop_face(self, frame, landmarks, scale=1.0):
        image_size = 224
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])

        h, wtarget_model_video_dict, _ = frame.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

        size = int(old_size * scale)

        src_pts = np.array(
            [
                [center[0] - size / 2, center[1] - size / 2],
                [center[0] - size / 2, center[1] + size / 2],
                [center[0] + size / 2, center[1] - size / 2],
            ]
        )
        DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
        tform = estimate_transform("similarity", src_pts, DST_PTS)

        return tform

    def get_landmarks(self, frames: Tensor) -> Tensor:
        landmarks = []
        for frame in frames:
            lm = self.landmarks_processor.get_landmarks(frame)
            landmarks.append(lm)

        return landmarks

    def get_debug_images(
        self, all_shape_images, all_images: dict, name="debug_spectre"
    ) -> None:
        vid_shape = tensor2video(torch.cat(all_shape_images, dim=0))[
            2:-2
        ]  # remove padding
        vid_orig = tensor2video(torch.cat(all_images, dim=0))[2:-2]  # remove padding
        grid_vid = np.concatenate((vid_orig, vid_shape), axis=2)
        for i in range(grid_vid.shape[0]):
            img = grid_vid[i, :, :, :].squeeze()
            save_debug_image(img, step=i, name=name, from_tensor=False)

    def fit_codedict_to_range(self, codedict: dict, ran) -> dict:
        new_dict = {}
        for key, value in codedict.items():
            # new_dict[key] = value[ran.start:ran.stop:ran.step]
            new_dict[key] = value[ran.start : ran.stop : ran.step]
        return new_dict


class SpectreLoss(nn.Module, SpectreUser):
    def __init__(self, device, yaw_focus=False) -> None:
        nn.Module.__init__(self)
        SpectreUser.__init__(self, device)
        self.loss_L2 = torch.nn.MSELoss(reduction="sum").to(device)
        self.yaw_focus = yaw_focus

    def generate_debug_image(
        self,
        i: int,
        input_images: Tensor,
        source_code_dict,
        target_code_dict,
        target_video_dict: dict,
        video_range: range,
    ):
        target_code_dict = self.fit_codedict_to_range(target_code_dict, video_range)
        exp_only = {}
        for key, value in target_code_dict.items():
            exp_only[key] = value.clone()

        exp_only["exp"] = target_code_dict["exp"]
        _, visdict = self.spectre.decode(
            exp_only, rendering=True, vis_lmk=False, return_vis=True, render_orig=True
        )
        idx = 2
        target_shape = tensor2image(visdict["shape_images"][idx])
        target_image = target_video_dict["video_dict"]["frames"][
            video_range.start + idx
        ]
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        source_image = tensor2image((input_images[idx] + 1) / 2)
        exp_only["exp"] = source_code_dict["exp"]
        # exp_only["tex"] = source_code_dict["tex"]
        exp_only["pose"][..., 3:] = source_code_dict["pose"][..., 3:]
        _, visdict = self.spectre.decode(
            exp_only, rendering=True, vis_lmk=False, return_vis=True, render_orig=True
        )
        source_exp_shape = tensor2image(visdict["shape_images"][idx])
        _, visdict = self.spectre.decode(
            source_code_dict,
            rendering=True,
            vis_lmk=False,
            return_vis=True,
            render_orig=True,
        )
        source_shape = tensor2image(visdict["shape_images"][idx])
        grid_img = np.concatenate(
            (target_image, target_shape, source_exp_shape, source_shape, source_image),
            axis=1,
        )

    def get_encoded_tensors(
        self,
        input_images: Tensor,
        target_video_dict: dict,
        video_range: range,
        debug: bool = False,
    ) -> None:
        target_code_dict = target_video_dict["additional_information"]["spectre"]
        target_spectre = torch.cat(
            (
                target_code_dict["exp"][video_range, :],
                target_code_dict["pose"][video_range, 3:],
            ),
            dim=1,
        ).detach()
        source_code_dict = self.get_encoding_no_crop(input_images, debug=False)
        source_spectre = torch.cat(
            (source_code_dict["exp"], source_code_dict["pose"][..., 3:]), dim=1
        )
        if debug:
            return source_code_dict, source_spectre, target_code_dict, target_spectre
        else:
            return source_spectre, target_spectre

    def forward(
        self,
        input_images: Tensor,
        target_video_dict: dict,
        video_range: range,
        debug: bool = True,
        i: int = 0,
    ) -> Tensor:

        source_spectre, target_spectre = self.get_encoded_tensors(
            input_images, target_video_dict, video_range, debug=False
        )
        loss = self.loss_L2(target_spectre, source_spectre)
        if self.yaw_focus:
            loss_only_yaw = (
                self.loss_L2(source_spectre[:, 50:], target_spectre[:, 50:]) * 1e4
            )
            loss += loss_only_yaw
        return loss

    def get_spectre_tensors_only_pictures(
        self, debug: bool, input_images: Tensor, target_images: Tensor
    ) -> None:
        source_code_dict = self.get_encoding_no_crop(input_images, debug=False)
        source_spectre = torch.cat(
            (source_code_dict["exp"], source_code_dict["pose"][..., 3:]), dim=1
        )

        target_code_dict = self.get_encoding_no_crop(target_images, debug=False)
        target_spectre = torch.cat(
            (target_code_dict["exp"], target_code_dict["pose"][..., 3:]), dim=1
        )
        if debug:
            return source_code_dict, source_spectre, target_code_dict, target_spectre
        else:
            return source_spectre, target_spectre

    def generate_debug_images(
        self,
        input_images: Tensor,
        source_code_dict,
        target_code_dict,
        target_images: Tensor,
    ):
        _, source_visdict = self.spectre.decode(
            source_code_dict,
            rendering=True,
            vis_lmk=False,
            return_vis=True,
            render_orig=True,
        )
        _, target_visdict = self.spectre.decode(
            target_code_dict,
            rendering=True,
            vis_lmk=False,
            return_vis=True,
            render_orig=True,
        )
        for i in range(input_images.shape[0]):
            input_image = t2i(input_images[i, ...])
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            target_image = t2i(target_images[i, ...])
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            input_shape = tensor2image(source_visdict["shape_images"][i])
            target_shape = tensor2image(target_visdict["shape_images"][i])
            grid_img = np.concatenate(
                (target_image, target_shape, input_shape, input_image), axis=1
            )

    def get_loss_only_picutres(
        self, input_images: Tensor, target_images: Tensor, debug: bool = True
    ) -> Tensor:
        if debug:
            source_code_dict, source_spectre, target_code_dict, target_spectre = (
                self.get_spectre_tensors_only_pictures(
                    debug, input_images, target_images
                )
            )
        else:
            source_spectre, target_spectre = self.get_spectre_tensors_only_pictures(
                debug, input_images, target_images
            )

        if debug:
            self.generate_debug_images(
                input_images, source_code_dict, target_code_dict, target_images
            )

        loss = self.loss_L2(target_spectre, source_spectre)
        if self.yaw_focus:
            loss_only_yaw = (
                self.loss_L2(source_spectre[:, 50:], target_spectre[:, 50:]) * 1e4
            )
            loss += loss_only_yaw
        return loss
