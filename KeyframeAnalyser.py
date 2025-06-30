import ffmpeg
from sklearn.cluster import KMeans
import numpy as np
from preprocess.Preprocess import Preprocessor
import cv2
from typing import List
import torch
from torch import Tensor



class KeyFrameAnalayser:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.frames = None
        self.poses = None
        self.flame_params = None
        self.preprocessor = Preprocessor()

    def get_frames_from_video(self, video_path, target_fps=10):
        frames = []
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        rotate = self.check_rotation(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return frames

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(np.round(original_fps / target_fps))
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.correct_rotation(frame, rotate)
                frames.append(frame)

            frame_count += 1
        cap.release()
        return frames

    def check_rotation(self, video_path):
        meta_dict = ffmpeg.probe(video_path)
        rotateCode = None
        if "rotate" in meta_dict["streams"][0]["tags"].keys():
            rotate_str = meta_dict["streams"][0]["tags"]["rotate"]
            if int(rotate_str) == 90:
                rotateCode = cv2.ROTATE_90_CLOCKWISE
            elif int(rotate_str) == 180:
                rotateCode = cv2.ROTATE_180
            elif int(rotate_str) == 270:
                rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
        return rotateCode

    def correct_rotation(self, frame, rotate):
        if rotate is None:
            return frame
        return cv2.rotate(frame, rotate)

    def get_poses(self, frames):
        masks, cam, cropped_images = self.preprocessor(frames, target_size=512)
        frames = self.preprocessor.mask_all_images(cropped_images, masks)
        return frames, cam

    def analyse_frames(self):
        self.frames: List[Tensor] = self.frames
        frames = torch.stack(self.frames, dim=0).permute(0, 3, 1, 2)
        with torch.no_grad():
            self.flame_params = self.spectre.get_encoding_no_crop(frames)

    def pick_frames(self, num: int):
        poses = [torch.from_numpy(arr) for arr in self.poses]
        poses = torch.stack(poses, dim=0).to(self.device)
        poses_dist = torch.cdist(poses, poses)
        yaw = self.flame_params["pose"][..., 3:]
        yaw_dist = torch.cdist(yaw, yaw)
        exp = self.flame_params["exp"]
        exp_dist = torch.cdist(exp, exp)
        yaw_dist = self.scale(yaw_dist, exp_dist)
        poses_dist = self.scale(poses_dist, exp_dist)

        exp_dist = torch.mean(exp_dist, dim=1)
        yaw_dist = torch.mean(yaw_dist, dim=1)
        poses_dist = torch.mean(poses_dist, dim=1)
        frame_dist = 0.5 * exp_dist + 0.5 * yaw_dist - poses_dist * 2
        _, key_frame_indices = torch.topk(frame_dist, num, largest=False)
        key_frame_indices = key_frame_indices.tolist()
        keyframes = torch.stack([self.frames[i] for i in key_frame_indices], dim=0)
        return keyframes

    def greedy_calculate_cam_distances(self, num):
        # initialization
        num_frames = len(self.poses)
        poses = [torch.from_numpy(arr) for arr in self.poses]
        poses = torch.stack(poses, dim=0).to(self.device)
        poses_dist = torch.cdist(poses, poses)
        max_index = torch.argmax(poses_dist)
        row_index, col_index = divmod(max_index.item(), poses_dist.size(1))
        keyframes = [row_index, col_index]
        not_keyframes = set(range(num_frames)).difference(set(keyframes))
        while len(keyframes) < num:
            submatrix = poses_dist[keyframes, :]
            col_indices = list(not_keyframes)
            submatrix = submatrix[:, col_indices]
            submatrix = torch.mean(submatrix, dim=0)
            max_index = torch.argmax(submatrix)
            keyframes.append(col_indices[max_index])
        frames = [self.frames[i] for i in keyframes]
        return frames

    def clustering_cam_distances(self, num):
        feature_matrix = np.vstack(self.poses)
        kmeans = KMeans(n_clusters=num)
        kmeans.fit(feature_matrix)
        labels = kmeans.labels_

        keyframes = []
        for cluster in range(num):
            cluster_indices = np.where(labels == cluster)[0]
            cluster_mean = np.mean(feature_matrix[cluster_indices], axis=0)
            distances = np.linalg.norm(
                feature_matrix[cluster_indices] - cluster_mean, axis=1
            )
            closest_index = cluster_indices[np.argmin(distances)]
            keyframes.append(closest_index)  # Get the original tensor
        # frames = [tensor_to_image(self.frames[i]) for i in keyframes]
        frames = []
        for i in keyframes:
            image = self.frames[i].clone().detach().cpu().numpy()
            image = 255 * ((image + 1) / 2)
            image = image.clip(0, 255).astype(np.uint8)
            frames.append(image)

        return frames

    def scale(self, scalee, scaler):
        scale = torch.mean(scaler[scaler > 0]) / torch.mean(scalee[scalee > 0])
        return scalee * scale

    def filter_blurry(self, frames):
        variances = []
        good_frames = []
        for frame in frames:
            variances.append(cv2.Laplacian(frame, cv2.CV_64F).var())

        for i in range(0, len(frames) - 1, 2):
            if variances[i] > variances[i + 1]:
                good_frames.append(frames[i])
            else:
                good_frames.append(frames[i + 1])
        return good_frames

    def color_correction(self, frames):
        box_size = 50
        reference_white = frames[0]
        H, W, C = reference_white.shape
        reference_white = reference_white[0:box_size, int(W * 0.8) - box_size:int(W * 0.8) + box_size]
        reference_white = reference_white.mean(axis=(0, 1))
        color_scale = 220 / reference_white
        corrected_frames = [np.clip(frame * color_scale[None, None, :], 0, 255).astype(np.uint8) for frame in frames]
        return corrected_frames

    def __call__(self, video_path, num_frames=5):
        frames = self.get_frames_from_video(video_path, target_fps=30)
        frames = self.filter_blurry(frames)
        frames = self.color_correction(frames)
        self.frames, self.poses = self.get_poses(frames)
        key_frames = self.clustering_cam_distances(num_frames)
        return key_frames
