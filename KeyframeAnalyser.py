import ffmpeg
from SpectreUser import SpectreUser
from sklearn.cluster import KMeans
from Inversion import save_image
import numpy as np
from preprocess.Preprocess import Preprocessor
from root import get_project_path
import cv2
from typing import Union, List
import torch
from torch import Tensor, sub
import os

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


class KeyFrameAnalayser:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.frames = None
        self.poses = None
        self.flame_params = None
        self.preprocessor = Preprocessor()
        self.spectre = SpectreUser()

    def get_frames_from_video(self, video_path, target_fps=10):
        frames = []
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        rotate = self.check_rotation(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return frames

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / target_fps)
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

    def get_poses(self):
        masks, cam, cropped_images = self.preprocessor(self.frames)
        images = self.preprocessor.mask_all_images(cropped_images, masks)
        self.frames = images
        self.poses = cam

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
        for i in key_frame_indices:
            img = self.frames[i].permute(2, 0, 1)
            print(i)

        print(self.poses)
        print(self.flame_params)

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
        frames = torch.stack([self.frames[i] for i in keyframes], dim=0)
        print(keyframes)

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
        frames = torch.stack([self.frames[i] for i in keyframes], dim=0)
        print(keyframes)

    def scale(self, scalee, scaler):
        scale = torch.mean(scaler[scaler > 0]) / torch.mean(scalee[scalee > 0])
        return scalee * scale

    def __call__(self, video_path):
        self.frames = self.get_frames_from_video(video_path, target_fps=10)
        i1 = self.frames[0]
        t = self.frames[-1]
        # j = self.frames[30]
        # k = torch.stack(self.frames[:2], dim=0)
        # l = torch.stack(self.frames[:13], dim=0)
        self.get_poses()
        self.analyse_frames()
        # self.pick_frames(5)
        # self.greedy_calculate_cam_distances(5)
        self.clustering_cam_distances(5)
        return self.poses
        # poses = self.get_poses()
        # keyframes = self.pick_frames(poses, flame_params)
        # return keyframes

    def check_rotation(self, video_path):
        meta_dict = ffmpeg.probe(video_path)
        if "rotate" in meta_dict["streams"][0]["tags"].keys():
            return True
        return False
        # if int(meta_dict["streams"][0]["tags"]["rotate"]) == 90:
        #     rotateCode = cv2.ROTATE_90_CLOCKWISE
        # elif int(meta_dict["streams"][0]["tags"]["rotate"]) == 180:
        #     rotateCode = cv2.ROTATE_180
        # elif int(meta_dict["streams"][0]["tags"]["rotate"]) == 270:
        #     rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
        # return rotateCode

    def correct_rotation(self, frame, rotate):
        if rotate:
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame


if __name__ == "__main__":
    k = KeyFrameAnalayser()
    k(f"{get_project_path()}/examples/in/me_faces.mp4")
