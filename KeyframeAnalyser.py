from SpectreUser import SpectreUser
from Inversion import save_image
import numpy as np
from preprocess.Preprocess import Preprocessor
from root import get_project_path
import cv2
from typing import Union, List
import torch
from torch import Tensor


class KeyFrameAnalayser:
    def __init__(self):
        self.frames = None
        self.poses = None
        self.preprocessor = Preprocessor()
        self.spectre = SpectreUser()

    def get_frames_from_video(self, video_path, target_fps=10):
        frames = []
        cap = cv2.VideoCapture(video_path)
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
        foo = self.spectre.get_encoding_no_crop(frames)
        return foo

    def pick_frames(self, poses, flame_params):
        pass

    def __call__(self, video_path):
        self.frames = self.get_frames_from_video(video_path)[:3]
        self.get_poses()
        self.analyse_frames()
        i = self.frames[0].permute(2, 0, 1)
        save_image(i, name="i")
        t = self.frames[-1].permute(2, 0, 1)
        save_image(t, name="t")
        j = self.frames[1].permute(2, 0, 1)
        save_image(j, name="j")
        return self.poses
        # poses = self.get_poses()
        # keyframes = self.pick_frames(poses, flame_params)
        # return keyframes


if __name__ == "__main__":
    k = KeyFrameAnalayser()
    k(f"{get_project_path()}/examples/in/me.mp4")
