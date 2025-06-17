import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from id_loss.models.encoders.model_irse import Backbone
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from root import get_project_path


class IDLoss(torch.nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        self.facenet = Backbone(
            input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se"
        )
        self.facenet.load_state_dict(
            torch.load(f"{get_project_path()}/models/model_ir_se50.pth")
        )
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet = self.facenet.to("cuda")

    def extract_feats(self, x, return_all=False):
        if x.shape[2] > 256:
            x = F.interpolate(x, size=(256, 256), mode="area")
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        if return_all:
            return x_feats
        return x_feats[0]

    def forward(self, synth_image, target_image):
        x_feats = self.extract_feats(synth_image)
        y_feats = self.extract_feats(target_image)
        y_feats = y_feats.detach()
        return 1 - y_feats.dot(x_feats)

    def similarity(self, synth_image, target_image):
        x_feats = self.extract_feats(synth_image)
        y_feats = self.extract_feats(target_image)

        return cosine_similarity(x_feats, y_feats)
