from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args
    ):
        self.dataset = dataset
        self.args = args
        # 수정: get flame
        try:
            self.get_flame = args.get_flame
        except:
            self.get_flame = None
    def __getitem__(self, index):
        caminfo = self.dataset[index]
        image = caminfo.image
        R = caminfo.R
        T = caminfo.T
        FovX = caminfo.FovX
        FovY = caminfo.FovY
        time = caminfo.time

        cam = Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                        image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time)
        # 수정: add flame infos
        if self.get_flame:
            cam.flame_pose = caminfo.flame_pose
            cam.flame_expression = caminfo.flame_expression
            cam.flame_shape = caminfo.flame_shape

        return cam

    def __len__(self):
        return len(self.dataset)
