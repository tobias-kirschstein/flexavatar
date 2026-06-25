import sys

from flexavatar.env import REPO_ROOT

sys.path.append(f"{REPO_ROOT}/submodules/matanyone")

from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from visage.matting.modnet import MODNetMatter

from matanyone.utils.device import safe_autocast
from matanyone.utils.inference_utils import gen_dilate, gen_erosion


from matanyone.inference.inference_core import InferenceCore


class MatAnyoneModel(InferenceCore):

    def __init__(self):
        super().__init__("PeiqingYang/MatAnyone")
        self._modnet = MODNetMatter()

    @torch.inference_mode()
    @safe_autocast()
    def process_video_memory(self,
                             frames: List[np.ndarray],
                             n_warmup: int = 10,
                             r_erode: int = 10,
                             r_dilate: int = 10,
                             save_image: bool = False,
                             max_size: int = -1,
                             ):
        r_erode = int(r_erode)
        r_dilate = int(r_dilate)
        n_warmup = int(n_warmup)
        max_size = int(max_size)

        # vframes, fps, length, video_name = read_frame_from_videos(input_path)
        vframes = torch.stack([torch.from_numpy(frame).permute(2, 0, 1) for frame in frames])
        length = len(frames)
        repeated_frames = vframes[0].unsqueeze(0).repeat(n_warmup, 1, 1, 1)
        vframes = torch.cat([repeated_frames, vframes], dim=0).float()
        length += n_warmup

        new_h, new_w = vframes.shape[-2:]
        if max_size > 0:
            h, w = new_h, new_w
            min_side = min(h, w)
            if min_side > max_size:
                new_h = int(h / min_side * max_size)
                new_w = int(w / min_side * max_size)
                vframes = F.interpolate(vframes, size=(new_h, new_w), mode="area")

        with torch.no_grad():
            mask = np.clip((self._modnet.parse(torch.from_numpy(frames[0]) / 255).detach().cpu().numpy() * 255), 0, 255).astype(np.uint8)

        if r_dilate > 0:
            mask = gen_dilate(mask, r_dilate, r_dilate)
        if r_erode > 0:
            mask = gen_erosion(mask, r_erode, r_erode)

        mask = torch.from_numpy(mask).float().to(self.device)
        if max_size > 0:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="nearest"
            )[0, 0]

        bgr = (np.array([120, 255, 155], dtype=np.float32) / 255).reshape((1, 1, 3))
        objects = [1]

        phas = []
        fgrs = []
        for ti in tqdm(range(length)):
            image = vframes[ti]
            image_np = np.array(image.permute(1, 2, 0))
            image = (image / 255.0).float().to(self.device)

            if ti == 0:
                output_prob = self.step(image, mask, objects=objects)
                output_prob = self.step(image, first_frame_pred=True)
            else:
                if ti <= n_warmup:
                    output_prob = self.step(image, first_frame_pred=True)
                else:
                    output_prob = self.step(image)

            mask = self.output_prob_to_mask(output_prob)
            pha = mask.unsqueeze(2).cpu().numpy()
            com_np = image_np / 255.0 * pha + bgr * (1 - pha)

            if ti > (n_warmup - 1):
                pha = (pha * 255).astype(np.uint8)
                phas.append(pha[..., 0])

        phas = np.array(phas)

        return phas