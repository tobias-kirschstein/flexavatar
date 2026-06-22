from dataclasses import replace

import numpy as np
import torch
from dreifus.camera import PoseType
from dreifus.matrix import Pose
from visage.matting.modnet import MODNetMatter

from flexavatar.config.dataset_config import SampleMetadata, FlexAvatarBatch
from flexavatar.data_adapter.base_data_adapter import BaseDataAdapter


def create_example_batch(data_adapter: BaseDataAdapter, person: str, n_frames: int = 1) -> FlexAvatarBatch:
    # 1. Load input image(s)
    sample_metadata = SampleMetadata(person, None, 0, None)
    timesteps = data_adapter.list_timesteps(None)
    if n_frames == -1:
        n_frames = len(timesteps)

    idx_timesteps = np.linspace(0, len(timesteps) - 1, n_frames, dtype=int)

    images_torch = []
    input_flame2world_poses = []
    input_intrinsics = []
    input_expression_codes = []
    for i_frame in range(n_frames):
        timestep = timesteps[idx_timesteps[i_frame]]
        sample_metadata_timestep = replace(sample_metadata, timestep=timestep)
        image = data_adapter.load_image(sample_metadata_timestep)

        # 2. Mask out background from input image
        image_torch = torch.tensor(image / 255, dtype=torch.float32).permute(2, 0, 1)[None]
        modnet_matter = MODNetMatter()
        with torch.no_grad():
            alpha_maps = modnet_matter.parse(image_torch).cpu()
        image_torch = image_torch * alpha_maps[:, None] + 1 - alpha_maps[:, None]

        # 3. Load input camera pose (camera pose of input image relative to FLAME's head-centric space)
        canonical_flame_to_world, _ = data_adapter.load_head_pose(sample_metadata_timestep)
        input_cam2world_pose, input_intrinsic = data_adapter.load_camera_params(sample_metadata_timestep)

        input_flame2world_pose = Pose(
            canonical_flame_to_world.invert().numpy() @ input_cam2world_pose,
            pose_type=PoseType.CAM_2_WORLD)  # Model takes input camera poses wrt head-centric FLAME space
        input_intrinsic = input_intrinsic.rescale(1 / 512)  # Model takes input intrinsics in canonical form

        # 4. Load expression code of input image (needed for fitting stage)
        input_expression_code = torch.tensor(data_adapter.load_expression_code(sample_metadata_timestep))[None]

        images_torch.append(image_torch)
        input_flame2world_poses.append(input_flame2world_pose)
        input_intrinsics.append(input_intrinsic)
        input_expression_codes.append(input_expression_code)

    images_torch = torch.stack(images_torch, dim=1) # [1, V, 3, H, W]
    input_expression_codes = torch.stack(input_expression_codes, dim=1) # [1, V, 135]

    batch = FlexAvatarBatch(images_torch,
                            None,
                            [input_flame2world_poses],
                            [input_intrinsics],
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            input_expression_codes=input_expression_codes,
                            dataset_ids=torch.ones((1, 1), dtype=torch.long))  # bias sink: 1 = 3D

    return batch