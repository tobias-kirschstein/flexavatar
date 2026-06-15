import torch
from dreifus.camera import PoseType
from dreifus.matrix import Pose
from visage.matting.modnet import MODNetMatter

from flexavatar.config.dataset_config import SampleMetadata, FlexAvatarBatch
from flexavatar.data_adapter.base_data_adapter import BaseDataAdapter


def create_example_batch(data_adapter: BaseDataAdapter, person: str) -> FlexAvatarBatch:
    # 1. Load input image
    sample_metadata = SampleMetadata(person, None, 0, None)
    image = data_adapter.load_image(sample_metadata)

    # 2. Mask out background from input image
    image_torch = torch.tensor(image / 255, dtype=torch.float32).permute(2, 0, 1)[None]
    modnet_matter = MODNetMatter()
    with torch.no_grad():
        alpha_maps = modnet_matter.parse(image_torch).cpu()
    image_torch = image_torch * alpha_maps[:, None] + 1 - alpha_maps[:, None]

    # 3. Load input camera pose (camera pose of input image relative to FLAME's head-centric space)
    canonical_flame_to_world, _ = data_adapter.load_head_pose(sample_metadata)
    input_cam2world_pose, input_intrinsics = data_adapter.load_camera_params(sample_metadata)

    input_flame2world_pose = Pose(
        canonical_flame_to_world.invert().numpy() @ input_cam2world_pose,
        pose_type=PoseType.CAM_2_WORLD)  # Model takes input camera poses wrt head-centric FLAME space
    input_intrinsics = input_intrinsics.rescale(1 / 512)  # Model takes input intrinsics in canonical form

    # 4. Load expression code of input image (needed for fitting stage)
    input_recordexpression_code = torch.tensor(data_adapter.load_expression_code(sample_metadata))[None, None]
    batch = FlexAvatarBatch(image_torch[:, None],
                            None,
                            [[input_flame2world_pose]],
                            [[input_intrinsics]],
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            input_expression_codes=input_recordexpression_code,
                            dataset_ids=torch.ones((1, 1), dtype=torch.long))  # bias sink: 1 = 3D

    return batch