from unittest import TestCase

import numpy as np
import torch
from dreifus.camera import PoseType
from dreifus.matrix import Pose, Intrinsics
from dreifus.vector import Vec3
from elias.util.io import load_json, save_img
from gaussian_splatting.arguments import PipelineParams2
from gaussian_splatting.gaussian_renderer import render_distwar
from gaussian_splatting.scene.cameras import pose_to_rendercam

from flexavatar.data_adapter.in_the_wild_data_adapter import InTheWildDataAdapter
from flexavatar.config.dataset_config import SampleMetadata, MVDatasetConfig, GaussianHeadLRMBatch
from visage.matting.modnet import MODNetMatter
from tqdm import tqdm
from dreifus.trajectory import circle_around_axis
import mediapy

from flexavatar.data_adapter.nersemble_data_adapter import NeRSembleDataAdapter
from flexavatar.model.flexavatar_model import GaussianHeadLRMConfig, GaussianHeadLRM
from flexavatar.model.flexavatar_preprocessor import GaussianHeadLRMPreprocessor


class ModelTest(TestCase):
    def test_model(self):
        model_folder = "D:/Projects/PhD-7_Photoreal_3DMM/code_release/models/SLRM-1522"
        dataset_config = MVDatasetConfig.from_json(load_json(f"{model_folder}/dataset_config.json"))
        model_config = GaussianHeadLRMConfig.from_json(load_json(f"{model_folder}/model_config.json"))
        model_config.use_bfloat16 = False
        device = torch.device('cuda')

        person = "tobi"
        data_adapter = InTheWildDataAdapter(person, expression_code_config=dataset_config.expression_code_config)
        data_adapter2 = InTheWildDataAdapter("gemini_cvpr_hat_14", expression_code_config=dataset_config.expression_code_config)
        data_adapter2 = NeRSembleDataAdapter(240, "EMO-1-shout+laugh", expression_code_config=dataset_config.expression_code_config)
        timesteps = data_adapter2.list_timesteps()
        expression_codes = [torch.tensor(data_adapter2.load_expression_code(SampleMetadata(person, None, timestep, None)), device=device)[None, None]
                            for timestep in timesteps]

        poses = circle_around_axis(len(expression_codes), up=Vec3(0, 1, 0), move=Vec3(0, 0, 1), distance=0.3)
        resolution = 512
        intrinsics = Intrinsics(1500 * resolution / 512, 1500 * resolution / 512, resolution / 2, resolution / 2)

        sample_metadata = SampleMetadata(person, None, 0, None)
        image = data_adapter.load_image(sample_metadata)

        canonical_flame_to_world, _ = data_adapter.load_head_pose(sample_metadata)
        input_cam2world_pose, input_intrinsics = data_adapter.load_camera_params(sample_metadata)

        input_flame2world_pose = Pose(
            canonical_flame_to_world.invert().numpy() @ input_cam2world_pose,
            pose_type=PoseType.CAM_2_WORLD)  # Model takes input camera poses wrt head-centric FLAME space
        input_intrinsics = input_intrinsics.rescale(1 / 512)  # Model takes input intrinsics in canonical form

        image_torch = torch.tensor(image / 255, dtype=torch.float32).permute(2, 0, 1)[None]
        modnet_matter = MODNetMatter()
        with torch.no_grad():
            alpha_maps = modnet_matter.parse(image_torch).cpu()
        image_torch = image_torch * alpha_maps[:, None] + 1 - alpha_maps[:, None]

        expression_code = torch.tensor(data_adapter2.load_expression_code(sample_metadata))[None, None]
        batch = GaussianHeadLRMBatch(image_torch[:, None], None, [[input_flame2world_pose]], [[input_intrinsics]], None, None, None, None,
                                     None,
                                     None,
                                     expression_codes=expression_code,
                                     dataset_ids=torch.ones((1, 1), dtype=torch.long))
        batch = batch.to(device)

        preprocessor = GaussianHeadLRMPreprocessor(dataset_config)
        batch = preprocessor.process(batch)

        model = GaussianHeadLRM(model_config)

        checkpoint = torch.load(f"{model_folder}/checkpoints/ckpt-1050k.pt")
        model.load_state_dict(checkpoint)
        model.to(device)
        with torch.no_grad():
            avatar_code = None

            frames = []
            for ex_code, pose in tqdm(zip(expression_codes, poses)):
                output = model.create_gaussian_models(batch.input_images,
                                                      batch.features,
                                                      batch.input_cam2worlds,
                                                      batch.input_intrinsics,
                                                      expression_codes=ex_code,
                                                      dataset_ids=batch.dataset_ids,
                                                      cached_internal_representations=avatar_code)
                if avatar_code is None:
                    # Cache avatar code for faster rendering of future frames
                    avatar_code = output.internal_representations

                render_cam = pose_to_rendercam(pose, intrinsics, resolution, resolution)
                rendering_output = render_distwar(render_cam, output.gaussian_models[0][0], PipelineParams2(), torch.ones((3,), device=device))
                rendered_image = rendering_output['render'].permute(1, 2, 0).detach().cpu().numpy()

                frames.append(rendered_image)

        mediapy.write_video("D:/Projects/PhD-7_Photoreal_3DMM/code_release/animation.mp4", frames, fps=24)
        save_img(np.clip(rendered_image * 255, 0, 255).astype(np.uint8), "D:/Projects/PhD-7_Photoreal_3DMM/code_release/regression.png")
        print('hi')