from unittest import TestCase

import mediapy
import numpy as np
import torch
from dreifus.camera import PoseType
from dreifus.matrix import Pose, Intrinsics
from dreifus.trajectory import circle_around_axis
from dreifus.vector import Vec3
from elias.util.io import save_img
from gaussian_splatting.arguments import PipelineParams2
from gaussian_splatting.gaussian_renderer import render_distwar
from gaussian_splatting.scene.cameras import pose_to_rendercam
from tqdm import tqdm
from visage.matting.modnet import MODNetMatter

from flexavatar.config.dataset_config import SampleMetadata, FlexAvatarBatch
from flexavatar.data_adapter.in_the_wild_data_adapter import InTheWildDataAdapter
from flexavatar.data_adapter.nersemble_data_adapter import NeRSembleDataAdapter
from flexavatar.model.flexavatar_preprocessor import FlexAvatarPreprocessor
from flexavatar.model.inversion import FittingManager, FittingConfig
from flexavatar.model_manager.flexavatar_model_manager import FlexAvatarModelManager


class ModelTest(TestCase):
    def test_model(self):
        source_person = "tobi"
        run_fitting = True
        model_name = 'FLEX-1'
        checkpoint = -1
        device = torch.device('cuda')
        output_folder = "D:/Projects/PhD-7_Photoreal_3DMM/code_release"

        model_manager = FlexAvatarModelManager(model_name)
        dataset_config = model_manager.load_dataset_config()

        #----------------------------------------------------------
        # Prepare model input
        #----------------------------------------------------------

        data_adapter_source = InTheWildDataAdapter(source_person, expression_code_config=dataset_config.expression_code_config)

        # 1. Load input image
        sample_metadata = SampleMetadata(source_person, None, 0, None)
        image = data_adapter_source.load_image(sample_metadata)

        # 2. Mask out background from input image
        image_torch = torch.tensor(image / 255, dtype=torch.float32).permute(2, 0, 1)[None]
        modnet_matter = MODNetMatter()
        with torch.no_grad():
            alpha_maps = modnet_matter.parse(image_torch).cpu()
        image_torch = image_torch * alpha_maps[:, None] + 1 - alpha_maps[:, None]

        # 3. Load input camera pose (camera pose of input image relative to FLAME's head-centric space)
        canonical_flame_to_world, _ = data_adapter_source.load_head_pose(sample_metadata)
        input_cam2world_pose, input_intrinsics = data_adapter_source.load_camera_params(sample_metadata)

        input_flame2world_pose = Pose(
            canonical_flame_to_world.invert().numpy() @ input_cam2world_pose,
            pose_type=PoseType.CAM_2_WORLD)  # Model takes input camera poses wrt head-centric FLAME space
        input_intrinsics = input_intrinsics.rescale(1 / 512)  # Model takes input intrinsics in canonical form

        # 4. Load expression code of input image (needed for fitting stage)
        input_recordexpression_code = torch.tensor(data_adapter_source.load_expression_code(sample_metadata))[None, None]
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
                                dataset_ids=torch.ones((1, 1), dtype=torch.long))
        batch = batch.to(device)

        # 5. Compute DinoV2 features for input image
        preprocessor = FlexAvatarPreprocessor(dataset_config)
        batch = preprocessor.process(batch)

        # ----------------------------------------------------------
        # Prepare animation controls
        # ----------------------------------------------------------

        # 1. Load expression codes from driving sequence
        data_adapter_driver = NeRSembleDataAdapter(240, "EMO-1-shout+laugh", expression_code_config=dataset_config.expression_code_config)
        timesteps = data_adapter_driver.list_timesteps()
        expression_codes = [torch.tensor(data_adapter_driver.load_expression_code(SampleMetadata(source_person, None, timestep, None)), device=device)[None, None]
                            for timestep in timesteps]

        # 2. Define camera trajectory for rendering (1 camera pose per expression code)
        poses = circle_around_axis(len(expression_codes), up=Vec3(0, 1, 0), move=Vec3(0, 0, 1), distance=0.3)
        resolution = 512
        intrinsics = Intrinsics(1500 * resolution / 512, 1500 * resolution / 512, resolution / 2, resolution / 2)

        #----------------------------------------------------------
        # Load FlexAvatar model
        #----------------------------------------------------------
        model = model_manager.load_checkpoint(checkpoint)
        model.to(device)

        #----------------------------------------------------------
        # Run fitting
        #----------------------------------------------------------
        if run_fitting:
            fitting_config = FittingConfig()
            fitting_manager = FittingManager(model, fitting_config)
            avatar_code, fitting_history, _ = fitting_manager.run_inversion(batch)
            mediapy.write_video(f"{output_folder}/fitting_history_{source_person}.mp4", fitting_history)
        else:
            avatar_code = None

        #----------------------------------------------------------
        # Create avatar and render images
        #----------------------------------------------------------

        with torch.no_grad():
            frames = []
            for ex_code, pose in tqdm(zip(expression_codes, poses), desc="Animating and Rendering"):
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

        mediapy.write_video(f"{output_folder}/animation.mp4", frames, fps=24)
        save_img(np.clip(rendered_image * 255, 0, 255).astype(np.uint8), f"{output_folder}/regression.png")
        print('hi')