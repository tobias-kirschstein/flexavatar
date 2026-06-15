import mediapy
import torch
import tyro
from dreifus.camera import PoseType
from dreifus.matrix import Pose, Intrinsics
from dreifus.trajectory import circle_around_axis
from dreifus.vector import Vec3
from elias.util import ensure_directory_exists_for_file
from gaussian_splatting.arguments import PipelineParams2
from gaussian_splatting.gaussian_renderer import render_distwar
from gaussian_splatting.scene.cameras import pose_to_rendercam
from tqdm import tqdm
from visage.matting.modnet import MODNetMatter

from flexavatar.config.dataset_config import SampleMetadata, FlexAvatarBatch
from flexavatar.data_adapter.example_data import create_example_batch
from flexavatar.data_adapter.in_the_wild_data_adapter import InTheWildDataAdapter
from flexavatar.data_adapter.nersemble_data_adapter import NeRSembleDataAdapter
from flexavatar.env import FLEXAVATAR_RENDERINGS_PATH
from flexavatar.model.flexavatar_preprocessor import FlexAvatarPreprocessor
from flexavatar.model.inversion import FittingManager, FittingConfig
from flexavatar.model_manager.avatar_code_manager import AvatarCodeManager
from flexavatar.model_manager.flexavatar_model_manager import FlexAvatarModelManager


def main(source_person: str = 'marble_sculpture',
         driving_sequence: str = 'EMO-1-shout+laugh',
         run_fitting: bool = True,
         render_360: bool = False,
         load_avatar_code: bool = False,
         use_itw_driver: bool = False):
    """

    Parameters
    ----------
    source_person:
        For which input image an avatar should be created.
        Available avatars are in data/inputs/itw/{$source_person}.jpg
    driving_sequence:
        driving video used to animate the avatar.
        Available driving sequences are in data/pixel3dmm_processing/tracking/nersemble/240
    run_fitting:
        Whether to run the fitting stage of FlexAvatar.
    render_360:
        Whether to render a 360° circular trajectory or a frontal circular trajectory.
    load_avatar_code:
        Whether to load the avatar code for a previously generated avatar from data/avatar_codes/itw
    use_itw_driver:
        If true, load driving expression codes from an in-the-wild tracked video.
        In this case, the `driving_sequence` parameter indicates the video name of the driving video.
    """

    model_name = 'FLEX-1'
    checkpoint = 900
    device = torch.device('cuda')
    output_folder = FLEXAVATAR_RENDERINGS_PATH

    model_manager = FlexAvatarModelManager(model_name)
    dataset_config = model_manager.load_dataset_config()

    # ----------------------------------------------------------
    # Prepare model input
    # ----------------------------------------------------------

    data_adapter_source = InTheWildDataAdapter(source_person, expression_code_config=dataset_config.expression_code_config)
    batch = create_example_batch(data_adapter_source, source_person)
    batch = batch.to(device)

    # 5. Compute DinoV2 features for input image
    preprocessor = FlexAvatarPreprocessor(dataset_config)
    batch = preprocessor.process(batch)

    # ----------------------------------------------------------
    # Prepare animation controls
    # ----------------------------------------------------------

    # 1. Load expression codes from driving sequence
    if use_itw_driver:
        data_adapter_driver = InTheWildDataAdapter(driving_sequence, expression_code_config=dataset_config.expression_code_config)
    else:
        data_adapter_driver = NeRSembleDataAdapter(240, driving_sequence, expression_code_config=dataset_config.expression_code_config)

    timesteps = data_adapter_driver.list_timesteps()
    expression_codes = [
        torch.tensor(data_adapter_driver.load_expression_code(SampleMetadata(None, None, timestep, None)), device=device)[None, None]
        for timestep in timesteps]

    # 2. Define camera trajectory for rendering (1 camera pose per expression code)
    if render_360:
        poses = circle_around_axis(len(expression_codes), axis=Vec3(0, 1, 0), up=Vec3(0, 1, 0), move=Vec3(0, 0, -0.05), look_at=Vec3(0, 0, -0.05),
                                   distance=1)
    else:
        poses = circle_around_axis(len(expression_codes), up=Vec3(0, 1, 0), move=Vec3(0, 0, 1), distance=0.3)

    resolution = 512
    intrinsics = Intrinsics(1500 * resolution / 512, 1500 * resolution / 512, resolution / 2, resolution / 2)

    # ----------------------------------------------------------
    # Load FlexAvatar model
    # ----------------------------------------------------------
    model = model_manager.load_checkpoint(checkpoint)
    model.to(device)

    # ----------------------------------------------------------
    # Run fitting
    # ----------------------------------------------------------
    avatar_code_manager = AvatarCodeManager()
    if load_avatar_code:
        # Load pre-existing avatar code (skip fitting)
        if not avatar_code_manager.has_avatar_code(source_person):
            print(f"No avatar code available for {source_person}")
            exit()
        avatar_code = avatar_code_manager.load_avatar_code(source_person)
        avatar_code = avatar_code.to(device)
    elif run_fitting:
        # Run fitting stage to refine avatar code from encoder
        fitting_config = FittingConfig()
        fitting_manager = FittingManager(model, fitting_config)
        avatar_code, fitting_history, _ = fitting_manager.run_inversion(batch)
        output_path = f"{output_folder}/fitting_history_{source_person}.mp4"
        ensure_directory_exists_for_file(output_path)
        mediapy.write_video(output_path, fitting_history)
    else:
        # No fitting, create raw feed-forward avatar
        avatar_code = None

    # ----------------------------------------------------------
    # Create avatar and render images
    # ----------------------------------------------------------

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

    if not load_avatar_code:
        avatar_code_manager.save_avatar_code(avatar_code, source_person)

    output_name = f"rendering_p{source_person}"
    if driving_sequence != 'EMO-1-shout+laugh':
        output_name += f"_dr{driving_sequence}"
    if render_360:
        output_name += f"_360"

    output_path = f"{output_folder}/{output_name}.mp4"
    ensure_directory_exists_for_file(output_path)
    mediapy.write_video(output_path, frames, fps=24)
    print("DONE")

if __name__ == '__main__':
    tyro.cli(main)