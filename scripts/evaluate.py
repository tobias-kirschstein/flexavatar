from dataclasses import replace
from typing import Optional

import mediapy
import torch
import tyro
from dreifus.image import torch_to_numpy_img
from elias.config import better_replace
from elias.util import ensure_directory_exists_for_file
from tqdm import tqdm
from visage.evaluator.paired_face_image_evaluator import PairedFaceImageEvaluator

from flexavatar.config.dataset_config import DatasetType, SampleMetadata
from flexavatar.constants import VFHQ_TEST_DATASET_NAME
from flexavatar.data_adapter.vfhq_data_adapter import VFHQTestDataAdapter
from flexavatar.dataset.vfhq_test_mv_dataset import VFHQTestMVDataset
from flexavatar.evaluation.evaluation_manager import EvaluationManager, EvaluationConfig, EvaluationResult
from flexavatar.model.flexavatar_preprocessor import FlexAvatarPreprocessor
from flexavatar.model.inversion import FittingManager, FittingConfig
from flexavatar.model_manager.flexavatar_model_manager import FlexAvatarModelManager


def main(model_name: str = 'FLEX-1', /,
         dataset_type: DatasetType = VFHQ_TEST_DATASET_NAME,
         checkpoint: Optional[int] = -1,
         run_fitting: bool = False,
         n_fitting_steps: int = 200,

         black: bool = False,
         use_cross_reenactment: bool = False,
         full_video: bool = False,
         n_frames_vfhq: int = 50):
    """
    Load model and obtain renderings. They will be stored locally for subsequent metric computation via `compute_metrics.py`.

    Parameters
    ----------
    model_name:
        Which model to evaluate
    dataset_type:
        On which dataset to evaluate
    checkpoint:
        Which model checkpoint to evaluate. -1 uses the latest checkpoint
    run_fitting:
        Whether to run the fitting stage for evaluation
    n_fitting_steps:
        How many fitting steps to run for evaluation
    black:
        Whether evaluation metrics should be computed with black or white background
    use_cross_reenactment:
        Only for VFHQ-Test evaluation. If set, uses the expression and pose control from the previous video (in alphabetical order) to animate the portrait
    full_video:
        Only for VFHQ-Test evaluation. If set, predict images for all video frames and store animated portrait videos
    n_frames_vfhq:
        Only for VFHQ-Test evaluation. How many frames per video should be used for evaluation. Lower this for faster evaluation
    """

    evaluator = PairedFaceImageEvaluator(exclude_mssim=True)

    model_manager = FlexAvatarModelManager(model_name)
    evaluation_manager = EvaluationManager(model_name, dataset_type, EvaluationConfig(run_fitting=run_fitting,
                                                                                      n_fitting_steps=n_fitting_steps,
                                                                                      black=black,
                                                                                      cross_reenactment=use_cross_reenactment))
    checkpoint_ids = model_manager.list_checkpoint_ids()
    if checkpoint == -1:
        checkpoint_id = checkpoint_ids[-1]
    else:
        checkpoint_id = checkpoint

    device = torch.device('cuda')

    dataset_config = model_manager.load_dataset_config()
    if dataset_type == VFHQ_TEST_DATASET_NAME:
        dataset_config_val = dataset_config.make_vfhq_test_eval(use_cross_reenactment=use_cross_reenactment, n_target_timesteps=n_frames_vfhq)
    else:
        raise NotImplementedError()

    dataset_config_val = better_replace(dataset_config_val, load_input_expression_codes=run_fitting)

    if black:
        dataset_config_val.bg_color = (0, 0, 0)

    if dataset_type == VFHQ_TEST_DATASET_NAME:
        dataset = VFHQTestMVDataset(dataset_config_val)
    else:
        raise NotImplementedError()
    data_preprocessor = FlexAvatarPreprocessor(dataset_config)

    with torch.no_grad():
        model = model_manager.load_checkpoint(checkpoint)
        model = model.to(device)
        model.eval()

        if run_fitting:
            fitting_manager = FittingManager(model, FittingConfig(steps=n_fitting_steps))

        predictions = []
        target_images = []
        previous_participant_id = None

        sample_idxs = list(range(len(dataset)))

        for idx in tqdm(sample_idxs):
            try:
                sample = dataset[idx]

                if full_video:
                    target_sample_metadata = sample.target_sample_metadatas[0]
                    driver_video_key = target_sample_metadata.participant_id
                    data_adapter = VFHQTestDataAdapter(driver_video_key, expression_code_config=dataset_config.expression_code_config)
                    timesteps = data_adapter.list_timesteps()
                    dataset._config.target_timestep_sampling = 'same_as_input'
                    dataset._config.n_target_timesteps = 1
                    target_samples = [dataset.get_target_sample(SampleMetadata(driver_video_key, target_sample_metadata.sequence_name, t, target_sample_metadata.serial, target_sample_metadata.dataset), sample) for t in timesteps]
                    dataset._config.target_timestep_sampling = 'evenly_spaced'
                    batch = dataset.collate_fn(target_samples)
                    batch.target_images = None
                else:
                    batch = dataset.collate_fn([sample])

                batch = batch.to(device)
                batch = data_preprocessor.process(batch)

                participant_id = sample.input_sample_metadatas[0].participant_id
                evaluation_manager.save_input_image(torch_to_numpy_img(sample.input_images[0]), participant_id)

                if run_fitting:
                    if previous_participant_id is None or participant_id != previous_participant_id:
                        if full_video:
                            latent_avatar_code, fitting_history, _ = fitting_manager.run_inversion(batch[[0]])
                        else:
                            latent_avatar_code, fitting_history, _ = fitting_manager.run_inversion(batch)
                        evaluation_manager.save_fitting_video(fitting_history, participant_id, checkpoint_id)
                        previous_participant_id = participant_id
                else:
                    latent_avatar_code = None

                if full_video:
                    rendered_valid_images = []
                    for i in range(len(batch)):
                        small_batch = batch[[i]]
                        output = model.forward(small_batch, cached_internal_representations=latent_avatar_code)
                        single_rendered_valid_images = output.rendering_output.rendered_images
                        rendered_valid_images.append(torch_to_numpy_img(single_rendered_valid_images[0, 0]))

                    video_path = evaluation_manager.get_prediction_video_path(f"{sample.input_sample_metadatas[0].participant_id}_dr-{sample.target_sample_metadatas[0].participant_id}", checkpoint=0)
                    ensure_directory_exists_for_file(video_path)
                    mediapy.write_video(video_path, rendered_valid_images, fps=24)
                else:
                    output = model.forward(batch, cached_internal_representations=latent_avatar_code)
                    rendered_valid_images = output.rendering_output.rendered_images


                    flat_rendered_valid_images = rendered_valid_images.flatten(0, 1)
                    flat_target_images = batch.target_images.flatten(0, 1)

                    for target_sample_metadata, rendered_img, target_img in zip(batch.target_sample_metadatas[0], flat_rendered_valid_images,
                                                                                flat_target_images):
                        rendered_img = torch_to_numpy_img(rendered_img)
                        target_img = torch_to_numpy_img(target_img)

                        if use_cross_reenactment:
                            # Convention for cross-reenactment: Video is named by source person, not by driver
                            target_sample_metadata = replace(target_sample_metadata, participant_id=participant_id)

                        evaluation_manager.save_prediction_image(rendered_img, target_sample_metadata, checkpoint_id)
                        predictions.append(rendered_img)

                        evaluation_manager.save_target_image(target_img, target_sample_metadata, checkpoint_id)
                        target_images.append(target_img)

            except FileNotFoundError as e:
                print(f"[WARNING] Skipping {dataset._sample_metadatas[idx]} due to: {e}")

        if use_cross_reenactment:
            print("No metrics will be computed at this point since no GT exists for cross-reenactment. Run compute_metrics.py to get cross reenactment metrics.")
            face_image_metrics = None
        else:
            face_image_metrics = evaluator.evaluate(predictions, target_images)

            evaluation_result = EvaluationResult(face_image_metrics)
            evaluation_manager.save_evaluation_result(evaluation_result, checkpoint_id)


    print(f'=== {model_name} ===')
    print(face_image_metrics)


if __name__ == '__main__':
    tyro.cli(main)
