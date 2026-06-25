from argparse import Namespace

import numpy as np
import torch
import torchvision
import tyro
from PIL import Image
from dreifus.image import torch_to_numpy_img, Img
from insightface.app import FaceAnalysis
from torch.nn.functional import l1_loss
from tqdm import tqdm
from visage.evaluator.csim_evaluator import CSIMEvaluator
from visage.evaluator.paired_face_image_evaluator import PairedFaceImageEvaluator, PairedFaceImageMetrics

from flexavatar.config.dataset_config import DatasetType
from flexavatar.constants import VFHQ_TEST_DATASET_NAME
from flexavatar.evaluation.evaluation_manager import EvaluationManager, EvaluationConfig, EvaluationResult


def crop_images_gagavatar(target, ins_kps):
    def expand_square_bbox(bbox, scale=3.0):
        assert bbox.shape == (4,)
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        # Expand the bbox
        bbox_size = int(max(bbox_height, bbox_width) * scale)
        half_bbox_size = min(bbox_size, 512, 512) // 2
        center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
        bbox[0], bbox[1] = center[0] - half_bbox_size, center[1] - half_bbox_size
        bbox[2], bbox[3] = center[0] + half_bbox_size, center[1] + half_bbox_size
        return bbox.float()
    ins_bbox = torch.tensor([ins_kps.min(0)[0][1], ins_kps.min(0)[0][0], ins_kps.max(0)[0][1], ins_kps.max(0)[0][0]])
    ins_bbox = expand_square_bbox(ins_bbox).long()
    ins_bbox_size = ins_bbox[2] - ins_bbox[0]
    target = torchvision.transforms.functional.crop(target.float(), top=ins_bbox[0], left=ins_bbox[1], height=ins_bbox_size, width=ins_bbox_size)
    target = torchvision.transforms.functional.resize(target, (512, 512), antialias=True)
    return target


def main(run_name: str = 'FLEX-1',
         /,
         dataset_type: DatasetType = VFHQ_TEST_DATASET_NAME,
         checkpoint: int = -1,
         crop_vfhq: bool = False,
         calc_apd: bool = False,
         run_fitting: bool = False,
         use_cross_reenactment: bool = False,
         black: bool = False):
    """
    Loads pairs of stored prediction/target images for computing evaluation metrics.
    Any flag that was used for `evaluate.py` should also be used here to ensure to correct images are loaded.

    Parameters
    ----------
    run_name:
        which model to compute metrics for
    dataset_type:
        on which dataset to compute metrics
    checkpoint:
        which checkpoint to compute metrics for
    crop_vfhq:
        Only for VFHQ-Test evaluation. Whether to apply the landmark-based cropping procedure following GAGAvatar's evaluation protocol
    calc_apd:
        Whether to also compute AED and APD
    run_fitting:
        Load images from model predictions that used fitting
    use_cross_reenactment:
        Only for VFHQ-Test evaluation. Whether to load cross reenactment images
    black
        Whether to load images with black background
    """

    device = torch.device('cuda')
    evaluation_manager = EvaluationManager(run_name, dataset_type, EvaluationConfig(crop_vfhq=crop_vfhq,
                                                                                    cross_reenactment=use_cross_reenactment,
                                                                                    black=black,
                                                                                    run_fitting=run_fitting))
    if checkpoint == -1:
        checkpoints = evaluation_manager.list_checkpoints_with_predictions()
        checkpoint = checkpoints[-1]

    sample_metadatas = evaluation_manager.list_sample_metadatas(checkpoint)

    image_evaluator = PairedFaceImageEvaluator(lpips_net_type='squeeze' if crop_vfhq else 'alex')
    if calc_apd:
        from eg3d_preprocessor.preprocess.extract_camera import CameraExtractor
        from eg3d_preprocessor.preprocess.extract_landmark import get_landmark
        camera_extractor = CameraExtractor()

    if crop_vfhq:
        app = FaceAnalysis(allowed_modules=['detection', 'landmark_3d_68'])
        app.prepare(ctx_id=0, det_size=(512, 512))

    predictions = []
    targets = []
    ref_images = []
    akds = []
    aeds = []
    apds = []
    for sample_metadata in tqdm(sample_metadatas, desc='load images'):
        prediction_img = evaluation_manager.load_prediction_image(sample_metadata, checkpoint)
        if use_cross_reenactment:
            # For cross reenactment, we also need reference images of the same person for CSIM calculation
            ref_img = evaluation_manager.load_input_image(sample_metadata.participant_id)
            ref_images.append(ref_img)

        target_img = evaluation_manager.load_target_image(sample_metadata, checkpoint)

        if calc_apd:
            # coeff_3dmm_pred = forward_deep3d(prediction_img)
            # coeff_3dmm_target = forward_deep3d(target_img)

            prediction_img_pil = Image.fromarray(prediction_img)
            target_img_pil = Image.fromarray(target_img)
            lm_pred = get_landmark(prediction_img_pil)
            lm_target = get_landmark(target_img_pil)
            coeff_3dmm_pred = camera_extractor.model_3dmm.get_3dmm([prediction_img_pil], [lm_pred])
            coeff_3dmm_target = camera_extractor.model_3dmm.get_3dmm([target_img_pil], [lm_target])
            aed = l1_loss(coeff_3dmm_pred['exp'], coeff_3dmm_target['exp'])
            apd = l1_loss(coeff_3dmm_pred['angle'], coeff_3dmm_target['angle'])
            aeds.append(aed.item())
            apds.append(apd.item())

        if crop_vfhq:
            faces_target = app.get(target_img)
            kps = faces_target[0]['kps']
            prediction_img_cropped = torch_to_numpy_img(crop_images_gagavatar(torch.tensor(prediction_img).permute(2, 0, 1) / 255, torch.tensor(kps)))
            target_img_cropped = torch_to_numpy_img(crop_images_gagavatar(torch.tensor(target_img).permute(2, 0, 1) / 255, torch.tensor(kps)))

            if not use_cross_reenactment:
                faces_prediction = app.get(prediction_img)
                landmarks_3d_target = faces_target[0]['landmark_3d_68']
                landmarks_3d_prediction = faces_prediction[0]['landmark_3d_68']
                akd = np.linalg.norm(landmarks_3d_target - landmarks_3d_prediction, axis=1).mean()
                akds.append(akd)

            prediction_img = prediction_img_cropped
            target_img = target_img_cropped

        predictions.append(prediction_img)
        targets.append(target_img)

    if use_cross_reenactment:
        predictions_torch = torch.stack([Img.from_numpy(prediction).to_torch().img.float() for prediction in predictions])
        ref_images_torch = torch.stack([Img.from_numpy(ref_img).to_torch().img.float() for ref_img in ref_images])

        csim_evaluator = CSIMEvaluator()
        csim = csim_evaluator(predictions_torch, ref_images_torch)
        face_image_metrics = PairedFaceImageMetrics(-1, -1, -1, csim=csim.item())
    else:
        face_image_metrics = image_evaluator.evaluate(predictions, targets)
        if crop_vfhq:
            face_image_metrics.akd = np.mean(akds).item()
    if calc_apd:
        face_image_metrics.apd = np.mean(apds).item()
        face_image_metrics.aed = np.mean(aeds).item()

    evaluation_result = EvaluationResult(face_image_metrics)
    evaluation_manager.save_evaluation_result(evaluation_result, checkpoint)
    print(evaluation_result)
    print('DONE')

if __name__ == '__main__':
    tyro.cli(main)