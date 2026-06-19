import face_alignment
import numpy as np
import torch
import torchvision.transforms.functional as TF
from face_alignment.detection.sfd.detect import batch_detect
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from sheap import load_sheap_model
from sheap.fa_landmark_utils import detect_face_and_crop
from sheap.landmark_utils import landmarks_2_face_bounding_box


class SheapModule:

    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._sheap_model = load_sheap_model(model_type="expressive").to(device)
        self._fa_model = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.THREE_D, flip_input=False
        )

    def prepare_sheap_input(self, input_image: np.ndarray) -> torch.Tensor:
        # Convert to torch tensor (C, H, W) with values in [0, 1]
        image = torch.from_numpy(input_image).permute(2, 0, 1).float() / 255.0

        # Detect face and crop
        bbox = detect_face_and_crop(image, self._fa_model, margin=0.9, shift_up=0.5)

        # Apply smoothing using exponential moving average
        # bbox = self._smooth_bbox(bbox)
        x0, y0, x1, y1 = bbox

        cropped = image[:, y0:y1, x0:x1]

        # Resize to 224x224 for SHEAP model
        cropped_resized = TF.resize(cropped, [224, 224], antialias=True)

        return cropped_resized

    def fast_prepare_sheap_input(self, input_image: np.ndarray) -> torch.Tensor:
        h, w, _ = input_image.shape

        frame_torch = torch.from_numpy(input_image).permute(2, 0, 1).cuda()
        bboxlist = batch_detect(self._fa_model.face_detector.face_detector, frame_torch[None], 'cuda')[0]
        detected_faces = self._fa_model.face_detector._filter_bboxes(bboxlist)

        bbox_coords = torch.tensor([
            [detected_faces[0][0], detected_faces[0][1]],
            # [detected_faces[0][0], detected_faces[0][3]],
            # [detected_faces[0][2], detected_faces[0][1]],
            [detected_faces[0][2], detected_faces[0][3]]
        ], dtype=torch.int)[None] / torch.tensor([[[w, h]]])
        bbox = landmarks_2_face_bounding_box(bbox_coords, torch.ones(1, dtype=torch.bool), margin=0.4, shift_up=0, clamp=True)
        x0, y0, x1, y1 = bbox[0].tolist()
        x0, y0, x1, y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)

        cropped = frame_torch[:, y0:y1, x0:x1] / 255
        cropped_resized = TF.resize(cropped, [224, 224], antialias=True)

        return cropped_resized

    def __call__(self, image: np.ndarray):
        sheap_input = self.fast_prepare_sheap_input(image)
        sheap_output = self._sheap_model(sheap_input.cuda())
        return sheap_output

    def to_expression_code(self, sheap_output) -> torch.Tensor:
        sheap_code = torch.cat([
            sheap_output['expr'],
            matrix_to_rotation_6d(axis_angle_to_matrix(sheap_output['eye_l_pose'])),
            matrix_to_rotation_6d(axis_angle_to_matrix(sheap_output['eye_r_pose'])),
            sheap_output['eyelids'],
            matrix_to_rotation_6d(axis_angle_to_matrix(sheap_output['neck_pose'])),
            matrix_to_rotation_6d(axis_angle_to_matrix(sheap_output['jaw_pose'])),
            matrix_to_rotation_6d(axis_angle_to_matrix(sheap_output['torso_pose'])),
            sheap_output['cam_trans'],
        ], axis=1)
        return sheap_code
