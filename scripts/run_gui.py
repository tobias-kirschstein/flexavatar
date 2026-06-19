#
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual
# property and proprietary rights in and to this software and related documentation.
# Any commercial use, reproduction, disclosure or distribution of this software and
# related documentation without an express license agreement from Toyota Motor Europe NV/SA
# is strictly prohibited.
#

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from shutil import copy2, rmtree
from threading import Thread
from typing import Literal, Optional

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import torch
import tyro
from dreifus.vector.vector_torch import to_homogeneous
from elias.folder import Folder
from elias.util import ensure_directory_exists_for_file, save_img
from elias.util.io import resize_img
from gaussian_splatting.gaussian_renderer import render_gsplat
from gaussian_splatting.scene import GaussianModel
from pixel3dmm.utils.utils_3d import rotation_6d_to_matrix, matrix_to_rotation_6d
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from scipy.spatial.transform import Rotation as R
from visage.matting.modnet import MODNetMatter

from flexavatar.config.dataset_config import SampleMetadata
from flexavatar.data_adapter.example_data import create_example_batch
from flexavatar.data_adapter.in_the_wild_data_adapter import InTheWildDataAdapter
from flexavatar.data_adapter.nersemble_data_adapter import NeRSembleDataAdapter
from flexavatar.env import FLEXAVATAR_INPUTS_PATH, FLEXAVATAR_AVATAR_CODE_PATH
from flexavatar.model.flexavatar_preprocessor import FlexAvatarPreprocessor
from flexavatar.model.inversion import FittingManager, FittingConfig
from flexavatar.model.sheap import SheapModule
from flexavatar.model_manager.avatar_code_manager import AvatarCodeManager
from flexavatar.model_manager.flexavatar_model_manager import FlexAvatarModelManager
from flexavatar.util.codes import interpolate_codes
from flexavatar.viewer.viewer_utils import Mini3DViewerConfig, Mini3DViewer


def transform_gaussian_model(gaussian_model: GaussianModel, rigid_transform: torch.Tensor):
    transformed_xyz = (to_homogeneous(gaussian_model._xyz) @ rigid_transform.float().T)[..., :3]
    gaussian_model._xyz = transformed_xyz
    rotation_matrices = quaternion_to_matrix(gaussian_model._rotation)
    rotation_matrices_unposed = (rigid_transform[:3, :3] @ rotation_matrices.T).T
    quat_unposed = matrix_to_quaternion(rotation_matrices_unposed)
    gaussian_model._rotation = quat_unposed

def run_pixel3dmm(image_path: str):
    image_name = Path(image_path).stem
    pixel3dmm_tracking_path = f"{PHO3DMM_ANALYSES_PATH}/viewer/tracking/{image_name}/tracking_nV1_noPho_uv2000.0_n1000.0/result.mp4"
    if Path(pixel3dmm_tracking_path).exists():
        print(f"[Skipping] {image_name} because Pixel3DMM tracking already exists")
    else:
        try:
            data_adapter = ViewerDataAdapter(image_name)
            viewer_image_path = f"{PHO3DMM_ANALYSES_PATH}/viewer/input_images/{image_name}{Path(image_path).suffix}"
            print(viewer_image_path)
            if viewer_image_path != image_path:
                copy2(image_path, viewer_image_path)

            pixel3dmm_image_folder = f"{PHO3DMM_ANALYSES_PATH}/viewer/processing/input/{image_name}"
            pixel3dmm_image_path = f"{pixel3dmm_image_folder}/{image_name}.png"
            ensure_directory_exists_for_file(pixel3dmm_image_path)
            copy2(image_path, pixel3dmm_image_path)

            main_pixel3dmm(pixel3dmm_image_folder, f"{image_name}/tracking", cleanup=True)
            if Path(pixel3dmm_image_folder).is_dir():
                rmtree(pixel3dmm_image_folder)
        except Exception as e:
            print(f"[ERROR] Skipping {image_name}")
            print(e)

    print("Pixel3DMM tracking DONE!")

@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


@dataclass
class Config(Mini3DViewerConfig):
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    """Pipeline settings for gaussian splatting rendering"""
    cam_convention: Literal["opengl", "opencv"] = "opencv"
    """Camera convention"""
    point_path: Optional[Path] = None
    """Path to the gaussian splatting file"""
    motion_path: Optional[Path] = None
    """Path to the motion file (npz)"""
    sh_degree: int = 3
    """Spherical Harmonics degree"""
    background_color: tuple[float, float, float] = (1., 1., 1.)
    """default GUI background color"""
    save_folder: Path = Path("./viewer_output")
    """default saving folder"""
    fps: int = 25
    """default fps for recording"""
    keyframe_interval: int = 1
    """default keyframe interval"""
    ref_json: Optional[Path] = None
    """ Path to a reference json file. We copy file paths from a reference json into 
    the exported trajectory json file as placeholders so that `render.py` can directly
    load it like a normal sequence. """
    demo_mode: bool = False
    """The UI will be simplified in demo mode."""


class LocalViewer(Mini3DViewer):
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # recording settings
        self.keyframes = []  # list of state dicts of keyframes
        self.all_frames = {}  # state dicts of all frames {key: [num_frames, ...]}
        self.num_record_timeline = 0
        self.playing = False
        self._window_closed = False
        self._need_encoder = False
        self._lock_head = False
        self._live_reenactment = False
        self._is_manual_animation = False

        #print("Initializing SHeaP...")
        # For real-time driving
        self._sheap_module = None
        self._cam_capture = cv2.VideoCapture(0)
        self._cam_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self._cam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self._cam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ret, frame = self._cam_capture.read()
        self._webcam_buffer = np.zeros_like(resize_img(frame, 1/3), dtype=np.float32)
        self._webcam_buffer_large = None
        self._webcam_recording = False

        self._modnet_matter = MODNetMatter()

        print("Initializing 3D Gaussians...")
        self.init_gaussians()

        self.last_time_animate = None
        self.start_animation_thread()

        super().__init__(cfg, 'FlexAvatar - Local Viewer')
        dpg.maximize_viewport()

        dpg.set_exit_callback(self._on_close_window)

    def init_gaussians(self):
        # load gaussians
        self.gaussians = GaussianModel(self.cfg.sh_degree)

        print("Loading FlexAvatar model...")
        checkpoint = -1
        device = torch.device('cuda')
        model_name = 'FLEX-1'
        model_manager = FlexAvatarModelManager(model_name)
        model = model_manager.load_checkpoint(checkpoint)
        model = model.to(device)
        self._model = model

        # self._render_manager = FancyRenderManager(run_name, checkpoint)

        self._avatar_code_manager = AvatarCodeManager()
        self._inversion_manager = FittingManager(model, FittingConfig())

        dataset_config = model_manager.load_dataset_config()
        self._dataset_config = dataset_config

        self._preprocessor = FlexAvatarPreprocessor(dataset_config)

        print("Loading sample input...")
        custom_avatar_name = 'marble_sculpture'
        self._custom_avatar_name = custom_avatar_name

        data_adapter_source = InTheWildDataAdapter(custom_avatar_name, expression_code_config=dataset_config.expression_code_config)
        batch = create_example_batch(data_adapter_source, custom_avatar_name)
        batch = batch.to(device)
        batch =  self._preprocessor.process(batch)
        self._batch = batch

        print("Loading example animation...")
        driving_sequence = 'SEN-10-port_strong_smokey'
        data_adapter_driver = NeRSembleDataAdapter(240, driving_sequence, expression_code_config=dataset_config.expression_code_config)

        timesteps = data_adapter_driver.list_timesteps()
        expression_codes = [data_adapter_driver.load_expression_code(SampleMetadata(None, None, timestep, None)) for timestep in timesteps]

        expression_transition = interpolate_codes([expression_codes[-1], expression_codes[0]], n_frames=12)
        expression_codes.extend(expression_transition)

        self._expression_codes = [torch.tensor(expression_code, dtype=torch.float32, device=device) for expression_code in expression_codes]
        self._last_expression_code = self._expression_codes[0]

        self._manual_expression_code = self._make_neutral_expression_code(device)

        print("Producing sample avatar...")
        if self._avatar_code_manager.has_avatar_code(custom_avatar_name):
            avatar_code = self._avatar_code_manager.load_avatar_code(custom_avatar_name).to(device)
        else:
            print(f'[WARNING] No avatar code found for default GUI avatar: {custom_avatar_name}')
            avatar_code = None

        with torch.no_grad():
            output = model.create_gaussian_models(batch.input_images,
                                                  batch.features,
                                                  batch.input_cam2worlds,
                                                  batch.input_intrinsics,
                                                  expression_codes=batch.input_expression_codes,
                                                  dataset_ids=batch.dataset_ids,
                                                  cached_internal_representations=avatar_code)

        self._avatar_code = output.internal_representations
        self.gaussians = output.gaussian_models[0][0]

    def _make_neutral_expression_code(self, device=None):
        if device is None:
            device = self._manual_expression_code.device
        neutral = torch.zeros(135, dtype=torch.float32, device=device)
        identity_6d = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=device)
        neutral[100:106] = identity_6d   # left eye
        neutral[106:112] = identity_6d   # right eye
        neutral[114:120] = identity_6d   # neck
        neutral[120:126] = identity_6d   # jaw
        neutral[126:132] = identity_6d   # head pose rotation
        return neutral

    def start_animation_thread(self):
        animation_thread = Thread(target=self._animate_avatar)
        self._animation_thread = animation_thread
        self.playing = True
        animation_thread.start()

    def _on_close_window(self):
        print("WINDOW CLOSE")
        self._window_closed = True

    def _animate_avatar(self):
        t = 0
        while not self._window_closed:
            if self.last_time_animate is not None:
                elapsed = time.time() - self.last_time_animate
                fps = 1 / (elapsed + 1e-6)
                self._log_animation_fps = f'{int(fps):<4d}'
            self.last_time_animate = time.time()

            if self._need_encoder:
                try:

                    print(f"Searching for existing avatar code: {self._custom_avatar_name}")
                    if self._avatar_code_manager.has_avatar_code(self._custom_avatar_name):
                        self._need_encoder = False
                        self._avatar_code = self._avatar_code_manager.load_avatar_code(self._custom_avatar_name).cuda()
                    else:
                        print(f"Creating new avatar {self._custom_avatar_name}...")
                        dataset = ViewerDataset(self._dataset_config, self._custom_avatar_name)
                        sample = dataset[0]
                        sample.dataset_ids = torch.tensor([1])
                        with torch.no_grad():
                            alpha_maps = self._modnet_matter.parse(sample.input_images).cpu()
                        sample.input_images = sample.input_images * alpha_maps[:, None] + 1 - alpha_maps[:, None]
                        batch = dataset.collate_fn([sample])
                        batch = batch.to(torch.device('cuda'))
                        batch = self._preprocessor.process(batch)
                        self._batch = batch
                        with torch.no_grad():
                            output = self._model.forward(self._batch, only_internal_representations=True)

                        self._avatar_code = output.gaussian_models_output.internal_representations
                        print("Done creating avatar")
                        self._need_encoder = False

                        print("Optimizing avatar...")
                        def set_inversion_gaussian_model(output_inversion):
                            self.gaussians = output_inversion.gaussian_models_output.gaussian_models[0][0]

                        latent_avatar_code, _, _ = self._inversion_manager.run_inversion(self._batch, set_inversion_gaussian_model)
                        self._avatar_code = latent_avatar_code
                        print("Done optimizing avatar")
                except Exception as e:
                    print(f"[ERROR] Error during creating avatar {self._custom_avatar_name}: {e}")
                    self._need_encoder = False

            # Real-time animation
            if self._live_reenactment:
                expression_code = self._last_expression_code
            elif self._is_manual_animation:
                expression_code = self._manual_expression_code
            else:
                expression_code = self._expression_codes[t]

            animated_batch = replace(self._batch, expression_codes=expression_code[None][None])

            with torch.no_grad():
                output = self._model.forward(animated_batch, cached_internal_representations=self._avatar_code, only_gaussian_models=True)

            gaussians = output.gaussian_models_output.gaussian_models[0][0]
            t = (t + 1) % len(self._expression_codes)

            if not self._lock_head:
                rotation = rotation_6d_to_matrix(expression_code[-9:-3])
                translation = expression_code[-3:]
                flame2world = torch.eye(4).cuda()
                flame2world[:3, :3] = rotation
                flame2world[:3, 3] = translation

                transform_gaussian_model(gaussians, flame2world)

            # TODO: make this assignment thread-safe?
            self.gaussians = gaussians

    def _start_live_reenactment(self):
        if self._sheap_module is None:
            print("Setting up SHeaP for live-reenactment...")
            # Lazy-load SHeaP such that startup time is not impaired
            self._sheap_module = SheapModule()

        thread = Thread(target=self._live_reenactment_worker)
        thread.start()

    def _live_reenactment_worker(self):
        while self._live_reenactment:
            ret, frame = self._cam_capture.read()
            frame = frame[..., [2, 1, 0]]

            # TODO: This is nasty, we force the webcam frame to be square because currently fast_prepare_sheap_input() doesn't work with non-square images
            crop_x = (frame.shape[1] - frame.shape[0]) // 2
            frame = frame[:, crop_x: frame.shape[0] + crop_x]

            try:
                sheap_output = self._sheap_module(frame)
                expression_code = self._sheap_module.to_expression_code(sheap_output)[0]
                expression_code[-3:] = 0  # Set translations to 0
                self._last_expression_code = expression_code
            except Exception as e:
                pass

    def _webcam_worker(self):
        while self._webcam_recording:
            ret, frame = self._cam_capture.read()
            frame = frame[..., [2, 1, 0]]

            self._webcam_buffer[:] = (resize_img(frame, 1/3) / 255).astype(np.float32)
            self._webcam_buffer_large = (frame / 255).astype(np.float32)

    def refresh_stat(self):
        if self.last_time_fresh is not None:
            elapsed = time.time() - self.last_time_fresh
            fps = 1 / (elapsed + 1e-6)
            dpg.set_value("_log_fps", f'{int(fps):<4d}')
        self.last_time_fresh = time.time()

        if self.last_time_animate is not None:
            dpg.set_value("_log_animation_fps", self._log_animation_fps)

    def get_state_dict(self):
        return {
            'rot': self.cam.rot.as_quat(),
            'look_at': np.array(self.cam.look_at),
            'radius': np.array([self.cam.radius]).astype(np.float32),
            'fovy': np.array([self.cam.fovy]).astype(np.float32),
            'interval': self.cfg.fps * self.cfg.keyframe_interval,
        }

    def get_state_dict_record(self):
        record_timestep = dpg.get_value("_slider_record_timestep")
        state_dict = {k: self.all_frames[k][record_timestep] for k in self.all_frames}
        return state_dict

    def apply_state_dict(self, state_dict):
        if 'rot' in state_dict:
            self.cam.rot = R.from_quat(state_dict['rot'])
        if 'look_at' in state_dict:
            self.cam.look_at = state_dict['look_at']
        if 'radius' in state_dict:
            self.cam.radius = state_dict['radius'].item()
        if 'fovy' in state_dict:
            self.cam.fovy = state_dict['fovy'].item()

    def parse_ref_json(self):
        if self.cfg.ref_json is None:
            return {}
        else:
            with open(self.cfg.ref_json, 'r') as f:
                ref_dict = json.load(f)

        tid2paths = {}
        for frame in ref_dict['frames']:
            tid = frame['timestep_index']
            if tid not in tid2paths:
                tid2paths[tid] = frame
        return tid2paths

    def reset_flame_param(self):
        self.flame_param = {
            'expr': torch.zeros(1, self.gaussians.n_expr),
            'rotation': torch.zeros(1, 3),
            'neck': torch.zeros(1, 3),
            'jaw': torch.zeros(1, 3),
            'eyes': torch.zeros(1, 6),
            'translation': torch.zeros(1, 3),
        }

    def define_gui(self):
        super().define_gui()

        # window: rendering options ==================================================================================================
        with dpg.window(label="Render", tag="_render_window", autosize=True, max_size=(480, 9999)):

            with dpg.group(horizontal=True):
                dpg.add_text("FPS:", show=not self.cfg.demo_mode)
                dpg.add_text("0   ", tag="_log_fps", show=not self.cfg.demo_mode)

            with dpg.group(horizontal=True):
                self._log_animation_fps = "0   "
                dpg.add_text("Animation FPS:", show=not self.cfg.demo_mode)
                dpg.add_text(self._log_animation_fps, tag="_log_animation_fps", show=not self.cfg.demo_mode)

            dpg.add_text(f"number of points: {self.gaussians._xyz.shape[0]}")

            with dpg.group(horizontal=True):
                def callback_lock_head(sender, app_data):
                    self.need_update = True
                    self._lock_head = app_data

                dpg.add_checkbox(label="lock head", default_value=self._lock_head, callback=callback_lock_head, tag="_checkbox_lock_head")

                def callback_live_reenactment(sender, app_data):
                    self._live_reenactment = app_data
                    if app_data:
                        self._start_live_reenactment()

                dpg.add_checkbox(label="live reenactment", default_value=self._live_reenactment, callback=callback_live_reenactment, tag="_checkbox_live_reenactment")

                dpg.add_spacer(width=10)

            with dpg.group(horizontal=True):
                def callback_manual_animation(sender, app_data):
                    self._is_manual_animation = app_data
                    dpg.configure_item("_group_expression_sliders", show=app_data)

                dpg.add_checkbox(label="Manual Animation", default_value=self._is_manual_animation, callback=callback_manual_animation, tag="_checkbox_manual_animation")

                def callback_reset_manual_expression(sender, app_data):
                    self._manual_expression_code = self._make_neutral_expression_code()
                    for _dim in range(10):
                        dpg.set_value(f"_slider_expr_dim_{_dim}", 0.0)
                    dpg.set_value("_slider_eyelid_112", 0.0)
                    for _ax in ['X', 'Y', 'Z']:
                        dpg.set_value(f"_slider_eye_100_{_ax}", 0.0)
                        dpg.set_value(f"_slider_neck_{_ax}", 0.0)
                        dpg.set_value(f"_slider_jaw_{_ax}", 0.0)
                        # dpg.set_value(f"_slider_head_trans_{_ax}", 0.0)

                dpg.add_button(label="Reset", callback=callback_reset_manual_expression, tag="_button_reset_manual_expression")

            with dpg.group(tag="_group_expression_sliders", show=self._is_manual_animation):
                dpg.add_text("Expression Code Dimensions")

                def make_expr_callback(dim):
                    def callback(sender, app_data):
                        self._manual_expression_code[dim] = app_data
                    return callback

                for _dim in range(10):
                    dpg.add_slider_float(label=f"Dim {_dim}", tag=f"_slider_expr_dim_{_dim}", width=200,
                                         min_value=-3.0, max_value=3.0, default_value=0.0,
                                         callback=make_expr_callback(_dim))

                dpg.add_separator()

                def callback_eyelids(sender, app_data):
                    self._manual_expression_code[112] = app_data
                    self._manual_expression_code[113] = app_data

                dpg.add_slider_float(label="Eyelids", tag="_slider_eyelid_112", width=180,
                                     min_value=-3.0, max_value=3.0, default_value=0.0,
                                     callback=callback_eyelids)

                dpg.add_separator()

                def callback_eyes(sender, app_data):
                    euler_angles = [dpg.get_value(f"_slider_eye_100_{ax}") for ax in ['X', 'Y', 'Z']]
                    rot_matrix = torch.tensor(
                        R.from_euler('xyz', euler_angles, degrees=True).as_matrix(),
                        dtype=torch.float32, device=self._manual_expression_code.device
                    )
                    rot6d = matrix_to_rotation_6d(rot_matrix)
                    self._manual_expression_code[100:106] = rot6d
                    self._manual_expression_code[106:112] = rot6d

                def callback_neck(sender, app_data):
                    euler_angles = [dpg.get_value(f"_slider_neck_{ax}") for ax in ['X', 'Y', 'Z']]
                    rot_matrix = torch.tensor(
                        R.from_euler('xyz', euler_angles, degrees=True).as_matrix(),
                        dtype=torch.float32, device=self._manual_expression_code.device
                    )
                    rot6d = matrix_to_rotation_6d(rot_matrix)
                    self._manual_expression_code[114:120] = rot6d
                    self._manual_expression_code[126:132] = rot6d

                def callback_jaw(sender, app_data):
                    euler_angles = [dpg.get_value(f"_slider_jaw_{ax}") for ax in ['X', 'Y', 'Z']]
                    rot_matrix = torch.tensor(
                        R.from_euler('xyz', euler_angles, degrees=True).as_matrix(),
                        dtype=torch.float32, device=self._manual_expression_code.device
                    )
                    self._manual_expression_code[120:126] = matrix_to_rotation_6d(rot_matrix)

                def make_head_trans_callback(dim):
                    def callback(sender, app_data):
                        self._manual_expression_code[dim] = app_data
                    return callback

                _slider_width = 120
                with dpg.table(header_row=False, borders_innerV=False, policy=dpg.mvTable_SizingFixedFit):
                    dpg.add_table_column()  # X slider
                    dpg.add_table_column()  # Y slider
                    dpg.add_table_column()  # Z slider
                    dpg.add_table_column()  # row label

                    _rot_rows = [
                        ("Jaw (°)",         [(f"_slider_jaw_{ax}",      -45., 45., callback_jaw)            for ax in ['X', 'Y', 'Z']]),
                        ("Eyes (°)",        [(f"_slider_eye_100_{ax}",  -45., 45., callback_eyes)           for ax in ['X', 'Y', 'Z']]),
                        ("Neck (°)",        [(f"_slider_neck_{ax}",     -45., 45., callback_neck)           for ax in ['X', 'Y', 'Z']]),
                        # ("Head Trans",     [(f"_slider_head_trans_{ax}", -0.1, 0.1, make_head_trans_callback(132 + i)) for i, ax in enumerate(['X','Y','Z'])]),
                    ]

                    for _row_label, _sliders in _rot_rows:
                        with dpg.table_row():
                            for _tag, _mn, _mx, _cb in _sliders:
                                dpg.add_slider_float(label="", tag=_tag, width=_slider_width,
                                                     min_value=_mn, max_value=_mx, default_value=0.0,
                                                     callback=_cb)
                            dpg.add_text(_row_label)

                    with dpg.table_row():
                        for _label, _indent in [("Pitch", 40), ("Yaw", 44), ("Roll", 47)]:
                            dpg.add_text(_label, indent=_indent)
                        dpg.add_text("")

            # timestep slider and buttons
            # if self.num_timesteps != None:
            #     def callback_set_current_frame(sender, app_data):
            #         if sender == "_slider_timestep":
            #             self.timestep = app_data
            #         elif sender in ["_button_timestep_plus", "_mvKey_Right"]:
            #             self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
            #         elif sender in ["_button_timestep_minus", "_mvKey_Left"]:
            #             self.timestep = max(self.timestep - 1, 0)
            #         elif sender == "_mvKey_Home":
            #             self.timestep = 0
            #         elif sender == "_mvKey_End":
            #             self.timestep = self.num_timesteps - 1
            #
            #         dpg.set_value("_slider_timestep", self.timestep)
            #         self.gaussians.select_mesh_by_timestep(self.timestep)
            #
            #         self.need_update = True
            #
            #     with dpg.group(horizontal=True):
            #         dpg.add_button(label='-', tag="_button_timestep_minus", callback=callback_set_current_frame)
            #         dpg.add_button(label='+', tag="_button_timestep_plus", callback=callback_set_current_frame)
            #         dpg.add_slider_int(label="timestep", tag='_slider_timestep', width=153, min_value=0, max_value=self.num_timesteps - 1, format="%d",
            #                            default_value=0, callback=callback_set_current_frame)

            # Inputs for avatar
            with dpg.group(horizontal=True):
                def select_input_image(sender, app_data, user_data):
                    print("Sender: ", sender)
                    print("App Data: ", app_data)
                    self._custom_avatar_input_image_path = app_data['file_path_name']

                    custom_avatar_name = Path(app_data['file_path_name']).stem
                    if custom_avatar_name.startswith('avatar_code_'):
                        custom_avatar_name = custom_avatar_name[len('avatar_code_'):]
                    else:
                        print("[ERROR] Not an avatar code selected! Expected file with name avatar_code_XXX.npy")

                    if not self._avatar_code_manager.has_avatar_code(custom_avatar_name):
                        print(f"[ERROR] Could not load avatar code from file {Path(app_data['file_path_name'])}!")
                        # run_pixel3dmm(self._custom_avatar_input_image_path)

                    self._custom_avatar_name = custom_avatar_name
                    self._need_encoder = True

                dpg.add_file_dialog(directory_selector=False, show=False, callback=select_input_image, tag="file_dialog_input_image", width=700, height=400,
                                    default_path=f"{FLEXAVATAR_AVATAR_CODE_PATH}/itw")
                dpg.add_file_extension('.npy', parent='file_dialog_input_image')
                # dpg.add_file_extension('.png', parent='file_dialog_input_image')
                # with dpg.file_dialog(directory_selector=False, show=False, callback=select_input_image, id="file_dialog_input_image", width=700, height=400):
                #     dpg.add_file_extension(".*")
                #
                dpg.add_button(label="Load existing avatar", callback=lambda: dpg.show_item("file_dialog_input_image"))

            # camera
            with dpg.group(horizontal=True):
                def callback_reset_camera(sender, app_data):
                    self.cam.reset()
                    self.need_update = True
                    # dpg.set_value("_slider_fovy", self.cam.fovy)

                dpg.add_button(label="reset camera", tag="_button_reset_pose", callback=callback_reset_camera, show=not self.cfg.demo_mode)

            with dpg.group():
                with dpg.texture_registry(show=False):
                    dpg.add_raw_texture(width=self._webcam_buffer.shape[1], height=self._webcam_buffer.shape[0], default_value=self._webcam_buffer,
                                        format=dpg.mvFormat_Float_rgb,
                                        tag="webcam_buffer")

                dpg.add_image("webcam_buffer", label="Webcam Input")

                def callback_start_webcam(sender, app_data):
                    self._webcam_recording = app_data
                    if app_data:
                        print("Starting webcam thread")
                        thread = Thread(target=self._webcam_worker)
                        thread.start()

                def callback_create_picture_create_avatar(sender, app_data):
                    if self._webcam_recording:
                        self._webcam_recording = False
                        webcam_ids = Folder(f"{FLEXAVATAR_INPUTS_PATH}/webcam").list_file_numbering("webcam_$.png", return_only_numbering=True)
                        if not webcam_ids:
                            webcam_id = 0
                        else:
                            webcam_id = max(webcam_ids) + 1
                        custom_avatar_name = f"webcam_{webcam_id:03d}"
                        print(custom_avatar_name)
                        input_image_path = f"{FLEXAVATAR_INPUTS_PATH}/webcam/webcam_{webcam_id:03d}.png"
                        print(input_image_path)
                        save_img(self._webcam_buffer_large, input_image_path)
                        run_pixel3dmm(input_image_path)
                        self._custom_avatar_name = custom_avatar_name
                        self._need_encoder = True

            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Webcam", default_value=self._webcam_recording, callback=callback_start_webcam,
                                 tag="_checkbox_webcam_recording")
                dpg.add_button(label="Take Picture & Create Avatar", tag="_button_picture_create_avatar", callback=callback_create_picture_create_avatar)

        # widget-dependent handlers ========================================================================================
        # with dpg.handler_registry():
        #     dpg.add_key_press_handler(dpg.mvKey_Left, callback=callback_set_current_frame, tag='_mvKey_Left')
        #     dpg.add_key_press_handler(dpg.mvKey_Right, callback=callback_set_current_frame, tag='_mvKey_Right')
        #     dpg.add_key_press_handler(dpg.mvKey_Home, callback=callback_set_current_frame, tag='_mvKey_Home')
        #     dpg.add_key_press_handler(dpg.mvKey_End, callback=callback_set_current_frame, tag='_mvKey_End')
        #
        #     def callbackmouse_wheel_slider(sender, app_data):
        #         delta = app_data
        #         if dpg.is_item_hovered("_slider_timestep"):
        #             self.timestep = min(max(self.timestep - delta, 0), self.num_timesteps - 1)
        #             dpg.set_value("_slider_timestep", self.timestep)
        #             self.gaussians.select_mesh_by_timestep(self.timestep)
        #             self.need_update = True
        #
        #     dpg.add_mouse_wheel_handler(callback=callbackmouse_wheel_slider)

    def prepare_camera(self):
        @dataclass
        class Cam:
            FoVx = float(np.radians(self.cam.fovx))
            FoVy = float(np.radians(self.cam.fovy))
            image_height = self.cam.image_height
            image_width = self.cam.image_width
            world_view_transform = torch.tensor(self.cam.world_view_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            full_proj_transform = torch.tensor(self.cam.full_proj_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            camera_center = torch.tensor(self.cam.pose[:3, 3]).cuda()
            cx = self.cam.image_width / 2
            cy = self.cam.image_height / 2

        return Cam

    @torch.no_grad()
    def run(self):
        print("Running LocalViewer...")

        while dpg.is_dearpygui_running():

            if self.need_update or self.playing:
                cam = self.prepare_camera()

                rendering_output = render_gsplat(cam, self.gaussians, torch.tensor(self.cfg.background_color).cuda())
                rgb = rendering_output['render'].permute(1, 2, 0)

                self.render_buffer = rgb.cpu().numpy()
                if self.render_buffer.shape[0] != self.H or self.render_buffer.shape[1] != self.W:
                    continue
                dpg.set_value("_texture", self.render_buffer)

                self.refresh_stat()
                self.need_update = False

            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    gui = LocalViewer(cfg)
    gui.run()