import os
from pathlib import Path
from shutil import rmtree, copy, copy2
from typing import Optional

import tyro
from elias.util import ensure_directory_exists_for_file, ensure_directory_exists
from pixel3dmm.scripts.run_pixel3dmm import main as run_pixel3dmm

from flexavatar.data_adapter.video_dataset_manager import VideoDatasetManager
from flexavatar.env import FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH, FLEXAVATAR_DATASETS_PATH


def main(video_key: Optional[str] = None, /):
    dataset_name = 'VFHQ-Test'
    if video_key is None:
        video_names = [path.stem for path in Path(f"{FLEXAVATAR_DATASETS_PATH}/{dataset_name}").iterdir()]
    else:
        video_names = [video_key]

    for video_name in video_names:
        pixel3dmm_tracking_path = f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/tracking/{dataset_name}/{video_name}/tracking_nV1_noPho_uv2000.0_n1000.0/result.mp4"
        if Path(pixel3dmm_tracking_path).exists():
            print(f"[Skipping] {video_name} because Pixel3DMM tracking already exists")
            continue

        data_manager = VideoDatasetManager(dataset_name)
        video_path = data_manager.get_video_path(video_name)
        pixel3dmm_image_folder = f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/processing/input/{dataset_name}/{video_name}"

        try:
            is_video = data_manager.is_video(video_name)
            is_folder = data_manager.is_folder(video_name)

            if is_video:
                pixel3dmm_video_path = f"{pixel3dmm_image_folder}/{video_name}.mp4"
                ensure_directory_exists_for_file(pixel3dmm_video_path)
                copy(video_path, pixel3dmm_video_path)
            elif is_folder:
                pixel3dmm_video_path = pixel3dmm_image_folder
                ensure_directory_exists(pixel3dmm_video_path)
                files = os.listdir(video_path)

                for file_name in files:
                    copy2(f"{video_path}/{file_name}", pixel3dmm_video_path)
            else:
                raise ValueError(f"Video {video_name} is neither a .mp4 nor a folder of images")

            run_pixel3dmm(pixel3dmm_video_path,
                          f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/processing/{dataset_name}",
                          f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/tracking/{dataset_name}",
                          cleanup=True,
                          is_discontinuous=False)
            if Path(pixel3dmm_image_folder).is_dir():
                rmtree(pixel3dmm_image_folder)
        except Exception as e:
            print(f"[ERROR] Skipping {video_name}")
            print(e)

            if Path(pixel3dmm_image_folder).is_dir():
                rmtree(pixel3dmm_image_folder)

            raise e

if __name__ == '__main__':
    tyro.cli(main)