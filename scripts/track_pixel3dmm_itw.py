import os
from pathlib import Path
from shutil import rmtree, copy, copy2
from typing import Optional

import tyro
from elias.util import ensure_directory_exists_for_file, load_img, save_img, ensure_directory_exists

from flexavatar.data_adapter.in_the_wild_data_adapter import InTheWildDataAdapter
from flexavatar.env import FLEXAVATAR_INPUTS_PATH, FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH
from pixel3dmm.scripts.run_pixel3dmm import main as run_pixel3dmm


def main(source_person: Optional[str] = None, /):
    if source_person is None:
        itw_names = [path.stem for path in Path(f"{FLEXAVATAR_INPUTS_PATH}/itw").iterdir()]
    else:
        itw_names = [source_person]

    for image_name in itw_names:
        pixel3dmm_tracking_path = f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/tracking/itw/{image_name}/tracking_nV1_noPho_uv2000.0_n1000.0/result.mp4"
        if Path(pixel3dmm_tracking_path).exists():
            print(f"[Skipping] {image_name} because Pixel3DMM tracking already exists")
            continue

        try:
            data_adapter = InTheWildDataAdapter(image_name)
            image_path = data_adapter.get_image_path(image_name)
            pixel3dmm_image_folder = f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/processing/input/itw/{image_name}"
            is_video = data_adapter.is_video()
            is_folder = data_adapter.is_folder()

            if is_video:
                pixel3dmm_image_path = f"{pixel3dmm_image_folder}/{image_name}.mp4"
                ensure_directory_exists_for_file(pixel3dmm_image_path)
                copy(image_path, pixel3dmm_image_path)
            elif is_folder:
                pixel3dmm_image_path = pixel3dmm_image_folder
                ensure_directory_exists(pixel3dmm_image_path)
                files = os.listdir(image_path)

                for file_name in files:
                    copy2(f"{image_path}/{file_name}", pixel3dmm_image_path)
            else:
                pixel3dmm_image_path = f"{pixel3dmm_image_folder}/{image_name}.jpg"
                ensure_directory_exists_for_file(pixel3dmm_image_path)
                image = load_img(image_path)
                save_img(image[..., :3], pixel3dmm_image_path)

            run_pixel3dmm(pixel3dmm_image_path,
                          f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/processing/itw",
                          f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/tracking/itw",
                          cleanup=True,
                          is_discontinuous=is_folder)
            if Path(pixel3dmm_image_folder).is_dir():
                rmtree(pixel3dmm_image_folder)
        except Exception as e:
            print(f"[ERROR] Skipping {image_name}")
            print(e)
            raise e

if __name__ == '__main__':
    tyro.cli(main)