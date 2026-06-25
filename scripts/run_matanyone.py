import tyro
from elias.util import ensure_directory_exists, save_img
from tqdm.contrib.concurrent import thread_map

from flexavatar.config.dataset_config import SampleMetadata
from flexavatar.constants import VFHQ_TEST_DATASET_NAME
from flexavatar.data_adapter.vfhq_data_adapter import VFHQTestDataAdapter, VFHQTestDataFolder
from flexavatar.env import FLEXAVATAR_MATANYONE_PROCESSING_PATH
from flexavatar.model.matanyone_model import MatAnyoneModel


def main(dataset_name: str = VFHQ_TEST_DATASET_NAME, start_idx: int = 0, n_videos: int = -1):
    if dataset_name == VFHQ_TEST_DATASET_NAME:
        data_folder = VFHQTestDataFolder()


    video_keys = data_folder.list_video_keys()
    if n_videos == -1:
        n_videos = len(video_keys)

    matanyone_model = MatAnyoneModel()

    for video_key in video_keys[start_idx: start_idx + n_videos]:
        if dataset_name == VFHQ_TEST_DATASET_NAME:
            data_adapter = VFHQTestDataAdapter(video_key)

        timesteps = data_adapter.list_timesteps()
        images = thread_map(lambda timestep: data_adapter.load_image(SampleMetadata(video_key, None, timestep, None)), timesteps)


        processed_masks = matanyone_model.process_video_memory(images)
        output_folder = f"{FLEXAVATAR_MATANYONE_PROCESSING_PATH}/{dataset_name}/{video_key}"
        ensure_directory_exists(output_folder)
        for timestep, image in zip(timesteps, processed_masks):
            save_img(image, f"{output_folder}/{timestep:05}.png", image)

if __name__ == '__main__':
    tyro.cli(main)