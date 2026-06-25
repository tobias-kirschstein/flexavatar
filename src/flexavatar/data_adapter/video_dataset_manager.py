import os
import re
import cv2
import numpy as np

from flexavatar.env import FLEXAVATAR_DATASETS_PATH


class VideoDatasetManager:

    def __init__(self, dataset_name: str):
        self._location = f"{FLEXAVATAR_DATASETS_PATH}/{dataset_name}"

    def is_folder(self, video_key: str) -> bool:
        return os.path.isdir(os.path.join(self._location, video_key))

    def is_video(self, video_key: str) -> bool:
        return os.path.isfile(os.path.join(self._location, f"{video_key}.mp4"))

    def get_video_path(self, video_key: str) -> str:
        if self.is_folder(video_key):
            return os.path.join(self._location, video_key)
        return os.path.join(self._location, f"{video_key}.mp4")

    def load_image(self, video_key: str, timestep: int) -> np.ndarray:
        folder_path = os.path.join(self._location, video_key)
        if os.path.isdir(folder_path):
            frames = sorted(
                f for f in os.listdir(folder_path)
                if re.search(r'\.(png|jpg|jpeg)$', f, re.IGNORECASE)
            )
            img_path = os.path.join(folder_path, frames[timestep])
            img = cv2.imread(img_path)
        else:
            video_path = os.path.join(self._location, f"{video_key}.mp4")
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, timestep)
            ret, img = cap.read()
            cap.release()
            if not ret:
                raise ValueError(f"Could not read frame {timestep} from {video_path}")

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

