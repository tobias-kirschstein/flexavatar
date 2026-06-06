# Official FlexAvatar Implementation [CVPR '26]
From the paper *"FlexAvatar: Learning Complete 3D Head Avatars with Partial Supervision"*

## 1. Setup
1. Create conda environment `flexavatar` with newest PyTorch and CUDA 11.8:
    ```shell
    conda env create -f environment.yml
    ```

2. Install the `flexavatar` Python module
    ```shell
    pip install -e .
    ```
3. Download the pre-trained `FlexAvatar` model weights file from [https://nextcloud.tobias-kirschstein.de/index.php/s/X29kqKNndpSAKfB](https://nextcloud.tobias-kirschstein.de/index.php/s/X29kqKNndpSAKfB) and put it into `models/FLEX-1/checkpoints/ckpt-900k.pt`.

# 2. Usage 

## 2.1. Render & Animate Example Avatars
The folder [data/inputs/itw](data/inputs/itw) contains example input images for which all preprocessing files are already present in the repository.
For these images, 3D head avatars can be created and rendered via:
```shell
python scripts/render_example.py
```
The resulting renderings will be stored in the `renderings` folder in the repository.

## 2.2. Create Avatars for Custom Inputs
*Coming soon...*

## 2.3. Drive Avatars with Custom Videos
*Coming soon...*