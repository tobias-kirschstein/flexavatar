# Official Implementation of FlexAvatar [CVPR '26]
From the paper *"FlexAvatar: Learning Complete 3D Head Avatars with Partial Supervision"*


[Paper](https://tobias-kirschstein.github.io/flexavatar/static/FlexAvatar_paper.pdf) | [Video](https://youtu.be/g8wxqYBlRGY) | [Project Page](https://tobias-kirschstein.github.io/flexavatar/)  
![](static/flexavatar_teaser.jpg)
[Tobias Kirschstein](https://tobias-kirschstein.github.io/), [Simon Giebenhain](https://simongiebenhain.github.io/), [Matthias Nießner](https://www.niessnerlab.org/)  
**CVPR 2026**

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
![](static/rendering_marble_sculpture.gif)

## 2.2. Create Avatars for Custom Inputs
*Coming soon...*

## 2.3. Drive Avatars with Custom Videos
*Coming soon...*

<hr>

If you find this repository useful please consider citing
```bibtex
@inproceedings{kirschstein2026flexavatar,
  title={Flexavatar: Learning complete 3d head avatars with partial supervision},
  author={Kirschstein, Tobias and Giebenhain, Simon and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18193--18203},
  year={2026}
}
```