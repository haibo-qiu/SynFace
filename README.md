# SynFace: Face Recognition with Synthetic Data
This is the Pytorch implementation of our ICCV 2021 paper
>[SynFace: Face Recognition with Synthetic Data](https://arxiv.org/abs/2108.07960). 
><br>Haibo Qiu, Baosheng Yu, Dihong Gong, Zhifeng Li, Wei Liu and Dacheng Tao<br>

## Requirements
Main packages:
- python=3.6.7
- pytorch=1.8.1
- torchvision=0.9.1
- cudatoolkit=10.2.89

Or directly create a conda env with
```
conda env create -f environment.yml
```

## Data preparation
1. Clone this repo:
    ```
    git clone https://github.com/haibo-qiu/SynFace.git
    ```
2. Clone the [DiscoFaceGAN](https://github.com/microsoft/DiscoFaceGAN) and insert our files as the mixup face generator (To run [DiscoFaceGAN](https://github.com/microsoft/DiscoFaceGAN), you also need to satisfy its requirements.):
    ```
    git clone https://github.com/microsoft/DiscoFaceGAN.git data/DiscoFaceGAN
    cp data/syn_factors.py data/DiscoFaceGAN/
    cp data/syn_images.py data/DiscoFaceGAN/
    ```

3. Generate the face images with identity mixup, following with face alignment and crop:
    ```
    bash data/syn.sh
    ```
4. (Optional) Check our generated synthetic dataset via this onedirve [link](https://unisydneyedu-my.sharepoint.com/:u:/g/personal/hqiu2518_uni_sydney_edu_au/EST71RzmSUNEoM34T3WEeh4BhVcw_HmrqcK-vOWX0dmxAg).
5. Download the real face data CASIA and LFW from [this link](https://drive.google.com/drive/folders/1XTkS2Rh7Q154rwcV0MfhZ69cG10bEFDt?usp=sharing)
6. Put all these data into ```data/datasets/```

## Training
Simply run the following script:
```
bash run.sh
```

## Testing
To reproduce the results in our paper, please download the [pretrained models](https://drive.google.com/drive/folders/1XTkS2Rh7Q154rwcV0MfhZ69cG10bEFDt?usp=sharing) and put them in ```pretrained/```, then run:
```
bash eval.sh
```

## Acknowledgement
The code of face alignment and crop ```data/imgs_crop/``` is borrowed from [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe#Data-Zoo) and re-written with multi-processing for acceleration.

## Citation
If you use our code or models in your research, please cite with:
```bibtex
@inproceedings{qiu2021synface,
  title={SynFace: Face Recognition with Synthetic Data},
  author={Qiu, Haibo and Yu, Baosheng and Gong, Dihong and Li, Zhifeng and Liu, Wei and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10880--10890},
  year={2021}
}
```
