# HMD-Poser: On-Device Real-time Human Motion Tracking from Scalable Sparse Observations

> [**HMD-Poser: On-Device Real-time Human Motion Tracking from Scalable Sparse Observations**](https://pico-ai-team.github.io/hmd-poser)  
> Peng Dai, Yang Zhang, Tao Liu, Zhen Fan, Tianyuan Du, Zhuo Su, Xiaozheng Zheng, Zeming Li  
> PICO, ByteDance  
> :partying_face: <strong>Accepted to CVPR 2024</strong>

[[`arXiv`](https://arxiv.org/abs/2403.03561)] [[`Project`](https://pico-ai-team.github.io/hmd-poser)]


## :mega: Updates
- [x] Release the pre-trained models and the evaluation results.
      
- [x] Release the PICO-FreeDancing dataset.
      
- [x] Release the training and testing codes.  


## :desktop_computer: Requirements
### 
- Python >= 3.9
- PyTorch >= 2.0.1
- numpy >= 1.23.1
- [human_body_prior](https://github.com/nghorbani/human_body_prior)


## :hammer_and_pick: Preparation
### AMASS
1. Please download the datasets from [AMASS](https://amass.is.tue.mpg.de/) and place them in `./data/AMASS` directory of this repository.
2. Download the required body models and place them in `./body_models` directory of this repository. For the SMPL+H body model, download it from http://mano.is.tue.mpg.de/. Please download the AMASS version of the model with DMPL blendshapes. You can obtain dynamic shape blendshapes, e.g. DMPLs, from http://smpl.is.tue.mpg.de.
3. Run  `./prepare_data.py` to preprocess the input data for faster training. The data split for training and testing data under Protocol 1 in our paper is stored under the folder `./prepare_data/data_split` (directly copy from [AvatarPoser](https://github.com/eth-siplab/AvatarPoser)).

```
python ./prepare_data.py --support_dir ./body_models/ --root_dir ./data/AMASS/ --save_dir [path_to_save]
```

## :bicyclist: Training
1. Modify the dataset_path in `./options/train_config.yaml` to your `[path_to_save]`.

```
python train.py --config ./options/train_config.yaml
```


## :running_woman: Evaluation
1. Modify the resume_model path in `./options/test_config.yaml`.
```
python test.py --config ./options/test_config.yaml
```

## :lollipop: Pre-trained Model
### Protocol1

Trained Model: `pretrained_model/pretrained_model_protocol1.pt`.

| Input Type | MPJRE  | MPJPE  | MPJVE  |
| :--------- | :----: | :----: | :----: |
| HMD        | 2.29   | 3.15   | 17.52  |
| HMD+2IMUs  | 1.88   | 2.30   | 13.34  |
| HMD+3IMUs  | 1.79   | 2.01   | 12.70  |


## :tada: PICO-FreeDancing Dataset
### Brief description of the dataset
There are in total 74 free-dancing motions from 8 subjects (3 male and 5 female).

For each motion, there are two files: `gt_body_parms.pt` and `hmd_sensor_data.pt`.

`gt_body_parms.pt` contains the ground-truth SMPL parameters obtained via OptiTrack and Mosh++.

`hmd_sensor_data.pt` contains the synchronized real-captured HMD and IMU sensor data.
Specifically, it has three types of data: 
- `sensor_coordinates`: with a shape of N * [head, left_hand, right_hand] * 3,
- `sensor_orientation`: with a shape of N * [head, left_hand, right_hand, left_foot, right_foot] * 3 * 3
- `sensor_acceleration`: with a shape of N * [head, left_hand, right_hand, left_foot, right_foot] * 3

where N is the number of frames.

### Download
[Google Drive](https://drive.google.com/file/d/1xEj3J0vJilx-jPCPbbsX6a6IF9LQ5URu/view?usp=drive_link)


## :love_you_gesture: Citation
If you find our work useful for your research, please consider citing the paper:
```
@inproceedings{
  daip2024hmdposer,
  title={HMD-Poser: On-Device Real-time Human Motion Tracking from Scalable Sparse Observations},
  author={Dai, Peng and Zhang, Yang and Liu, Tao and Fan, Zhen and Du, Tianyuan and Su, Zhuo and Zheng, Xiaozheng and Li, Zeming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```



## :newspaper_roll: License

Distributed under the MIT License. See `LICENSE` for more information.


## :raised_hands: Acknowledgements
This project refers to source codes shared by [AvatarPoser](https://github.com/eth-siplab/AvatarPoser). We thank the authors for their great job!
