# 4D Gaussian Head Reconstruction
## ⭐Project in 3rd YAICON⭐


<img src="https://github.com/whwjdqls/4D-Gaussian-Head/blob/master/assets/card1.png" width="400" height="400"/>

<img src="https://github.com/whwjdqls/4D-Gaussian-Head/blob/master/assets/card2.png" width="400" height="400"/> <img src="https://github.com/whwjdqls/4D-Gaussian-Head/blob/master/assets/card3.png" width="400" height="400"/>

<img src="https://github.com/whwjdqls/4D-Gaussian-Head/blob/master/assets/card4.png" width="400" height="400"/> <img src="https://github.com/whwjdqls/4D-Gaussian-Head/blob/master/assets/card5.png" width="400" height="400"/>

<img src="https://github.com/whwjdqls/4D-Gaussian-Head/blob/master/assets/card6.png" width="400" height="400"/> <img src="https://github.com/whwjdqls/4D-Gaussian-Head/blob/master/assets/card7.png" width="400" height="400"/>


## Results: Test Frames
Result with Expression -> Artifacts Alleviated (check 0:04)
| with out the FLAME expression prior |With the FLAME expression prior|
|:---:|:---:|
| <video width="10" height="50" src="https://github.com/whwjdqls/4D-Gaussian-Head/assets/73946308/73209cc2-2166-4e21-ac12-c43505e98950" />|<video width="50" height="50" src="https://github.com/whwjdqls/4D-Gaussian-Head/assets/73946308/cf82b8c5-41fe-41ba-a1fc-4996e78497d5" />|
 
**※ Use NF-exp branch to implement the 4D gaussian with additional input(expression)**
## Environmental Setups
Please follow the [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) to install the relative packages.
```bash
git clone https://github.com/whwjdqls/4D-Gaussian-Head
cd 4D-Gaussian-Head
git submodule update --init --recursive
conda create -n 4D-Gaussian-Head python=3.7 
conda activate 4D-Gaussian-Head

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```
In our environment, we use pytorch=1.13.1+cu116.
## Data Preparation
**For synthetic scenes:**  
The dataset provided in [D-NeRF](https://github.com/albertpumarola/D-NeRF) is used. You can download the dataset from [dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0).

**For real dynamic scenes:**  
The dataset provided in [HyperNeRF](https://github.com/google/hypernerf) is used. You can download scenes from [Hypernerf Dataset](https://github.com/google/hypernerf/releases/tag/v0.1) and organize them as [Nerfies](https://github.com/google/nerfies#datasets). Meanwhile, [Plenoptic Dataset](https://github.com/facebookresearch/Neural_3D_Video) could be downloaded from their official websites. To save the memory, you should extract the frames of each video and then organize your dataset as follows.
```
├── data
│   | dnerf 
│     ├── mutant
│     ├── standup 
│     ├── ...
│   | hypernerf
│     ├── interp
│     ├── misc
│     ├── virg
│   | dynerf
│     ├── cook_spinach
│       ├── cam00
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── 0002.png
│               ├── ...
│       ├── cam01
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│     ├── cut_roasted_beef
|     ├── ...
```


## Training
For training synthetic scenes such as `bouncingballs`, run 
``` 
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py 
``` 
You can customize your training config through the config files.
## Rendering
Run the following script to render the images.  

```
python render.py --model_path "output/dnerf/bouncingballs/"  --skip_train --configs arguments/dnerf/bouncingballs.py  &
```

## Citation
This project face is a fork from 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering project page. 
```
@article{wu20234dgaussians,
  title={4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
  author={Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and Zhang, Xiaopeng and Wei Wei and Liu, Wenyu and Tian, Qi and Wang Xinggang},
  journal={arXiv preprint arXiv:2310.08528},
  year={2023}
}

```
