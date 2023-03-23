# SMKD (wip)

This is the PyTorch implementation of ["Supervised Masked Knowledge Distillation for Few-Shot Transformers"](). 


[Han Lin\*](https://hl-hanlin.github.io/), [Guangxing Han\*](https://guangxinghan.github.io/), [Jiawei Ma](http://www.columbia.edu/~jm4743/), [Shiyuan Huang](https://shiyuanh.github.io/), [Xudong Lin](https://xudonglinthu.github.io/), [Shih-Fu Chang](https://www.ee.columbia.edu/~sfchang/)

Columbia University
Department of Computer Science
The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR), 2023


## Installation

Python 3.8, Pytorch 1.11, CUDA 11.3. The code is tested on Ubuntu 20.04.


We have prepared a conda YAML file which contains all the python dependencies.

```sh
conda env create -f environment.yml
```

To activate this conda environment,

```sh
conda activate smkd
```

We use [wandb](https://wandb.ai/site) to log the training stats (optional). 

## Datasets

We prepare 𝒎𝒊𝒏𝒊ImageNet and 𝒕𝒊𝒆𝒓𝒆𝒅ImageNet and resize the images following the guidelines from [HCTransformers](https://github.com/StomachCold/HCTransformers). 

- **𝒎𝒊𝒏𝒊ImageNet**


> The 𝑚𝑖𝑛𝑖ImageNet dataset was proposed by [Vinyals et al.](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) for few-shot learning evaluation. Its complexity is high due to the use of ImageNet images but requires fewer resources and infrastructure than running on the full [ImageNet dataset](https://arxiv.org/pdf/1409.0575.pdf). In total, there are 100 classes with 600 samples of color images per class. These 100 classes are divided into 64, 16, and 20 classes respectively for sampling tasks for meta-training, meta-validation, and meta-test. To generate this dataset from ImageNet, you may use the repository [𝑚𝑖𝑛𝑖ImageNet tools](https://github.com/y2l/mini-imagenet-tools).

Note that in our implemenation images are resized to 480 × 480 because the data augmentation we used require the image resolution to be greater than 224 to avoid distortions. Therefore, when generating 𝒎𝒊𝒏𝒊ImageNet, you should set ```--image_resize 0``` to keep the original size or ```--image_resize 480``` as what we did.



- **𝒕𝒊𝒆𝒓𝒆𝒅ImageNet**

> The [𝑡𝑖𝑒𝑟𝑒𝑑ImageNet](https://arxiv.org/pdf/1803.00676.pdf) dataset is a larger subset of ILSVRC-12 with 608 classes (779,165 images) grouped into 34 higher-level nodes in the ImageNet human-curated hierarchy. To generate this dataset from ImageNet, you may use the repository 𝑡𝑖𝑒𝑟𝑒𝑑ImageNet dataset: [𝑡𝑖𝑒𝑟𝑒𝑑ImageNet tools](https://github.com/y2l/tiered-imagenet-tools). 

Similar to 𝒎𝒊𝒏𝒊ImageNet, you should set ```--image_resize 0``` to keep the original size or ```--image_resize 480``` as what we did when generating 𝒕𝒊𝒆𝒓𝒆𝒅ImageNet.


- **CIFAR-FS and FC100**

CIFAR-FS and FC100 can be download using the [scripts](https://github.com/icoz69/DeepEMD/tree/master/datasets) from [DeepEMD](https://github.com/icoz69/DeepEMD). 


<!--- After getting the data, we can resize the images to 480 × 480 using ```create_cifar_fs.py``` and ```create_fc100.py``` under the ```./prepare_data``` directory." -->


## Training

Our model are trained on 8 RTX3090 GPUs by default (24GB memory). You can specify the argument ```--nproc_per_node``` in the following ```command``` file as the number of GPUs available in your server, and increase/decrease the argument ```--batch_size_per_gpu``` if your GPU has more/less memory.

- **Phase1 (self-supervised)**

In this phase, we pretrain our model using the self-supervised learning method [iBOT](https://github.com/bytedance/ibot). All models are trained for a maximum of 1600 epochs. We evaluate our model on the validation set after training for every 50 epochs, and report the best. 
1-shot and 5-shot evaluation results with _Prototype_ method is given in the following table. We also provide full checkpoints and test-set features for pretrained models, and command to replicate the results.

```--data_path```: need to be set as the location of the training set of dataset XXX (e.g. miniImageNet). 
```--output_dir```: location where the phase1 checkpoints and evaluation files to be stored.


<table>
  <tr>
    <th>Dataset</th>
    <th>1-shot</th>
    <th>5-shot</th>
    <th colspan="3">Download</th>
  </tr>
  <tr>
    <td>𝒎𝒊𝒏𝒊ImageNet</td>
    <td>60.93%</td>
    <td>80.38%</td>
    <td><a href="https://drive.google.com/file/d/1cHRiySKgrgbGqnNvMFY0D9IvWgpO75Jm/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/drive/folders/1YSxoCnuLidqwXsJCwnuvA3_6JEB4zFm1?usp=share_link">features</a></td>
    <td><a href="https://drive.google.com/file/d/1hJCVuLJQdGbvUjlRv2XzbsxLB6VQvKcm/view?usp=share_link">command</a></td>
  </tr>
  <tr>
    <td>𝒕𝒊𝒆𝒓𝒆𝒅ImageNet</td>
    <td>71.36%</td>
    <td>83.28%</td>
    <td><a href="https://drive.google.com/file/d/1udnoJrpOs5tcfSsGWBsoUQRiGaIZzx29/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/drive/folders/1i1XHoySqThAm_EOSxB6BhbA6BH6GRat4?usp=share_link">features</a></td>
    <td><a href="https://drive.google.com/file/d/1zjRRRzc_RU_jXcA8YGQVuyTgHeidDPmo/view?usp=share_link">command</a></td>
  </tr>
  <tr>
    <td>CIFAR-FS</td>
    <td>65.70%</td>
    <td>83.45%</td>
    <td><a href="https://drive.google.com/file/d/1tag6WuM9Ps1PnLgt7VoCIcPrxCqVEqO3/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/drive/folders/1phzC-CuER4QvhP3XTrl7a2g7uks6SCbK?usp=share_link">features</a></td>
    <td><a href="https://drive.google.com/file/d/1dGEUgq0HOJ0nL2jMHxdcNiVkJeOmUCdr/view?usp=share_link">command</a></td>
  </tr>
    <tr>
    <td>FC100</td>
    <td>44.20%</td>
    <td>61.64%</td>
    <td><a href="https://drive.google.com/file/d/1CAWtHJvvVKjptQh07sb9T50UeaqKYdru/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/drive/folders/1VRZ-McBcHHFwsA-h8QVDNBdSQK5CKrbH?usp=share_link">features</a></td>
    <td><a href="https://drive.google.com/file/d/1KhfZq2OcmTvT-xjCzaEI2NaHkCo45WnD/view?usp=share_link">command</a></td>
  </tr>
</table>


- **Phase2 (supervised)**

In this second phase, we start from the checkpoint in phase 1 and further train the model using the supervised knowledge distillation method proposed in our paper. All models are trained for a maximum of 150 epochs. We evaluate our model on the validation set after training for every 5 epochs, and report the best. Similarly, 1-shot and 5-shot evaluation results with _Prototype_ method is given in the following table. We also provide checkpoints and features for pretrained models.

```--pretrained_dino_path```: should be set as the same location as ```--output_dir``` in phase1. 
```--pretrained_dino_file```: which checkpoint file to resume from (e.g. ```checkpoint1250.pth```).
```--output_dir```: location where the phase2 checkpoints and evaluation files to be stored.

<table>
  <tr>
    <th>Dataset</th>
    <th>1-shot</th>
    <th>5-shot</th>
    <th colspan="3">Download</th>
  </tr>
  <tr>
    <td>𝒎𝒊𝒏𝒊ImageNet</td>
    <td>74.28%</td>
    <td>88.82%</td>
    <td><a href="https://drive.google.com/file/d/10dGfyf0t5dyhZ3WwcSzWZS6qIhsoayUz/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/drive/folders/1h8Rvyz5JQsTvxGg7GR1lO4XqhX8WfzNw?usp=share_link">features</a></td>
    <td><a href="https://drive.google.com/file/d/1r0TnVHZn_IKXi8A63Rj5RBNxwaF6--e-/view?usp=share_link">command</a></td>
  </tr>
  <tr>
    <td>𝒕𝒊𝒆𝒓𝒆𝒅ImageNet</td>
    <td>78.83%</td>
    <td>91.02%</td>
    <td><a href="https://drive.google.com/file/d/1Dbit0iKSXHtsdrxDTACsNbrLsp-J4XTh/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/drive/folders/1aFP8FozdkKdU1aF2GW6wDCNLugAgHF5P?usp=share_link">features</a></td>
    <td><a href="https://drive.google.com/file/d/1FXz7ZaRzej_T9qXpJ8vMoUPXHhu5bEN_/view?usp=share_link">command</a></td>
  </tr>
  <tr>
    <td>CIFAR-FS</td>
    <td>80.08%</td>
    <td>90.63%</td>
    <td><a href="https://drive.google.com/file/d/1CbWeO5HAGsCWpxr6xMbqTmgRdLH0NCLS/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/drive/folders/164wPnxW2bi6th_FMn8Da07IalhWsNRlo?usp=share_link">features</a></td>
    <td><a href="https://drive.google.com/file/d/1gdwa_6xrdqEE38iH9PaPinwIBZHW8g_J/view?usp=share_link">command</a></td>
  </tr>
    <tr>
    <td>FC100</td>
    <td>50.38%</td>
    <td>68.37%</td>
    <td><a href="https://drive.google.com/file/d/1YtrWdAdL_ywKLMa67iXOdA25-k0qNakw/view?usp=share_link">checkpoint</a></td>
    <td><a href="https://drive.google.com/drive/folders/1orw1YslQR5jkeJV89lXGBY_TfVXGiC3v?usp=share_link">features</a></td>
    <td><a href="https://drive.google.com/file/d/1SF8K_eFGPfWU558txXkWMZD-Nz9oUJJH/view?usp=share_link">command</a></td>
  </tr>
</table>



## Evaluation 

We use ```eval_smkd.py``` to evaluate a trained model (either from phase1 or phase2). Before running the evaluation code, we need to specify the image data path in ```server_dict``` of this python file.

For example, we can use the following code to do 5-way 5-shot evaluation on the model trained in phase2 on mini-ImageNet:

- **prototype**:
```sh
python eval_smkd.py --server mini --num_shots 5 --ckp_path /root/autodl-nas/FSVIT_results/MINI480_phase2 --ckpt_filename checkpoint0040.pth --output_dir /root/autodl-nas/FSVIT_results/MINI480_prototype --evaluation_method cosine --iter_num 10000
```

- **classifier**:
```sh
python eval_smkd.py --server mini --num_shots 5 --ckp_path /root/autodl-nas/FSVIT_results/MINI480_phase2 --ckpt_filename checkpoint0040.pth --output_dir /root/autodl-nas/FSVIT_results/MINI480_classifier --evaluation_method classifier --iter_num 1000
```




## Citation


## Acknowledgement

This repo is developed based on [HCTransformers](https://github.com/StomachCold/HCTransformers), [iBOT](https://github.com/bytedance/ibot) and [DINO](https://github.com/facebookresearch/dino). Thanks for their wonderful codebases.
