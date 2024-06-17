# SVLL-ReID
This is an implementation of Image Re-Identification: Where Self-supervisionMeets Vision-Language Learning
### Requirements
```
conda create -n svll-reid python=3.8
conda activate svll-reid
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install yacs
pip install timm
pip install scikit-image
pip install tqdm
pip install ftfy
pip install regex
```

### Train
```
    For example,you attempt to train our model on DukeMTMC:

        1, modify here: (in configs/person/vit_svllreid.yml)
            ###
            DATASETS:
               NAMES: ('dukemtmc')
            #   NAMES: ('occ_duke')
            #   NAMES: ('market1501')
            #   NAMES: ('msmt17')
               ROOT_DIR: ('/your_dataset_dir')
            OUTPUT_DIR: '/your_output_dir'
            ###

        2, and run:
            ###
            CUDA_VISIBLE_DEVICES=0 python train_svlleid.py --config_file configs/person/vit_svllreid.yml
            ###
```
### Test
    CUDA_VISIBLE_DEVICES=0 python test_svllreid.py --config_file configs/person/vit_svllreid.yml TEST.WEIGHT 'your_output_dir/your.pth'

### Pre-trained model 


|       Datasets        |                            MSMT17                            |                            Market                            |                             Duke                             |                           Occ-Duke                           |                             VeRi                             |                          VehicleID                           |
| :-------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| SVLL-ReID | [model](https://drive.google.com/file/d/1AIhZRnbphAj3rzyLEtLuMOBGR7Fp8IRE/view?usp=drive_link)  | [model](https://drive.google.com/file/d/1jXc30q9p09B7hJQj2kIx-fn0oL8h3sl8/view?usp=drive_link)  | [model](https://drive.google.com/file/d/1BVaZo93kOksYLjFNH3Gf7JxIbPlWSkcO/view?usp=share_link)  | [model](https://drive.google.com/file/d/1wE_AQUB_uVKsyqYuCX0DoeG-sxppDC8k/view?usp=drive_link)  | [model](https://drive.google.com/file/d/1BVaZo93kOksYLjFNH3Gf7JxIbPlWSkcO/view?usp=share_link)  | [model](https://drive.google.com/file/d/1BVaZo93kOksYLjFNH3Gf7JxIbPlWSkcO/view?usp=share_link)  |

