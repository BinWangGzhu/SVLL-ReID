# Image re-identification: Where self-supervision meets vision-language learning [PDF](https://www.sciencedirect.com/science/article/abs/pii/S0262885625000034)
This is an implementation of Image Re-Identification: Where Self-supervisionMeets Vision-Language Learning  
### Pipeline
<img width="1373" height="750" alt="image" src="https://github.com/user-attachments/assets/dee677d5-39d9-40a1-a330-69458c05235c" />

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
    For example,you attempt to train your model on DukeMTMC:

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


### Important Parameters for SVLL-ReID:
```
Taking the person reid task as an example, you can find svllreid.yml in SAVLL-ReID\Codes\configs\person\config.
Below is the explanation of the key self-supervised parameters for SVLL-ReID:

Language Self-Supervision: STAGE1

Language Self-Supervision Weight: SSL_LOSS_WEIGHT

Method for Augmenting Prompts: PROMPT_AUG_METHOD

Augmentation Strength Rate for Prompts: PROMPT_AUG_SCALE

Vision Self-Supervision: STAGE2

Vision Self-Supervision Weight: SSL_LOSS_WEIGHT

Random Erasing Strength: INPUT: RE_PROB
```

### Enabling and Disabling Language and Vison Self-Supervision:

```
In SVLL-ReID/processor/processor_svllreid_stage1.py, set USE_SSL_stage1 to True or False.

In SVLL-ReID/processor/processor_svllreid_stage2.py, set USE_SSL_stage1 to True or False.
```
    

### Pre-trained model 


|       Datasets        |                            MSMT17                            |                            Market                            |                             Duke                             |                           Occ-Duke                           |                             VeRi                             |                          VehicleID                           |
| :-------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| SVLL-ReID | [model](https://drive.google.com/file/d/1AIhZRnbphAj3rzyLEtLuMOBGR7Fp8IRE/view?usp=drive_link)  | [model](https://drive.google.com/file/d/1jXc30q9p09B7hJQj2kIx-fn0oL8h3sl8/view?usp=drive_link)  | [model]()  | [model](https://drive.google.com/file/d/1wE_AQUB_uVKsyqYuCX0DoeG-sxppDC8k/view?usp=drive_link)  | [model]()  | [model]()  |


### Citation
If you use this code for your research, please cite
```
@article{WANG2025105415,
title = {Image re-identification: Where self-supervision meets vision-language learning},
journal = {Image and Vision Computing},
volume = {154},
pages = {105415},
year = {2025}
}
```

