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

Stage 1: Language Self-Supervision

Language Self-Supervision Weight: SSL_LOSS_WEIGHT

Method for Augmenting Prompts: PROMPT_AUG_METHOD

Augmentation Strength Rate for Prompts: PROMPT_AUG_SCALE

Stage 2: Visual Self-Supervision

Visual Self-Supervision Weight: SSL_LOSS_WEIGHT

Random Erasing Strength: INPUT: RE_PROB

Enabling and Disabling Language and Visual Self-Supervision:

In SVLL-ReID/processor/processor_svllreid_stage1.py, set USE_SSL_stage1 to True or False.

In SVLL-ReID/processor/processor_svllreid_stage2.py, set USE_SSL_stage1 to True or False.
```
    

### Pre-trained model 


|       Datasets        |                            MSMT17                            |                            Market                            |                             Duke                             |                           Occ-Duke                           |                             VeRi                             |                          VehicleID                           |
| :-------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| SVLL-ReID | [model](https://drive.google.com/file/d/1AIhZRnbphAj3rzyLEtLuMOBGR7Fp8IRE/view?usp=drive_link)  | [model](https://drive.google.com/file/d/1jXc30q9p09B7hJQj2kIx-fn0oL8h3sl8/view?usp=drive_link)  | [model]()  | [model](https://drive.google.com/file/d/1wE_AQUB_uVKsyqYuCX0DoeG-sxppDC8k/view?usp=drive_link)  | [model]()  | [model]()  |

