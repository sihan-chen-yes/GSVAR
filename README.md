# GS-VAR: High-fidelity Gaussian Splatting via Visual Autoregressive Models
Summer project at Disney Research Studios (DRS) at Zurich
<p align="center">
  <img src="./asset/arch.png" alt="Architecture" width="80%">
</p>

<p align="center">
  <img src="./asset/teaser.png" alt="Teaser" width="80%">
</p>

[report](./asset/GSVAR_report.pdf) \
see our [video](https://youtu.be/oFNeO3beryY) for more visual results
* We propose GS-VAR, the first method, to the best of our knowledge, that integrates a Visual Autoregressive (VAR) model into the 3DGS reconstruction pipeline. This enables real-time, high-fidelity enhancement of novel views rendered from 3DGS, offering a new direction beyond diffusion-based fixers

* We design two novel modules to improve enhancement quality: Reference Mixing Block (RMB), which aggregates spatial cues across neighboring views via multi-scale self-attention, and the VAE-bridge, which adaptively fuses input and generated content to mitigate hallucination and preserve structural fidelity

* Extensive experiment results show that GS-VAR achieves competitive results, paving the way for the practical deployment of VAR-based fixers in 3DGS-based 3D reconstruction pipeline

Limitations:
* VAR only works in image space, temporal consistency is not guaranteed
* Scale decomposition mechanism in VAR does not support dynamic resolution for input and output, while simple resize will drop
the performance drastically and tiling will make the enhancement stage even longer
* Only supports static scene currently, 3DGS with deformation field will be a natural and promising future path

## dataset preparation

1. install [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
2. download [DL3DV](https://dl3dv-10k.github.io/DL3DV-10K/)
3. prepare the triplets using the curation pipeline:

### curation triplets preparation

```python
# 3DGS reconstruction on DL3DV
python train.py -s ../dataset/DL3DV-10K-Benchmark/9641a1ed7963ce5ca734cff3e6ccea3dfa8bcb0b0a3ff78f65d32a080de2d71e/gaussian_splat --model_path ../dataset/DL3DV-10K-Benchmark/9641a1ed7963ce5ca734cff3e6ccea3dfa8bcb0b0a3ff78f65d32a080de2d71e/gs_recon --images images_4 --port 7008     --test_iterations 1000, 2000, 3000, 4000, 5000, 6000, 7_000, 30_000
    --save_iterations 1000, 2000, 3000, 4000, 5000, 6000, 7_000, 30_000

python render.py --model_path ../dataset/DL3DV-10K-Benchmark/9641a1ed7963ce5ca734cff3e6ccea3dfa8bcb0b0a3ff78f65d32a080de2d71e/gs_recon --iteration 1000 --images images_4

```

```shell
# sh for curation pipeline
python gen_data/gen_data.py --base_path ../dataset/DL3DV-10K-Benchmark --idx_start 0 --idx_end 10 --save_iter 1000 2000 3000 4000 5000 6000 7000 30000 --render_iter 1000 2000 3000 4000 5000 6000 7000 30000 --image_folder images_4 --strategy underfitting

python gen_data/gen_data.py --base_path ../dataset/DL3DV-10K-Benchmark --idx_start 105 --idx_end 140 --save_iter 30000 --render_iter 30000 --image_folder images_4 --strategy sparse --train_every 10

python gen_data/gen_data.py --base_path ../dataset/DL3DV-10K-Benchmark --idx_start 105 --idx_end 140 --save_iter 30000 --render_iter 30000 --image_folder images_4 --strategy sparse --train_every 40
```

## GSVAR

### installation

```shell
# create an environment with python >= 3.9
conda create -n gsvar python=3.9
conda activate gsvar
pip install -r requirements.txt

# for flex attn
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# flash attn ensure CUDA version 12.6!! (> 12) otherwise, gcc complains
MAX_JOBS=4 pip install flash-attn==2.5.6 --no-build-isolation

```

### checkpoints
* please download [VARSR](https://github.com/quyp2000/VARSR) ckpt for reproducing our result, we finetune our model based on the pretrained ckpt in ``VARSR``
* please download our [ckpt](https://1drv.ms/f/c/8763cc3dfead79f2/EjWzxlBY5_9HnrzhkVf5HboBCjkJGs_3MJtrj8MmBFdNFQ?e=unSRsN), and place it at ``GSVAR/results`` for evaluation
### train

We first train the fixer, including training in the latent space (stage 1), training in the image space(stage 2)

```python
###### stage 1 ######
# w ref w flex attn more 
torchrun --nproc-per-node=1 --master_addr=127.0.0.1 --master_port=29319 VAR_train.py --depth=24 --batch_size=16 --ep=5 --fp16=2 --tblr=5e-5 --alng=1e-4 --wpe=0.01 --wandb_flag=True --fuse=1 --exp_name='GS_VARSR_w_ref' --var_pretrain_path=checkpoints/VARSR.pth --use_lora=False --use_ref=True --use_ref_view_loss=False

# wo ref w flex attn more 
torchrun --nproc-per-node=1 --master_addr=127.0.0.1 --master_port=29712 VAR_train.py --depth=24 --batch_size=16 --ep=5 --fp16=2 --tblr=5e-5 --alng=1e-4 --wpe=0.01 --wandb_flag=True --fuse=1 --exp_name='GS_VARSR_wo_ref' --var_pretrain_path=checkpoints/VARSR.pth --use_lora=False --use_ref=False --use_ref_view_loss=False

###### stage 2 ######
# VAE bridge wo ref
torchrun --nproc-per-node=1 --master_addr=127.0.0.1  --master_port=29801 VAE_train.py --depth=24 --batch_size=4 --ep=5 --fp16=2 --tblr=5e-5 --alng=1e-4 --wpe=0.01 --wandb_flag=True --fuse=1 --exp_name='VAE_bridge_wo_ref' --var_pretrain_path=./results/GS_VARSR_wo_ref/ar-ckpt-last.pth --use_lora=False --use_ref=False --use_ref_drop=False --use_ref_view_loss=False


# VAE bridge w ref
torchrun --nproc-per-node=1 --master_addr=127.0.0.1  --master_port=29855 VAE_train.py --depth=24 --batch_size=4 --ep=5 --fp16=2 --tblr=5e-5 --alng=1e-4 --wpe=0.01 --wandb_flag=True --fuse=1 --exp_name='VAE_bridge_w_ref' --var_pretrain_path=./results/GS_VARSR_w_ref/ar-ckpt-last.pth --use_lora=False --use_ref=True --use_ref_drop=False --use_ref_view_loss=False

```



### test

#### fixer test

Fixer not integrated into 3DGS reconstruction pipeline yet, only evaluate in terms of image space:

```python
##############################################tiling ##############################################
############## w ref ################
# w bridge
python test_tile.py --var_test_path=./results/VAE_bridge_w_ref/vae-ckpt-last.pth --vae_model_path=./results/VAE_bridge_w_ref/vae-ckpt-last.pth --depth=24 --use_lora=False --testset_path=../dataset/DL3DV-val/underfitting --json_path=data/data_underfitting.json --use_ref=True --use_bridge=True


# wo bridge
python test_tile.py --var_test_path=./results/GS_VARSR_wo_ref/ar-ckpt-last.pth --depth=24 --use_lora=False --testset_path=../dataset/DL3DV-val/underfitting --json_path=data/data_underfitting.json --use_ref=True --use_bridge=False

############## wo ref ################

# w bridge
python test_tile.py --var_test_path=./results/VAE_bridge_wo_ref/vae-ckpt-last.pth --vae_model_path=./results/VAE_bridge_wo_ref/vae-ckpt-last.pth --depth=24 --use_lora=False --testset_path=../dataset/DL3DV-val/underfitting --json_path=data/data_underfitting.json --use_ref=False --use_bridge=True


# wo bridge
python test_tile.py --var_test_path=./results/GS_VARSR_wo_ref/ar-ckpt-last.pth --depth=24 --use_lora=False --testset_path=../dataset/DL3DV-val/underfitting --json_path=data/data_underfitting.json --use_ref=False --use_bridge=False



#############################################################################
# tiling
# underfitting subset
python test_tile.py --var_test_path=./results/VAE_bridge_w_ref/vae-ckpt-last.pth --vae_model_path=./results/VAE_bridge_w_ref/vae-ckpt-last.pth --depth=24 --use_lora=False --json_path=data/data_underfitting.json --testset_path=../dataset/DL3DV-val/underfitting --use_ref=True --use_bridge=True --fuse=1


# sparse subset
python test_tile.py --var_test_path=./results/VAE_bridge_w_ref/vae-ckpt-last.pth --vae_model_path=./results/VAE_bridge_w_ref/vae-ckpt-last.pth --depth=24 --use_lora=False --json_path=data/data_sparse.json --testset_path=../dataset/DL3DV-val/sparse --use_ref=True --use_bridge=True --fuse=1
#############################################################################
# resize
# underfitting
python test_varsr.py --var_test_path=./results/VAE_bridge_w_ref/vae-ckpt-last.pth --vae_model_path=./results/VAE_bridge_w_ref/vae-ckpt-last.pth --depth=24 --use_lora=False --json_path=data/data_underfitting.json --testset_path=../dataset/DL3DV-val/underfitting --use_ref=True --use_bridge=True --fuse=1


# sparse
python test_varsr.py --var_test_path=./results/VAE_bridge_w_ref/vae-ckpt-last.pth --vae_model_path=./results/VAE_bridge_w_ref/vae-ckpt-last.pth --depth=24 --use_lora=False --json_path=data/data_sparse.json --testset_path=../dataset/DL3DV-val/sparse --use_ref=True --use_bridge=True --fuse=1

```

#### Integrate into 3DGS reconstruction pipeline

Fixer integrated into the 3DGS reconstruction pipeline:

```python
# Difix3d
SCENE_ID=ded5e4b46aedbef4cdb7bd1db7fc4cc5b00a9979ad6464bdadfab052cd64c101
DATA=/cluster/work/drzrh/video-group/sihan/dataset/DL3DV-10K-Benchmark/${SCENE_ID}/gaussian_splat
DATA_FACTOR=4
EVERY=100
OUTPUT_DIR=/cluster/work/drzrh/video-group/sihan/dataset/DL3DV-val/output/difix/gsplat/${SCENE_ID}/${EVERY}
CKPT_PATH=${OUTPUT_DIR}/ckpts/ckpt_39999_rank0.pt # Path to the pretrained checkpoint file

CUDA_VISIBLE_DEVICES=0 python -m examples.gsplat.simple_trainer_difix3d default \
    --data_dir ${DATA} --data_factor ${DATA_FACTOR} \
    --result_dir ${OUTPUT_DIR} --no-normalize-world-space --test_every ${EVERY} 
    

# GSVAR
SCENE_ID=ded5e4b46aedbef4cdb7bd1db7fc4cc5b00a9979ad6464bdadfab052cd64c101
DATA=../dataset/DL3DV-10K-Benchmark/${SCENE_ID}/gaussian_splat
DATA_FACTOR=4
EVERY=100
OUTPUT_DIR=../DL3DV-val/output/gsvar/gsplat/${SCENE_ID}/${EVERY}
CKPT_PATH=${OUTPUT_DIR}/ckpts/ckpt_34999_rank0.pt # Path to the pretrained checkpoint file

CUDA_VISIBLE_DEVICES=0 python -m examples.gsplat.simple_trainer_difix3d default \
    --data_dir ${DATA} --data_factor ${DATA_FACTOR} \
    --result_dir ${OUTPUT_DIR} --no-normalize-world-space --var_test_path=./results/GS_VARSR_w_ref_1_flex_more/ar-ckpt-last.pth --depth=24 --use_ref --test_every ${EVERY} 
    

# vanilla 3DGS
SCENE_ID=ded5e4b46aedbef4cdb7bd1db7fc4cc5b00a9979ad6464bdadfab052cd64c101
DATA=/cluster/work/drzrh/video-group/sihan/dataset/DL3DV-10K-Benchmark/${SCENE_ID}/gaussian_splat
DATA_FACTOR=4
EVERY=50
OUTPUT_DIR=/cluster/work/drzrh/video-group/sihan/dataset/DL3DV-val/output/3DGS/gsplat/${SCENE_ID}/${EVERY}
CKPT_PATH=${OUTPUT_DIR}/ckpts/ckpt_39999_rank0.pt # Path to the pretrained checkpoint file

CUDA_VISIBLE_DEVICES=0 python -m examples.gsplat.simple_trainer_difix3d default \
    --data_dir ${DATA} --data_factor ${DATA_FACTOR} \
    --result_dir ${OUTPUT_DIR} --no-normalize-world-space --test_every ${EVERY} --no-fix
    
```



```shell
FIXER=gsvar EVERY=10 USE_REF=True OUTPUT_BASE_DIR=../dataset/DL3DV-val/output/gsvar_org/gsplat bash scripts/recon_gsvar.sh ; FIXER=gsvar EVERY=35 USE_REF=True OUTPUT_BASE_DIR=../dataset/DL3DV-val/output/gsvar_org/gsplat bash scripts/recon_gsvar.sh;
FIXER=gsvar EVERY=50 USE_REF=True OUTPUT_BASE_DIR=../dataset/DL3DV-val/output/gsvar_org/gsplat bash scripts/recon_gsvar.sh ; FIXER=gsvar EVERY=100 USE_REF=True OUTPUT_BASE_DIR=../dataset/DL3DV-val/output/gsvar_org/gsplat bash scripts/recon_gsvar.sh ;

FIXER=gsvar EVERY=10 USE_REF=True OUTPUT_BASE_DIR=../dataset/DL3DV-val/output/gsvar_dir/gsplat bash scripts/recon_gsvar.sh ; FIXER=gsvar EVERY=35 USE_REF=True OUTPUT_BASE_DIR=../dataset/DL3DV-val/output/gsvar_dir/gsplat bash scripts/recon_gsvar.sh;
FIXER=gsvar EVERY=50 USE_REF=True OUTPUT_BASE_DIR=../dataset/DL3DV-val/output/gsvar_dir/gsplat bash scripts/recon_gsvar.sh ; FIXER=gsvar EVERY=100 USE_REF=True OUTPUT_BASE_DIR=../dataset/DL3DV-val/output/gsvar_dir/gsplat bash scripts/recon_gsvar.sh ;
```

