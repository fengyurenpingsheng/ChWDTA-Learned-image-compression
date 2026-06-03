# ChWDTA-Learned-image-compression

# ChWDTA: Channel-wise Wavelet-Structured Transformer and Entropy Modeling for Learned Image Compression



## Introduction

This repository is the official PyTorch implementation of the paper *"ChWDTA: Channel-wise Wavelet-Structured Transformer and Entropy Modeling for Learned Image Compression"*.

**Abstract:** State-of-the-art learned image compression (LIC) schemes are increasingly based on hybrid CNN–transformer architectures. To further improve rate–distortion performance, we introduce channel-wise wavelet transforms into both the transformer and entropy-coding components. First, we propose a channel-wise wavelet-domain transformer attention (ChWDTA) mechanism. ChWDTA keeps the efficient windowed spatial self-attention used in modern LIC backbones, but computes the Q/K/V projections on channel-wise wavelet-transformed features before mapping the attention output back with the inverse transform. The resulting Channel-wise Wavelet-Domain Transformer Block (ChWDTB) therefore preserves the spatial tokenization pattern of windowed attention while sparsifying the channel covariance seen by the attention projections. Second, in the entropy-coding stage, we introduce a channel-wise wavelet packet (ChWP) decomposition that produces four equal-sized subbands, which better fit channel-wise slice-based autoregressive entropy modeling. When each channel-wise subband is divided into two slices, we use eight slices for entropy coding. With this configuration, the proposed scheme obtains BD-rate reductions of −17.82%, −19.15%, and −22.56% on the Kodak, CLIC Professional Validation, and Tecnick test sets, respectively. Even when each channel-wise subband is coded as a single slice, the scheme still retains most of the coding gains with lower complexity. The results confirm the advantage of introducing wavelet transform in the CNN-transformer-based LIC schemes. 


### Performance

Our ChWDTA model achieves state-of-the-art rate-distortion performance with competitive computational complexity. Compared with VVC VTM-9.1, ChWDTA obtains BD-rate reductions of **-17.82%**, **-19.15%**, and **-22.56%** on Kodak, CLIC Professional Validation, and Tecnick, respectively.

![image](assets/Rate_speed_comparison_on_Kodak.PNG)

***

![image](assets/sota.PNG)

***

![image](assets/ablation.PNG)

### Installation

Clone this repository:

```bash
git clone https://github.com/fengyurenpingsheng/ChWDTA-Learned-image-compression.git
cd ChWDTA-Learned-image-compression
```

Install the required packages:

```bash
pip install torch torchvision compressai tensorboard thop timm einops PyWavelets pytorch-msssim pillow numpy
```

### Datasets

Download [OpenImages](https://github.com/openimages) for training, and [Kodak](https://r0k.us/graphics/kodak/), [CLIC Professional Validation](https://www.compression.cc/), and [TESTIMAGES/Tecnick](https://testimages.org/) for evaluation.

The training script follows the `ImageFolder` format used by CompressAI. The dataset directory should contain `train` and `test` folders, for example:

```text
[path-to-dataset]/
  train/
    xxx.png
    ...
  test/
    xxx.png
    ...
```

### Training

The training script is provided in:

```bash
train_spilt_8_wave_transformer_bior44_learnable_continue_train.py
```

We train each rate-distortion point independently using the following three-stage schedule:

- Stage 1: train for 45 epochs with `256 x 256` patches and learning rate `1e-4`.
- Stage 2: continue training to epoch 65 with `256 x 256` patches and learning rate `1e-5`.
- Stage 3: fine-tune for another 5 epochs with `512 x 512` patches and learning rate `1e-5`.

The batch size is set to `8` in all three stages.

#### Stage 1: 256 x 256 training, learning rate 1e-4

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_spilt_8_wave_transformer_bior44_learnable_continue_train.py \
    -d [path-to-training-dataset] \
    --lambda [lambda] \
    --type mse \
    -e 45 \
    -lr 1e-4 \
    --lr_epoch 45 \
    --batch-size 8 \
    --test-batch-size 8 \
    --patch-size 256 256 \
    --save_path [path-to-save-checkpoints] \
    --save \
    --cuda
```

#### Stage 2: continue training to epoch 65, learning rate 1e-5

The checkpoint stores the optimizer and scheduler states. When Stage 1 is trained with `--lr_epoch 45`, the learning rate is reduced to `1e-5` when continuing from the Stage-1 checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_spilt_8_wave_transformer_bior44_learnable_continue_train.py \
    -d [path-to-training-dataset] \
    --lambda [lambda] \
    --type mse \
    -e 65 \
    -lr 1e-5 \
    --lr_epoch 45 \
    --batch-size 8 \
    --test-batch-size 8 \
    --patch-size 256 256 \
    --save_path [path-to-save-checkpoints] \
    --checkpoint [path-to-stage1-checkpoint] \
    --save \
    --cuda
```

#### Stage 3: 512 x 512 fine-tuning, learning rate 1e-5

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_spilt_8_wave_transformer_bior44_learnable_continue_train.py \
    -d [path-to-training-dataset] \
    --lambda [lambda] \
    --type mse \
    -e 70 \
    -lr 1e-5 \
    --lr_epoch 45 \
    --batch-size 8 \
    --test-batch-size 8 \
    --patch-size 512 512 \
    --save_path [path-to-save-checkpoints] \
    --checkpoint [path-to-stage2-checkpoint] \
    --save \
    --cuda
```

For MSE-optimized models, we use the following rate-distortion trade-off parameters:

```bash
--lambda 0.0025
--lambda 0.0035
--lambda 0.0067
--lambda 0.0130
--lambda 0.0250
--lambda 0.0500
```

For MS-SSIM-optimized models, use:

```bash
--type ms-ssim --lambda [3/5/8/16/32/64]
```

### Evaluation

The evaluation script is provided in:

```bash
eval_spilt8_wave_transfomer_bior44_learnable.py
```

#### Estimated rate-distortion evaluation

This mode estimates bpp from the likelihoods produced by the entropy model.

```bash
CUDA_VISIBLE_DEVICES=0 python -u eval_spilt8_wave_transfomer_bior44_learnable.py \
    --checkpoint [path-to-checkpoint] \
    --data [path-to-test-dataset] \
    --cuda
```

#### Actual bitstream evaluation

This mode runs real compression and decompression with entropy coding and reports the actual bitrate from the generated bitstream.

```bash
CUDA_VISIBLE_DEVICES=0 python -u eval_spilt8_wave_transfomer_bior44_learnable.py \
    --checkpoint [path-to-checkpoint] \
    --data [path-to-test-dataset] \
    --cuda \
    --real
```

The script reports the average PSNR, MS-SSIM, bitrate, encoding time, decoding time, and total runtime over the test set.

### Compress images into bitstreams

The model follows the CompressAI-style `compress()` and `decompress()` interface. To compress images into binary streams, use the provided compression script if included in the repository:

```bash
CUDA_VISIBLE_DEVICES=0 python -u compress_and_decompress.py \
    --cuda \
    --data [path-to-images-to-be-compressed] \
    --save_path [path-to-save-bitstreams] \
    --mode compress \
    --checkpoint [path-to-checkpoint]
```

### Decompress images from bitstreams

```bash
CUDA_VISIBLE_DEVICES=0 python -u compress_and_decompress.py \
    --cuda \
    --data [path-to-bitstreams] \
    --save_path [path-to-save-decompressed-images] \
    --mode decompress \
    --checkpoint [path-to-checkpoint]
```


### Pretrained Model

ChWDTA models:

| Lambda | Metric | Link                                                         | Lambda | Metric  | Link                                                         |
| ------ | ------ | ------------------------------------------------------------ | ------ | ------- | ------------------------------------------------------------ |
| 0.0025 | MSE    | [Link](https://drive.google.com/file/d/1F-4OZlh7Q_oVgoGF4Q1ibWV9aezqkfAm/view?usp=drive_link) | 3    | MS-SSIM | [Link](https://drive.google.com/file/d/1jHzlC_h3n0U8qKhmAif_DECSBFoU2c8W/view?usp=sharing) |
| 0.0035 | MSE    | [Link](https://drive.google.com/file/d/1UY9IK-C16574ShmiWwT7aJISB9jQDv-d/view?usp=sharing) | 5   | MS-SSIM | [Link](https://drive.google.com/file/d/1E9hFPojsIjGAbLfHyGvL296OPbGezQPf/view?usp=drive_link) |
| 0.0067 | MSE    | [Link](https://drive.google.com/file/d/1rLnBFZAKHKVSuzSujWz8kw2vkOv_0x8f/view?usp=sharing) | 8   | MS-SSIM | [Link](https://drive.google.com/file/d/1kDh4zWfuxCntrtwDc_HNpQ35k6sEYDpy/view?usp=drive_link) |
| 0.013  | MSE    | [Link](https://drive.google.com/file/d/1xM6c_L_L9FRLpYPa0qiADNR1m8u936VP/view?usp=sharing) | 16  | MS-SSIM | [Link](https://drive.google.com/file/d/1qR_lrnKXP8nbNKnbn4pJNcHpz3FXhVak/view?usp=drive_link) |
| 0.025  | MSE    | [Link](https://drive.google.com/file/d/1F-4OZlh7Q_oVgoGF4Q1ibWV9aezqkfAm/view?usp=drive_link) | 32  | MS-SSIM | [Link](https://drive.google.com/file/d/1TXsqH0mfTWWDobkqPFyyEDL2y1W6tFv9/view?usp=drive_link) |
| 0.05 | MSE    | [Link](https://drive.google.com/file/d/1x1ldrbmy7aGwbmhzMxv9_CbAgqaYjiZh/view?usp=sharing) |  64   | MS-SSIM | [Link](https://drive.google.com/file/d/1nWq6_5RjrBthI8p17x2tjYTKNjCFHrWt/view?usp=drive_link) |

## R-D Data

R-D data on CLIC Pro Valid and Tecnick datasets is in `R-D_Data.md`.


#### ChWDTA, Kodak, PSNR

```
bpp = [0.130,0.171,0.266, 0.400,0.577,0.834];
PSNR = [29.88,30.72,32.36,34.17,36.05,38.15];
```

#### ChWDTA, Kodak, MS-SSIM

```
Kodak_MSSSIM_bpp = [0.123, 0.166, 0.215, 0.307, 0.439, 0.630];
Kodak_MSSSIM_MSSSIM = [14.1808, 15.5179, 16.6469, 18.3947, 20.1416,22.0136];
```

#### ChWDTA, Tecnick, PSNR

```
Tecnick_100_bpp = [0.107,0.133,0.190,0.272,0.387,0.564];
Tecnick_100_PSNR = [32.14,32.89,34.29,35.76,37.24,38.88];
```



#### ChWDTA, CLIC_pro, PSNR
```
CLIC_professional_bpp = [0.100, 0.128, 0.193, 0.208,0.285, 0.413, 0.610];
CLIC_professional_PSNR = [31.78, 32.57, 33.95, 34.26, 35.51, 37.06, 38.85];
```

## Acknowledgement

Part of our code is implemented based on [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [DCAE](https://github.com/CVL-UESTC/DCAE). Thank for the excellent jobs!


## Citation

@misc{fu2026chwdta,
      title={ChWDTA: Channel-wise Wavelet-Domain Transformer Attention and Entropy Modeling for Learned Image Compression}, 
      author={Haisheng Fu and Runyu Yang and Feng Ding and Siyu Zhu and Jie Liang and Xiaoxiao Li and Zhenman Fang and Jingning Han},
      year={2026},
      eprint={2606.00111},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://doi.org/10.48550/arXiv.2606.00111}
}

```

```

## Contact

If you have any questions, please feel free to contact Haisheng.fu@ubc.ca.
