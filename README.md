# ChWDTA-Learned-image-compression

# ChWDTA: Channel-wise Wavelet-Structured Transformer and Entropy Modeling for Learned Image Compression



## Introduction

This repository is the official PyTorch implementation of the paper *"ChWDTA: Channel-wise Wavelet-Structured Transformer and Entropy Modeling for Learned Image Compression"*.

**Abstract:** Recent learned image compression methods have achieved strong rate-distortion performance by combining convolutional networks with Transformer-based spatial attention and expressive entropy models. However, most existing hybrid codecs apply attention and entropy modeling directly in the native channel coordinates, leaving the statistical structure along the channel dimension insufficiently exploited. In this paper, we propose ChWDTA, a channel-wise wavelet-structured learned image compression framework. The core component is a Channel-wise Wavelet-Decorrelated Transformer Block (ChWDTB), which wraps windowed spatial self-attention with an invertible lifting-based wavelet transform along the channel axis. This design keeps the efficient spatial-token attention pattern while computing attention from a structured channel representation. For entropy coding, we further introduce a Channel-wise Wavelet Packet (ChWP) transform to reorganize the latent representation into subbands that are better aligned with slice-based entropy modeling. The proposed wavelet transforms are initialized from the CDF 9/7 lifting structure and can be learned end-to-end while preserving exact invertibility. Experiments on Kodak, CLIC Professional Validation, and Tecnick show that ChWDTA achieves state-of-the-art rate-distortion performance with competitive computational complexity.

## Highlights

Our **Ch**annel-wise **W**avelet-**D**ecorrelated **T**ransformer **A**ttention (**ChWDTA**) framework improves learned image compression with the following key designs:

- Channel-wise wavelet-structured Transformer attention via an invertible lifting-based wavelet wrapper.
- Slice-aligned Channel-wise Wavelet Packet entropy coding for more effective latent distribution modeling.
- End-to-end learnable lifting wavelets initialized from the CDF 9/7 transform while preserving perfect reconstruction.
- Consistent integration of channel-wise wavelet modeling into the main backbone, hyperprior pathway, and entropy model.

Our method achieves strong compression performance on Kodak (-17.82%), CLIC Professional Validation (-19.15%), and Tecnick (-22.56%) over VVC VTM-9.1.

## Performance

The following table reports the rate-distortion performance and complexity comparison with representative learned image compression methods.

<img src="./assets/table1.png" alt="table1" style="zoom:25%;" />

The rate-distortion curves on Kodak, CLIC Professional Validation, and Tecnick are provided below.

<img src="./assets/rd_curves.png" alt="rd_curves" style="zoom:25%;" />

## Installation

This implementation requires Python 3.8 and PyTorch. We recommend creating a clean conda environment before installation.

1. Create and activate the environment

   ```bash
   conda create -n chwdta python=3.8
   conda activate chwdta

## Pretrained Model

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
bpp = [0.0951, 0.1537, 0.2438, 0.3778, 0.5516, 0.7843]
psnr = [28.9135, 30.4490, 32.1219, 33.9923, 35.8511, 37.7480]
```

#### ChWDTA, Kodak, MS-SSIM

```
bpp = [0.0943, 0.1429, 0.2090, 0.2935, 0.4016, 0.5577]
db_msssim = [13.2098, 14.7740, 16.6119, 18.2272, 19.7947, 21.4393]
```

## Acknowledgement

Part of our code is implemented based on [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [DCVC-DC](https://github.com/microsoft/DCVC/tree/main/DCVC-family/DCVC-DC). Thank for the excellent jobs!


## Citation

```

```

## Contact

If you have any questions, please feel free to contact Haisheng.fu@ubc.ca.
