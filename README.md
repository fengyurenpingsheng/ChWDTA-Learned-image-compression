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
| 0.0025 | MSE    | [Link](https://drive.google.com/file/d/1E1DUaPsIrfNPwfk4qD-630hhxx5n_BJ4/view?usp=drive_link) | 3    | MS-SSIM | [Link](https://drive.google.com/file/d/1RUM2a1wdI8Yj9-tvzO_MnHGZWZRp2-W6/view?usp=drive_link) |
| 0.0035 | MSE    | [Link](https://drive.google.com/file/d/15yDUVvEBn-7dMA9SBIQ2w28LJXBGntQo/view?usp=drive_link) | 5   | MS-SSIM | [Link](https://drive.google.com/file/d/1TL_QDlfzHvmerN1p0rn5mJbSNwn3LXXx/view?usp=drive_link) |
| 0.0067 | MSE    | [Link](https://drive.google.com/file/d/1yzZKji6RpsyQPD6KFr_weavVrlmn-V4R/view?usp=drive_link) | 8   | MS-SSIM | [Link](https://drive.google.com/file/d/1nIEJY9ecr9uA9XidtiQRXQ2rzm1DWKM0/view?usp=drive_link) |
| 0.013  | MSE    | [Link](https://drive.google.com/file/d/1L19zjwOpbbFPw0FxnyVLcHATxCaorjUV/view?usp=drive_link) | 16  | MS-SSIM | [Link](https://drive.google.com/file/d/1sKnWry4LIZPawwv08TH3l_41giUuElCx/view?usp=drive_link) |
| 0.025  | MSE    | [Link](https://drive.google.com/file/d/1oh8OwCLc8PEVMW1fc9LoC7G4385kHU5D/view?usp=drive_link) | 32  | MS-SSIM | [Link](https://drive.google.com/file/d/1rR0vFbQ2fOT7EgJbYg5f0OdiIT5jbPPu/view?usp=drive_link) |
| 0.05 | MSE    | [Link](https://drive.google.com/file/d/1VWLPQeDzBZgb1D2mZ9jLzLppXL8gUanH/view?usp=drive_link) |  64   | MS-SSIM | [Link](https://drive.google.com/file/d/1ITR5JEzLjmdHLp20GYzIdwE8eEK2d7ns/view?usp=drive_link) |

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
