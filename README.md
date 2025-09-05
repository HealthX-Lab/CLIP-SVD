# Singular Value Few-shot Adaptation of Vision-Language Models
**[Health-X Lab](http://www.healthx-lab.ca/)** | **[IMPACT Lab](https://users.encs.concordia.ca/~impact/)** 

[Taha Koleilat](https://tahakoleilat.github.io/), [Hassan Rivaz](https://users.encs.concordia.ca/~hrivaz/), [Yiming Xiao](https://yimingxiao.weebly.com/curriculum-vitae.html)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2509.03740)
[![Overview](https://img.shields.io/badge/Overview-Read-blue.svg)](#overview)
[![Datasets](https://img.shields.io/badge/Datasets-Access-yellow.svg)](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp)
[![BibTeX](https://img.shields.io/badge/BibTeX-Cite-blueviolet.svg)](#citation)

## Overview

![main figure](assets/overview.png)
> **<p align="justify"> Abstract:** *Vision-language models (VLMs) like CLIP have shown impressive zero-shot and few-shot learning capabilities across diverse applications. However, adapting these models to new fine-grained domains remains difficult due to reliance on prompt engineering and the high cost of full model fine-tuning. Existing adaptation approaches rely on augmented components, such as prompt tokens and adapter modules, which could limit adaptation quality, destabilize the model, and compromise the rich knowledge learned during pretraining. In this work, we present **CLIP-SVD**, a novel *multi-modal* and *parameter-efficient* adaptation technique that leverages Singular Value Decomposition (SVD) to modify the internal parameter space of CLIP without injecting additional modules. Specifically, we fine-tune only the singular values of the CLIP parameter matrices to rescale the basis vectors for domain adaptation while retaining the pretrained model. This design enables enhanced adaptation performance using only **0.04%** of the model's total parameters and better preservation of its generalization ability. CLIP-SVD achieves state-of-the-art classification results on 11 natural and 10 biomedical datasets, outperforming previous methods in both accuracy and generalization under few-shot settings. Additionally, we leverage a natural language-based approach to analyze the effectiveness and dynamics of the CLIP adaptation to allow interpretability of **CLIP-SVD**.* </p>

## Method

<p float="left">
  <img src="assets/CLIP-SVD.png" width="100%" />
</p>

1) **SVD-Based Few-Shot Adaptation**: We propose an SVD-based adaptation framework for Transformer-based multi-modal models (e.g., CLIP and BiomedCLIP) for the first time, requiring only **0.04%** of the model’s total parameters—significantly lower than other multi-modal methods.  
2) **Comprehensive Validation Across Domains**: We perform extensive evaluation on 11 natural and 10 biomedical datasets, showing that CLIP-SVD outperforms state-of-the-art methods in both accuracy and generalization.  
3) **Interpretable Adaptation Dynamics**: By analyzing ranked weight changes, we employ a natural language-facilitated approach to intuitively interpret the effectiveness and dynamics of task-specific CLIP adaptation.  
4) **Semantic Interpretation for Biomedical Applications**: To address the need for interpretability of attention heads in CLIP for biomedical use cases (e.g., CLIP-SVD analysis), we build the first corpus of biomedical image descriptions.  

## Results
Results reported below show accuracy for few-shot scenarios as well as base and novel classes across 11 natural domain and 10 biomedical domain recognition datasets averaged over 3 seeds.
### Natural Few-shot Evaluation
| **Method**       | K=1   | K=2   | K=4   | K=8   | K=16  |
|------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
| Zero-shot CLIP   |   –   |   –   | 65.36 |   –   |   –   |
| CoOp             | 68.09 | 70.13 | 73.59 | 76.45 | 79.01 |
| CoCoOp           | 66.95 | 67.63 | 71.98 | 72.92 | 75.02 |
| ProGrad          | 68.20 | 71.78 | 74.21 | 77.93 | 79.20 |
| KgCoOp           | 69.51 | 71.57 | 74.48 | 75.82 | 77.26 |
| MaPLe            | 69.27 | 72.58 | 75.37 | 78.89 | 81.79 |
| Linear Probing   | 45.77 | 56.92 | 66.79 | 73.43 | 78.39 |
| LP++             | 70.35 | 72.93 | 75.77 | 77.94 | 80.32 |
| CLIP-Adapter     | 67.87 | 70.20 | 72.65 | 76.92 | 79.86 |
| Tip-Adapter      | 68.89 | 70.42 | 72.69 | 74.41 | 76.44 |
| Tip-Adapter-F    | 70.62 | 73.08 | 75.75 | 78.51 | 81.15 |
| GDA              | 69.39 | 73.09 | 76.24 | 79.71 | 81.70 |
| ProKeR           | 71.32 | 73.74 | 76.23 | 79.84 | 82.01 |
| AdaLoRA          | 69.04 | 72.21 | 75.50 | 78.13 | 80.95 |
| TCP              | 70.63 | 73.59 | 76.07 | 78.39 | 80.98 |
| CLIP-LoRA        | _72.20_ | _75.41_ | _77.32_ | _80.10_ | _82.89_ |
| **CLIP-SVD (Ours)** | **73.20** | **76.06** | **78.18** | **80.55** | **82.97** |

### Biomedical Few-shot Evaluation
| **Method**       | K=1   | K=2   | K=4   | K=8   | K=16  |
|------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
| Zero-shot BiomedCLIP | 42.38 |   –   |   –   |   –   |   –   |
| CoOp             | 52.59 | 55.71 | 61.35 | 67.74 | 71.48 |
| CoCoOp           | 50.88 | 53.91 | 57.63 | 63.15 | 67.51 |
| ProGrad          | 53.67 | 56.42 | 62.10 | 67.06 | 69.21 |
| KgCoOp           | 54.31 | 55.79 | 60.92 | 66.00 | 67.71 |
| Linear Probing   | 48.91 | 55.82 | 62.12 | 67.33 | 70.81 |
| LP++             | 49.27 | 55.88 | 61.30 | 65.48 | 70.09 |
| CLIP-Adapter     | 45.53 | 44.70 | 45.30 | 46.54 | 48.46 |
| Tip-Adapter      | 50.35 | 53.50 | 58.33 | 62.01 | 67.60 |
| Tip-Adapter-F    | 52.55 | 54.17 | 62.30 | 68.12 | 68.12 |
| MaPLe            | 37.99 | 40.89 | 44.09 | 47.37 | 52.93 |
| BiomedCoOp   | **56.87** | _59.32_ | _64.34_ | _68.96_ | _73.41_ |
| **CLIP-SVD (Ours)** | _56.35_ | **62.63** | **68.02** | **73.26** | **76.46** |

### Base-to-Novel Generalization (Natural Domain)  
| Name | Base Acc. | Novel Acc. | HM |  
|--------------------------------------------|:---------:|:----------:|:---------:|  
| CLIP | 69.34 | 74.22 | 71.70 |  
| CoOp | 82.69 | 63.22 | 71.66 |  
| CoCoOp | 80.47 | 71.69 | 75.83 |  
| KgCoOp | 80.73 | 73.60 | 77.00 |  
| ProGrad | 82.48 | 70.75 | 76.16 |  
| MaPLe | 82.28 | 75.14 | 78.55 |  
| IVLP | 84.21 | 71.79 | 77.51 |  
| GDA | 83.96 | 74.53 | 78.72 |  
| TCP | 84.13 | 75.36 | 79.51 |  
| CLIP-LoRA | 84.10 | 74.80 | 79.18 |  
| **CLIP-SVD (ours)** | **84.38** | **76.29** | **80.13** |   

### Base-to-Novel Generalization (Biomedical Domain)  
| Name | Base Acc. | Novel Acc. | HM |  
|------------------------------------------------|:---------:|:----------:|:---------:|  
| BiomedCLIP | 49.27 | 67.17 | 55.23 |  
| CoOp | 76.71 | 65.34 | 68.80 |  
| CoCoOp | 75.52 | 67.74 | 69.11 |  
| KgCoOp | 71.90 | 65.94 | 67.22 |  
| ProGrad | 75.69 | 67.33 | 69.86 |  
| MaPLe | 65.40 | 49.51 | 53.10 |  
| XCoOp | 74.62 | 63.19 | 68.43 |  
| BiomedCoOp | 78.60 | 73.90 | 74.04 |  
| GDA | 57.70 | 64.66 | 60.98 |  
| DCPL | 73.70 | 69.35 | 71.46 |  
| CLIP-LoRA | 70.56 | 59.84 | 64.76 |  
| **CLIP-SVD (ours)** | **82.64** | **74.31** | **78.25** |  

---


## Installation 
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](assets/INSTALL.md). 

## Data preparation
Please follow the instructions at [DATASETS.md](assets/DATASETS.md) to prepare all datasets.

## Training and Evaluation
Please refer to the [RUN.md](assets/RUN.md) for detailed instructions on training, evaluating and reproducing the results using our pre-trained models.

<hr />

## Citation
If you use our work, please consider citing:
```bibtex
@inproceedings{koleilat2025biomedcoop,
  title={Biomedcoop: Learning to prompt for biomedical vision-language models},
  author={Koleilat, Taha and Asgariandehkordi, Hojat and Rivaz, Hassan and Xiao, Yiming},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={14766--14776},
  year={2025}
}
```

## Acknowledgements

Our code builds upon the [CoOp](https://github.com/KaiyangZhou/CoOp), [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), and [LP++](https://github.com/FereshteShakeri/FewShot-CLIP-Strong-Baseline) repositories. We are grateful to the authors for making their code publicly available. If you use our model or code, we kindly request that you also consider citing these foundational works.
