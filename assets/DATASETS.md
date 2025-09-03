# Natural Domain Datasets

We suggest putting all datasets under the same folder (say `data`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
data/
|–– imagenet/
|–– caltech-101/
|–– oxford_pets/
|–– stanford_cars/
```

If you have some datasets already installed somewhere else, you can create symbolic links in `data/dataset_name` that point to the original data to avoid duplicate download.

Datasets list:
- [ImageNet](#imagenet)
- [Caltech101](#caltech101)
- [OxfordPets](#oxfordpets)
- [StanfordCars](#stanfordcars)
- [Flowers102](#flowers102)
- [Food101](#food101)
- [FGVCAircraft](#fgvcaircraft)
- [SUN397](#sun397)
- [DTD](#dtd)
- [EuroSAT](#eurosat)
- [UCF101](#ucf101)
- [ImageNetV2](#imagenetv2)
- [ImageNet-Sketch](#imagenet-sketch)
- [ImageNet-A](#imagenet-a)
- [ImageNet-R](#imagenet-r)

The instructions to prepare each dataset are detailed below. To ensure reproducibility and fair comparison for future work, we provide fixed train/val/test splits for all datasets except ImageNet where the validation set is used as test set. The fixed splits are either from the original datasets (if available) or created by us.

### ImageNet
- Create a folder named `imagenet/` under `data`.
- Create `images/` under `imagenet/`.
- Download the dataset from the [official website](https://image-net.org/index.php) and extract the training and validation sets to `data/imagenet/images`. The directory structure should look like
```
imagenet/
|–– images/
|   |–– train/ # contains 1,000 folders like n01440764, n01443537, etc.
|   |–– val/
```
- If you had downloaded the ImageNet dataset before, you can create symbolic links to map the training and validation sets to `data/imagenet/images`.
- Download the `classnames.txt` to `data/imagenet/` from this [link](https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view?usp=sharing). The class names are copied from [CLIP](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).

### Caltech101
- Create a folder named `caltech-101/` under `data`.
- Download `101_ObjectCategories.tar.gz` from http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz and extract the file under `data/caltech-101`.
- Download `split_zhou_Caltech101.json` from this [link](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and put it under `data/caltech-101`. 

The directory structure should look like
```
caltech-101/
|–– 101_ObjectCategories/
|–– split_zhou_Caltech101.json
```

### OxfordPets
- Create a folder named `oxford_pets/` under `data`.
- Download the images from https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz.
- Download the annotations from https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz.
- Download `split_zhou_OxfordPets.json` from this [link](https://drive.google.com/file/d/1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs/view?usp=sharing). 

The directory structure should look like
```
oxford_pets/
|–– images/
|–– annotations/
|–– split_zhou_OxfordPets.json
```

### StanfordCars
- Create a folder named `stanford_cars/` under `data`.
- Download the train images http://ai.stanford.edu/~jkrause/car196/cars_train.tgz.
- Download the test images http://ai.stanford.edu/~jkrause/car196/cars_test.tgz.
- Download the train labels https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz.
- Download the test labels http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat.
- Download `split_zhou_StanfordCars.json` from this [link](https://drive.google.com/file/d/1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT/view?usp=sharing).

The directory structure should look like
```
stanford_cars/
|–– cars_test\
|–– cars_test_annos_withlabels.mat
|–– cars_train\
|–– devkit\
|–– split_zhou_StanfordCars.json
```

### Flowers102
- Create a folder named `oxford_flowers/` under `data`.
- Download the images and labels from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz and https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat respectively.
- Download `cat_to_name.json` from [here](https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0/view?usp=sharing). 
- Download `split_zhou_OxfordFlowers.json` from [here](https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT/view?usp=sharing).

The directory structure should look like
```
oxford_flowers/
|–– cat_to_name.json
|–– imagelabels.mat
|–– jpg/
|–– split_zhou_OxfordFlowers.json
```

### Food101
- Download the dataset from https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ and extract the file `food-101.tar.gz` under `data`, resulting in a folder named `data/food-101/`.
- Download `split_zhou_Food101.json` from [here](https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl/view?usp=sharing).

The directory structure should look like
```
food-101/
|–– images/
|–– license_agreement.txt
|–– meta/
|–– README.txt
|–– split_zhou_Food101.json
```

### FGVCAircraft
- Download the data from https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.
- Extract `fgvc-aircraft-2013b.tar.gz` and keep only `data/`.
- Move `data/` to `data` and rename the folder to `fgvc_aircraft/`.

The directory structure should look like
```
fgvc_aircraft/
|–– images/
|–– ... # a bunch of .txt files
```

### SUN397
- Create a folder named  `sun397/` under `data`.
- Download the images http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz.
- Download the partitions https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip.
- Extract these files under `data/sun397/`.
- Download `split_zhou_SUN397.json` from this [link](https://drive.google.com/file/d/1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq/view?usp=sharing).

The directory structure should look like
```
sun397/
|–– SUN397/
|–– split_zhou_SUN397.json
|–– ... # a bunch of .txt files
```

### DTD
- Download the dataset from https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz and extract it to `data`. This should lead to `data/dtd/`.
- Download `split_zhou_DescribableTextures.json` from this [link](https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view?usp=sharing).

The directory structure should look like
```
dtd/
|–– images/
|–– imdb/
|–– labels/
|–– split_zhou_DescribableTextures.json
```

### EuroSAT
- Create a folder named `eurosat/` under `data`.
- Download the dataset from http://madm.dfki.de/files/sentinel/EuroSAT.zip and extract it to `data/eurosat/`.
- Download `split_zhou_EuroSAT.json` from [here](https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o/view?usp=sharing).

The directory structure should look like
```
eurosat/
|–– 2750/
|–– split_zhou_EuroSAT.json
```

### UCF101
- Create a folder named `ucf101/` under `data`.
- Download the zip file `UCF-101-midframes.zip` from [here](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing) and extract it to `data/ucf101/`. This zip file contains the extracted middle video frames.
- Download `split_zhou_UCF101.json` from this [link](https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y/view?usp=sharing).

The directory structure should look like
```
ucf101/
|–– UCF-101-midframes/
|–– split_zhou_UCF101.json
```

### ImageNetV2
- Create a folder named `imagenetv2/` under `data`.
- Go to this github repo https://github.com/modestyachts/ImageNetV2.
- Download the matched-frequency dataset from https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz and extract it to `data/imagenetv2/`.
- Copy `data/imagenet/classnames.txt` to `data/imagenetv2/`.

The directory structure should look like
```
imagenetv2/
|–– imagenetv2-matched-frequency-format-val/
|–– classnames.txt
```

### ImageNet-Sketch
- Download the dataset from https://github.com/HaohanWang/ImageNet-Sketch.
- Extract the dataset to `data/imagenet-sketch`.
- Copy `data/imagenet/classnames.txt` to `data/imagenet-sketch/`.

The directory structure should look like
```
imagenet-sketch/
|–– images/ # contains 1,000 folders whose names have the format of n*
|–– classnames.txt
```

### ImageNet-A
- Create a folder named `imagenet-adversarial/` under `data`.
- Download the dataset from https://github.com/hendrycks/natural-adv-examples and extract it to `data/imagenet-adversarial/`.
- Copy `data/imagenet/classnames.txt` to `data/imagenet-adversarial/`.

The directory structure should look like
```
imagenet-adversarial/
|–– imagenet-a/ # contains 200 folders whose names have the format of n*
|–– classnames.txt
```

### ImageNet-R
- Create a folder named `imagenet-rendition/` under `data`.
- Download the dataset from https://github.com/hendrycks/imagenet-r and extract it to `data/imagenet-rendition/`.
- Copy `data/imagenet/classnames.txt` to `data/imagenet-rendition/`.

The directory structure should look like
```
imagenet-rendition/
|–– imagenet-r/ # contains 200 folders whose names have the format of n*
|–– classnames.txt
```
---
# Biomedical Domain Datasets

Our study includes 11 biomedical image classification datasets. Place all the datasets in one directory under `data` to ease management. The file structure looks like

```
data/
|–– BTMRI/
|–– BUSI/
|–– CHMNIST/
|–– COVID_19/
|–– CTKidney/
|–– DermaMNIST/
|–– KneeXray/
|–– Kvasir/
|–– LungColon/
|–– OCTMNIST/
|–– RETINA/
```

## Datasets Description
| **Modality**               | **Organ(s)**      | **Name**                                                                                           | **Classes**                                                                                                       | **# train/val/test** |
|:---------------------------:|:-----------------:|:-------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|:--------------------:|
| Computerized Tomography     | Kidney            | [CTKidney](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)| Kidney Cyst, Kidney Stone, Kidney Tumor, Normal Kidney                                                            | 6221/2487/3738       |
| Dermatoscopy                | Skin              | [DermaMNIST](https://medmnist.com/)                                                                | Actinic Keratosis, Basal Cell Carcinoma, Benign Keratosis, Dermatofibroma, Melanocytic nevus, Melanoma, Vascular Lesion | 7007/1003/2005       |
| Endoscopy                   | Colon             | [Kvasir](https://www.kaggle.com/datasets/abdallahwagih/kvasir-dataset-for-classification-and-segmentation)| Dyed Lifted Polyps, Normal Cecum, Esophagitis, Dyed Resection Margins, Normal Pylorus, Normal Z Line, Polyps, Ulcerative Colitis | 2000/800/1200        |
| Fundus Photography          | Retina            | [RETINA](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)               | Cataract, Diabetic Retinopathy, Glaucoma, Normal Retina                                                           | 2108/841/1268        |
| Histopathology              | Lung, Colon       | [LC25000](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)| Colon Adenocarcinoma, Colon Benign Tissue, Lung Adenocarcinoma, Lung Benign Tissue, Lung Squamous Cell Carcinoma   | 12500/5000/7500      |
| Histopathology              | Colorectal        | [CHMNIST](https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist)                        | Adipose Tissue, Complex Stroma, Debris, Empty Background, Immune Cells, Normal Mucosal Glands, Simple Stroma, Tumor Epithelium | 2496/1000/1504       |
| Magnetic Resonance Imaging  | Brain             | [BTMRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)                  | Glioma Tumor, Meningioma Tumor, Normal Brain, Pituitary Tumor                                                     | 2854/1141/1717       |
| Optical Coherence Tomography| Retina            | [OCTMNIST](https://medmnist.com/)                                                                 | Choroidal Neovascularization, Drusen, Diabetic Macular Edema, Normal                                             | 97477/10832/1000     |
| Ultrasound                  | Breast            | [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)                | Benign Tumors, Malignant Tumors, Normal Scans                                                                    | 389/155/236          |
| X-Ray                       | Chest             | [COVID-QU-Ex](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)          | COVID-19, Lung Opacity, Normal Lungs, Viral Pneumonia                                                             | 10582/4232/6351      |
| X-Ray                       | Knee              | [KneeXray](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity) | No, Doubtful, Minimal, Moderate, and Severe Osteoarthritis                                                       | 5778/826/1656        |


### Download the datasets
All the datasets can be found [here](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/tree/main) on HuggingFace. Download each dataset seperately:

- <b>BTMRI</b> [[Drive](https://drive.google.com/file/d/1_lJLZRUmczqZqoN-dNqkAzGzmi4ONoU5/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/BTMRI.zip)]
- <b>BUSI</b> [[Drive](https://drive.google.com/file/d/1hB5M7wcAUTV9EtiYrijACoQ36R6VmQaa/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/BUSI.zip)]
- <b>CHMNIST</b> [[Drive](https://drive.google.com/file/d/1tyQiYQmqAGNaY4SCK_8U5vEbbaa1AD-g/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/CHMNIST.zip)]
- <b>COVID_19</b> [[Drive](https://drive.google.com/file/d/1zMLN5q5e_tmH-deSZQiY4Xq0M1EqCrML/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/COVID_19.zip)]
- <b>CTKidney</b> [[Drive](https://drive.google.com/file/d/1PBZ299k--mZL8JU7nhC1Wy8yEmlqmVDh/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/CTKidney.zip)]
- <b>DermaMNIST</b> [[Drive](https://drive.google.com/file/d/1Jxd1-DWljunRDZ8fY80dl5zUMefriQXt/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/DermaMNIST.zip)]
- <b>KneeXray</b> [[Drive](https://drive.google.com/file/d/1DBVraYJmxy2UcQ_nGLYvTB2reITOm453/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/KneeXray.zip)]
- <b>Kvasir</b> [[Drive](https://drive.google.com/file/d/1T_cqnNIjmGazNeg6gziarvCNWGsFEkRi/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/Kvasir.zip)]
- <b>LungColon</b> [[Drive](https://drive.google.com/file/d/1YIu5fqMXgyemisiL1L1HCvES2nVpCtun/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/LungColon.zip)]
- <b>OCTMNIST</b> [[Drive](https://drive.google.com/file/d/1mYZNWxbPxnnVvcwHQYybA8gdMzQAoOem/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/OCTMNIST.zip)]
- <b>RETINA</b> [[Drive](https://drive.google.com/file/d/18U-Gc22h5QryomNNzY4r4Qfrq52yf5EO/view?usp=sharing) | [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/RETINA.zip)]

After downloading each dataset, unzip and place each under its respective directory like the following

```
BTMRI/
|–– BTMRI/
|   |–– glioma_tumor/
|   |–– meningioma_tumor/
|   |–– normal_brain/
|   |–– pituitary_tumor/
|–– split_BTMRI.json
```
