# Training and Evaluation

Below we provide training and evaluation instructions for CLIP-SVD. The same instructions applies for all other techniques.


### Training Compute
We train CLIP-SVD on each dataset with a batch size of 32 using a **single** NVIDIA A100 GPU.

## CLIP-SVD

#### (1) Few-shot evaluation setting

The default training settings are provided in the config files at `configs/few_shot`. All hyper-parameters can be modified using this config file.

Below, we provide instructions to train CLIP-SVD on any dataset. 

```bash

# trains and evaluates in a few-shot setting on all 3 seeds
bash scripts/fewshot.sh <dataset> <nb of shots> <output directory>

# Example on BTMRI using 16 shots
bash scripts/fewshot.sh data btmri 16 outputs_fewshot
```

You can reproduce the few-shot results by running the following:

```bash

bash scripts/fewshot_reproduce.sh
```

You can use the script `read_fewshot_results` to get the combined results in csv format:

```bash

python read_fewshot_results.py --dataset_type biomedical --output_name biomedical_fewshot_results.csv --output_dir outputs_fewshot

python read_fewshot_results.py --dataset_type natural --output_name natural_fewshot_results.csv --output_dir outputs_fewshot
```

#### (2) Base-to-Novel class generalization setting

The default training settings are provided in the config files at `configs/base2new`. All hyper-parameters can be modified using this config file.

```bash

bash scripts/base2new.sh <dataset> <output directory>

# Example on BTMRI
bash scripts/base2new.sh data btmri outputs_base2new
```

You can reproduce the base-to-novel results by running the following:

```bash

bash scripts/base2new_reproduce.sh
```

You can use the script `read_base2new_results` to get the combined results in csv format:

```bash

python read_base2new_results.py --dataset_type biomedical --output_name biomedical_base2new_results.csv --output_dir outputs_base2new

python read_base2new_results.py --dataset_type natural --output_name natural_base2new_results.csv --output_dir outputs_base2new
```

# Interpreting the Singular Values

You can run the following scripts to see interpret the adapted CLIP and BiomedCLIP models:

```bash

python interpret_CLIP.py --load_path <checkpoint path>

python interpret_BiomedCLIP.py --load_path <checkpoint path>
```


#### Acknowledgements
This file for running the methods has been borrowed from [BiomedCoOp's](https://github.com/HealthX-Lab/BiomedCoOp/blob/main/assets/RUN.md) official repository.
