import torch
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader

from utils import *
from train import run_training, run_other_training, run_biomedical_training
import os
import pandas as pd
from open_clip import create_model_from_pretrained
import random
import numpy as np

import argparse

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml', help='setting of Few-shot CLIP')
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dtd')
    parser.add_argument('--shots', default=1, type=int)
    parser.add_argument('--rank', default=1.0, type=float)
    parser.add_argument('--subsample', default="all", choices = ["all","base","new"])
    parser.add_argument('--tasks', default=3, type=int)
    parser.add_argument('--n_iters', default=200, type=int)
    parser.add_argument('--save_path', default=None, help='path to save the weights after training, not saved if None')
    parser.add_argument('--output_dir', required=True, default="outputs_ablations", help='path to save the results after training')
    parser.add_argument('--filename', default='base_model', help='file name to save the weights (.pt extension will be added)')
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the model (save_path should not be None)')
    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.config)

    cfg.update({k: v for k, v in vars(args).items()})


    return cfg

def main():

    # Load config file
    args = get_arguments()

    predictions = []
    for i in range(args.tasks):
    
        set_random_seed(i+1)

        args.seed = i+1 # For saving the models

        if(args.model == "CLIP"):
            # CLIP
            clip_model, preprocess = clip.load(args.backbone)
            logit_scale = 100
        
        elif(args.model == "BiomedCLIP"):
            # Load the model and config files from the Hugging Face Hub
            clip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            clip_model = clip_model.cuda()
            logit_scale = clip_model.logit_scale.exp().detach()

        clip_model.eval()
        # Prepare dataset
        print("Preparing dataset.")
            
        dataset = build_dataset(args.dataset, args.root_path, args.shots, args.template, args.subsample)
        
        val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
            
        train_loader = None
        if not args.eval_only:

            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                ])

            train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_transform, is_train=True, shuffle=True, num_workers=8)

        if(args.subsample == "all"):
            if(args.model == "CLIP"):
                test_acc = run_training(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader)
            elif(args.model == "BiomedCLIP"):
                test_acc = run_biomedical_training(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader)
        else:
            test_acc = run_other_training(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader)
        
        predictions.append(test_acc)
    
    tasks_acc, tasks_std = np.mean(predictions), np.std(predictions)
    test_stats = {}
    test_stats['acc'] = tasks_acc
    test_stats['std'] = tasks_std

    print('Total Accuracy and std on {} tasks: {:.4f} , {:.4f}'.format(
        str(args.tasks), tasks_acc, tasks_std))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    csv_path = os.path.join(args.output_dir, args.dataset+".csv")
    write_to_csv(args, csv_path, test_stats)

def write_to_csv(args, path, test_stats):
    
    try:
        res = pd.read_csv(path)
    except:
        res = pd.DataFrame()
    records = res.to_dict('records')
    test_stats['acc'] = round(test_stats['acc'],4)
    test_stats['std'] = round(test_stats['std'],4)
    test_stats['num_shots'] = args.shots
    test_stats['tasks'] = args.tasks
    test_stats['subsample'] = args.subsample

    records.append(test_stats)
    # Save back to dataframe
    df = pd.DataFrame.from_records(records)
    df.to_csv(path, index=False)

if __name__ == '__main__':
    main()