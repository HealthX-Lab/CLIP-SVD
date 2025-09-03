import torch
import torch.nn.functional as F
import torch.nn as nn

from utils import *

from svf_utils.utils import set_clip_trainable_parameters, set_biomedclip_trainable_parameters, \
                            print_trainable_parameters, save_model, load_model
from svf_utils.svf_torch import resolver
from svf_utils.PlainHeadAttention import PlainMultiHeadAttention
from svf_utils.BiomedCLIPHeadAttention import BiomedCLIPMultiHeadAttention
from open_clip import create_model_from_pretrained, get_tokenizer
import copy


def evaluate_model(args, clip_model, logit_scale, loader, dataset, mc_samples=3):
    
    embeddings = []
    targets = []
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        if(args.model == "CLIP"):
            texts = clip.tokenize(texts).cuda()
        elif(args.model == "BiomedCLIP"):
            tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            texts = tokenizer(texts).cuda()

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            class_embeddings = clip_model.encode_text(texts)
            text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()

            image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)

            embeddings.append(image_features)
            targets.append(target)

            cosine_similarity = logit_scale * image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)

    acc /= tot_samples

    return acc


def run_training(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    VALIDATION = False
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    if(args.model == "CLIP"):
        textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)
    elif(args.model == "BiomedCLIP"):
        textual_features = biomedclip_classifier(dataset.classnames, dataset.template, clip_model)
    
    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
 
    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    clip_model = clip_model.float().cuda()

    if(args.model == "CLIP"):
        for i,module in enumerate(clip_model.visual.transformer.resblocks):
            new_module = PlainMultiHeadAttention()
            new_module.set_parameters(module.attn)
            module.attn = new_module

        for i,module in enumerate(clip_model.transformer.resblocks):
            new_module = PlainMultiHeadAttention(embed_dim=512, num_heads=8)
            new_module.set_parameters(module.attn)
            module.attn = new_module

    elif(args.model == "BiomedCLIP"):
        for i,module in enumerate(clip_model.visual.trunk.blocks):
            new_module = BiomedCLIPMultiHeadAttention()
            new_module.set_parameters(module.attn)
            module.attn = new_module

    clip_model = resolver(clip_model.cuda(), global_low_rank_ratio=args.rank)
    clip_model = clip_model.cuda() 

    if args.eval_only:
        load_model(args, clip_model)
        acc_test = evaluate_model(args, clip_model, logit_scale, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return acc_test

    if args.model == "CLIP":
        set_clip_trainable_parameters(args, clip_model)
    elif args.model == "BiomedCLIP":
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        set_biomedclip_trainable_parameters(args, clip_model)

    print_trainable_parameters(clip_model)

    if(args.model == "CLIP"):
        total_iters = args.n_iters * args.shots
    elif args.model == "BiomedCLIP":
        total_iters = args.n_iters * min(args.shots,2)

    optimizer = torch.optim.AdamW(clip_model.parameters(),
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999),
                                  lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           total_iters,
                                                           eta_min=1e-6)

    count_iters = 0
    finish = False
    while count_iters < total_iters:
        clip_model.train()
        acc_train, tot_samples, loss_epoch = 0, 0, 0.

        for i, (images, target) in enumerate(tqdm(train_loader)):
            images, target = images.cuda(), target.cuda()

            template = dataset.template
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            if args.model == "CLIP":
                texts = clip.tokenize(texts).cuda()
            elif args.model == "BiomedCLIP":
                texts = tokenizer(texts).cuda()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                cosine_similarity = logit_scale * image_features @ text_features.t()

                loss = F.cross_entropy(cosine_similarity, target)

            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            count_iters += 1
            
            if count_iters == total_iters:
                break
            
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('Iter {}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters+1, current_lr, acc_train, loss_epoch))


        
        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_model(args, clip_model, logit_scale, val_loader, dataset)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
        
    acc_test = evaluate_model(args, clip_model, logit_scale, test_loader, dataset)
    
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    
    if args.save_path != None:
        save_model(args, clip_model)
    
    return acc_test

def run_biomedical_training(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    

    clip_model = clip_model.float().cuda()

    for i,module in enumerate(clip_model.visual.trunk.blocks):
        new_module = BiomedCLIPMultiHeadAttention()
        new_module.set_parameters(module.attn)
        module.attn = new_module

    clip_model = resolver(clip_model.cuda(), global_low_rank_ratio=args.rank)
    clip_model = clip_model.cuda() 

    if args.eval_only:
        load_model(args, clip_model)
        acc_test = evaluate_model(args, clip_model, logit_scale, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return acc_test

    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    set_biomedclip_trainable_parameters(args, clip_model)

    print_trainable_parameters(clip_model)
    total_epochs = args.train_epochs

    optimizer = torch.optim.AdamW(clip_model.parameters(),
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999),
                                  lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           total_epochs,
                                                           eta_min=1e-6)

    for train_idx in range(total_epochs):
        clip_model.train()
        acc_train, tot_samples, loss_epoch = 0, 0, 0.

        for i, (images, target) in enumerate(tqdm(train_loader)):
            images, target = images.cuda(), target.cuda()

            template = dataset.template
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            texts = tokenizer(texts).cuda()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                cosine_similarity = logit_scale * image_features @ text_features.t()

                loss = F.cross_entropy(cosine_similarity, target)

            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        acc_train /= tot_samples
        loss_epoch /= tot_samples
        current_lr = scheduler.get_last_lr()[0]
        print('Epoch {}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(
            train_idx + 1, current_lr, acc_train, loss_epoch))

        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_model(args, clip_model, logit_scale, val_loader, dataset)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
        
    acc_test = evaluate_model(args, clip_model, logit_scale, test_loader, dataset)
    
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    
    if args.save_path != None:
        save_model(args, clip_model)
    
    return acc_test

def run_other_training(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):

    clip_model = clip_model.float().cuda()

    if(args.model == "CLIP"):
        for i,module in enumerate(clip_model.visual.transformer.resblocks):
            new_module = PlainMultiHeadAttention()
            new_module.set_parameters(module.attn)
            module.attn = new_module

        for i,module in enumerate(clip_model.transformer.resblocks):
            new_module = PlainMultiHeadAttention(embed_dim=512, num_heads=8)
            new_module.set_parameters(module.attn)
            module.attn = new_module

    elif(args.model == "BiomedCLIP"):
        for i,module in enumerate(clip_model.visual.trunk.blocks):
            new_module = BiomedCLIPMultiHeadAttention()
            new_module.set_parameters(module.attn)
            module.attn = new_module

    clip_model = resolver(clip_model.cuda(), global_low_rank_ratio=args.rank)
    clip_model = clip_model.cuda() 


    if args.eval_only:
        load_model(args, clip_model)
        acc_test = evaluate_model(args, clip_model, logit_scale, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return acc_test

    if args.model == "CLIP":
        set_clip_trainable_parameters(args, clip_model)
    elif args.model == "BiomedCLIP":
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        set_biomedclip_trainable_parameters(args, clip_model)

    print_trainable_parameters(clip_model)
    total_epochs = args.train_epochs

    optimizer = torch.optim.AdamW(clip_model.parameters(),
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999),
                                  lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           total_epochs,
                                                           eta_min=1e-6)


    for train_idx in range(total_epochs):
        clip_model.train()
        acc_train, tot_samples, loss_epoch = 0, 0, 0.

        for i, (images, target) in enumerate(tqdm(train_loader)):
            images, target = images.cuda(), target.cuda()

            template = dataset.template
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            if args.model == "CLIP":
                texts = clip.tokenize(texts).cuda()
            elif args.model == "BiomedCLIP":
                texts = tokenizer(texts).cuda()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                cosine_similarity = logit_scale * image_features @ text_features.t()

                loss = F.cross_entropy(cosine_similarity, target)

            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        acc_train /= tot_samples
        loss_epoch /= tot_samples
        current_lr = scheduler.get_last_lr()[0]
        print('Epoch {}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(
            train_idx + 1, current_lr, acc_train, loss_epoch))

    acc_test = evaluate_model(args, clip_model, logit_scale, test_loader, dataset)
    
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    
    if args.save_path != None:
        save_model(args, clip_model)
    
    return acc_test