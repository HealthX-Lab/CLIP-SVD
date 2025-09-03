for DATASET in eurosat fgvc_aircraft stanford_cars sun397 ucf101 imagenet oxford_flowers oxford_pets food101 dtd caltech101
do
for SHOTS in 1 2 4 8 16
do
python main.py --root_path data --dataset ${DATASET} --tasks 3 --shots ${SHOTS} \
                --output_dir outputs_fewshot --config configs/few_shot/${DATASET}.yaml 
done
done

for DATASET in busi btmri covid ctkidney chmnist kvasir retina octmnist kneexray lungcolon
do
for SHOTS in 1 2 4 8 16
do
python main.py --root_path data --dataset ${DATASET} --tasks 3 --shots ${SHOTS} \
                --output_dir outputs_fewshot --config configs/few_shot/${DATASET}.yaml 
done
done