SHOTS=16

for DATASET in eurosat fgvc_aircraft stanford_cars sun397 ucf101 oxford_flowers oxford_pets food101 dtd caltech101 imagenet
do
python main.py --root_path data --dataset ${DATASET} --tasks 3 \
                --shots ${SHOTS} --subsample base \
                --output_dir outputs_base2new_natural_reproduce \
                --save_path outputs_base2new_natural_reproduce \
                --config configs/base2new/${DATASET}.yaml

python main.py --root_path data --dataset ${DATASET} --tasks 3 \
                --shots ${SHOTS} --subsample new \
                --output_dir outputs_base2new_natural_reproduce \
                --save_path outputs_base2new_natural_reproduce/${DATASET} \
                --config configs/base2new/${DATASET}.yaml \
                --eval_only                       
done

for DATASET in btmri covid ctkidney kvasir kneexray retina octmnist chmnist lungcolon
do
python main.py --root_path data --dataset ${DATASET} --tasks 3 \
                --shots ${SHOTS} --subsample base \
                --output_dir outputs_base2new_biomedical_reproduce \
                --save_path outputs_base2new_biomedical_reproduce \
                --config configs/base2new/${DATASET}.yaml 

python main.py --root_path data --dataset ${DATASET} --tasks 3 \
                --shots ${SHOTS} --subsample new \
                --output_dir outputs_base2new_biomedical_reproduce \
                --save_path outputs_base2new_biomedical_reproduce \
                --config configs/base2new/${DATASET}.yaml \
                --eval_only   
done