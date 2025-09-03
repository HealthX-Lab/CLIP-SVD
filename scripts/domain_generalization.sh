OUTPUT=$1
SHOTS=16

python main.py --root_path data --dataset imagenet --tasks 3 \
                --shots ${SHOTS} \
                --output_dir ${OUTPUT} \
                --save_path ${OUTPUT} \
                --config configs/domain_generalization/imagenet.yaml              

for DATASET in imagenet_sketch imagenet_a imagenet_r imagenetv2
do
python main.py --root_path data --dataset ${DATASET} --tasks 3 \
                --shots ${SHOTS}  \
                --output_dir ${OUTPUT} \
                --save_path ${OUTPUT}/imagenet \
                --config configs/domain_generalization/${DATASET}.yaml \
                --eval_only  
done