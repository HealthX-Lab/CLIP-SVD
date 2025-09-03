OUTPUT=$1
SHOTS=16

python main.py --root_path data --dataset imagenet --tasks 3 \
                --shots ${SHOTS} \
                --output_dir ${OUTPUT} \
                --save_path ${OUTPUT} \
                --config configs/cross_dataset/imagenet.yaml    

for DATASET in fgvc_aircraft stanford_cars sun397 ucf101 oxford_flowers oxford_pets food101 dtd caltech101
do
python main.py --root_path data --dataset ${DATASET} --tasks 3 \
                --shots ${SHOTS}  \
                --output_dir ${OUTPUT} \
                --save_path ${OUTPUT}/imagenet \
                --config configs/cross_dataset/${DATASET}.yaml \
                --eval_only  
done