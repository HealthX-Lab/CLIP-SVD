DATASET=$1
OUTPUT=$2
SHOTS=16

python main.py --root_path data --dataset ${DATASET} --tasks 3 \
                --shots ${SHOTS} --subsample base \
                --output_dir ${OUTPUT} \
                --save_path ${OUTPUT} \
                --config configs/base2new/${DATASET}.yaml

python main.py --root_path data --dataset ${DATASET} --tasks 3 \
                --shots ${SHOTS} --subsample new \
                --output_dir ${OUTPUT} \
                --save_path ${OUTPUT}/${DATASET} \
                --config configs/base2new/${DATASET}.yaml \
                --eval_only     