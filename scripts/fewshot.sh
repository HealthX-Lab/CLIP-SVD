DATASET=$1
SHOTS=$2
OUTPUT=$3

python main.py --root_path data --dataset ${DATASET} --tasks 3 --shots ${SHOTS} \
                --output_dir ${OUTPUT} --config configs/few_shot/${DATASET}.yaml 