#!/bin/bash
#SBATCH --job-name=titans-train
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --output=logs/job-%j.log

echo "--- STARTING 'train_titans.sh' ---"

export TRANSFORMERS_NO_TORCHVISION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apptainer run --nv /shared/sifs/latest.sif python -m src.training.train_titans \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir experiments/checkpoints/titans \
    --hf-dataset wikitext \
    --hf-dataset-config wikitext-2-raw-v1 \
    --hf-dataset-split "train[:1%]" \
    --max-length 128 \
    --batch-size 4 \
    --phase1-epochs 1 \
    --phase2-epochs 1 \
    --lr 4e-4 \
    --warmup-steps 50 \
    --chunk-size 64 \
    --n-persistent 16

[ $? -eq 0 ] && echo "--- COMPLETED ---" || echo "--- FAILED ---"
