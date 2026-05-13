#!/bin/bash
#SBATCH --job-name=emg_train_v4
#SBATCH --output=emg_train_v4_%j.out
#SBATCH --error=emg_train_v4_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

module load Python3/3.10.14

export CUDA_VISIBLE_DEVICES=""

cd /nfshomes/adnan16/emg_project/files

echo "=== EMG Training v4 ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

python3 train_v4.py \
    --gt_csv ground_truth_labeled.csv \
    --emg_dir ./set_1 \
    --quick

echo "=== Done: $(date) ==="
