#!/bin/bash

models=("DCNN" "UNet" "SwinT2UNet")
orographies=("false" "true")
add_vars=("si10" "si10,t2m" "si10,t2m,sh2")
years=("2018" "2018,2019" "2018,2020")
n_stations=("50" "75" "100")

for model in "${models[@]}"; do
  for yrs in "${years[@]}"; do
          sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ddp_train
#SBATCH --output=slurmout/ddp_train-%j.out
#SBATCH --error=slurmout/ddp_train-%j.err
#SBATCH --time=02-00:00:00
#SBATCH --mem=160gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --exclude=dgx02
# #SBATCH --container-image='docker://nvcr.io/nvidia/physicsnemo/physicsnemo:25.03'
#SBATCH --container-image='/network/rit/dgx/dgx_basulab/enroot_tmp/physicsnemo:25.03.sqsh'
#SBATCH --container-mounts=/network/rit/dgx/dgx_basulab/Harish:/mnt/dgx_basulab/Harish,/network/rit/lab/basulab/Harish:/mnt/basulab/Harish,/network/rit/home/hb533188:/mnt/home/hb533188,/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables:/mnt/current_project
#SBATCH --container-workdir=/mnt/current_project

export nproc_per_node=\${SLURM_NTASKS_PER_NODE}
export num_workers=\${SLURM_CPUS_PER_TASK}
export MASTER_PORT=\$((20000 + RANDOM % 20000))

torchrun --nproc_per_node=\$nproc_per_node --master_port=\$MASTER_PORT train.py \
  --checkpoint_dir checkpoints \
  --variable i10fg \
  --model $model \
  --orography_as_channel 'true' \
  --additional_input_variables 'none' \
  --train_years_range $yrs \
  --global_seed 42 \
  --n_random_stations none \
  --loss MaskedCharbonnierLoss \
  --transform standard \
  --epochs 120 \
  --batch_size 16 \
  --num_workers \$num_workers \
  --wandb_id none
EOF
  done
done