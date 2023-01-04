#!/bin/bash -l        
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=30g
#SBATCH --tmp=30g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xu000114@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:1
#SBATCH --output=Train_CMB_inf_SCIF_SF.txt

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_a100

run_task=Train_CMB_inf_SCIF_SF
output_vars=Q_pred_mm
rversion=hs32
n_steps=366
hidden_size=32
epochs=500
patience=50
learning_rate=0.01
sim_type=full_wsl
batch_size=64
input_size=8
dropout=0
# number of RNN layers 
nlayers=1

for run_iter in 0 1 2 3 4
do
    echo $run_iter $run_task $output_vars $rversion $n_steps $hidden_size $epochs $patience $learning_rate $sim_type $batch_size $input_size $dropout $nlayers
    python3 Train_CMB_inf_SCIF_demo.py $run_iter $run_task $output_vars $rversion $n_steps $hidden_size $epochs $patience $learning_rate $sim_type $batch_size $input_size $dropout $nlayers
    # python3 -m pdb Train_CMB_inf_SCIF_demo.py $run_iter $run_task $output_vars $rversion $n_steps $hidden_size $epochs $patience $learning_rate $sim_type $batch_size $input_size $dropout $nlayers
done
