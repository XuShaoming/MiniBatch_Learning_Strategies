# This is the shell script to run SSMB training.
run_task=Train_SSMB_inf_SSIF_SNO
output_vars=SNOmm
rversion=hs32
n_steps=366
hidden_size=32
epochs=5 #500
patience=50
learning_rate=0.01
sim_type=full_wsl
batch_size=64
input_size=7
dropout=0
# number of RNN layers
nlayers=1

for run_iter in 0 #1 2 3 4
do
    echo $run_iter $run_task $output_vars $rversion $n_steps $hidden_size $epochs $patience $learning_rate $sim_type $batch_size $input_size $dropout $nlayers
    python3 Train_SSMB_inf_SSIF_demo.py $run_iter $run_task $output_vars $rversion $n_steps $hidden_size $epochs $patience $learning_rate $sim_type $batch_size $input_size $dropout $nlayers
    # python3 -m pdb Train_SSMB_inf_SSIF_demo.py $run_iter $run_task $output_vars $rversion $n_steps $hidden_size $epochs $patience $learning_rate $sim_type $batch_size $input_size $dropout $nlayers
done
