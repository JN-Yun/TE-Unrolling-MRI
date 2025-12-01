num_unroll_list=(10)     
acc_rate_list=(4)         
dataset_list=(PD)        
domain=(ksp)              
models=(ResNet_time_emb)   
Unroll_algos=(VAMP)       

for model in "${models[@]}"; do
    for num_unroll in "${num_unroll_list[@]}"; do
        for acc_rate in "${acc_rate_list[@]}"; do
            for dataset in "${dataset_list[@]}"; do
                for Unroll_algo in "${Unroll_algos[@]}"; do
                    dir=results/${model}_${Unroll_algo}_${dataset}_${num_unroll}unrolls_R${acc_rate}

                    python test.py \
                        --out_path "${dir}" \
                        --model "${model}" \
                        --data_type "${dataset}" \
                        --acc_rate "${acc_rate}" \
                        --nb_unroll_blocks "${num_unroll}" \
                        --Unroll_algo "${Unroll_algo}" \
                        --weight_epoch 100

                done
            done
        done
    done
done


