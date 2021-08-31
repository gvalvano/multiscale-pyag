# Flag for CUDA_VISIBLE_DEVICE:
CUDA_VD=0

dpath='../DATA/ACDC'
res_dir='.'
dset_name='acdc'
EPOCHS=2

reg_weight=1.0

for split in 'split0'
    do for perc in 'perc100'
    do

        # ------------------------------------------------------------------
        for run_id_and_path in \
            'UNetPyAG exp_unet_pyag'
            do

            # shellcheck disable=SC2086
            set -- ${run_id_and_path}
            run_id=$1
            path=$2
            warm_up_time=0

            r_id="${run_id}"_${perc}_${split}
            echo "${r_id}"
            python -m train --RUN_ID="${r_id}" --n_epochs=${EPOCHS} --CUDA_VISIBLE_DEVICE=${CUDA_VD} --data_path=${dpath} \
                            --experiment="${path}" --warm_up_period="${warm_up_time}"\
                            --dataset_name=${dset_name} --notify=n --verbose=y --n_sup_vols=${perc} \
                            --split_number=${split} --results_dir=${res_dir}

            python -m test --RUN_ID="${r_id}" --n_epochs=${EPOCHS} --CUDA_VISIBLE_DEVICE=${CUDA_VD} --data_path=${dpath} \
                            --experiment="${path}" --warm_up_period="${warm_up_time}"\
                            --dataset_name=${dset_name} --notify=n --verbose=y --n_sup_vols=${perc} \
                            --split_number=${split} --results_dir=${res_dir} \
                            --w_regularisation=${reg_weight}
            done

        done
    done
