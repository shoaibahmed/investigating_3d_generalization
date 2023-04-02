#!/bin/bash

log_dir=logs_v6_wds
if [[ ! -d ${log_dir} ]]; then
    mkdir ${log_dir}
    echo "Created log directory: "${log_dir}
fi

base_path="/netscratch/siddiqui/Datasets/"
val_tar_base="${base_path}/Paperclips_train_v6/Paperclips_v6_classes_"
base_url="${base_path}/Paperclips_train_v6/Paperclips_v6_classes_"
file_ext=".tar.xz"
output_dir="./output_paperclips_v6_wds"

echo "============= Training models on equidistant views! ============="
for num_classes in 10 100 1000 10000; do
    for num_views in 1 2 3 4 6 12; do
        view_type="equidistant_views_${num_views}"
        train_tar_file="${base_url}${num_classes}_${view_type}${file_ext}"
        val_tar_file="${val_tar_base}${num_classes}_eval${file_ext}"

        for model in "resnet18"; do
            if [[ ${model} == "resnet18" || ${model} == "resnet50" ]]; then
                lr=0.1
            elif [[ ${model} == "vgg11_bn" || ${model} == "vit_b_16" ]]; then
                lr=0.01
            fi
            echo "Model: ${model} / Selected LR: ${lr}"

            job_name=paperclip_v6_wds_cls_${num_classes}_${view_type}_${model}
            srun -p A100-SDS -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --exclude=serv-[3328-3335] \
                --mem=256G --kill-on-bad-exit --job-name ${job_name} --nice=100 --time 10-00:00:00 \
                --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui \
                --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.10-py3.sqsh \
                --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
                    /opt/conda/bin/python trainer.py \
                        --dataset paperclip_wds \
                        --model-name ${model} \
                        --train_tar_file ${train_tar_file} \
                        --val_tar_file ${val_tar_file} \
                        --output-dir ${output_dir}/${model}/ \
                        --optimizer sgd \
                        --optimizer-batch-size 128 \
                        --lr ${lr} \
                        --momentum 0.9 \
                        --wd 1e-4 \
                        --batch-size 128 \
                        --test-batch-size 1024 \
                        --epochs 300 \
                        --lr-scheduler cosine \
                        --lr-steps "" \
                        --lr-decay 0.2 \
                        --num-workers 8 \
                        --seed 1 \
                        --no-eval \
                        --clip-grad 10.0 \
                        --multirot-stride 10 \
                    > ./${log_dir}/${job_name}.log 2>&1 &
            
        done
    done
done

echo "============= Training models on selected views! ============="
for num_classes in 10 100 1000 10000; do
    for selected_views in "-15,15" "-30,30" "-30,0,30" "-30,-10,10,30" "-30,-20,-10,0,10,20,30" "-30,-24,-18,-12,-6,0,6,12,18,24,30" "-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30" "-30,-27,-24,-21,-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18,21,24,27,30"; do
        view_type="selected_views_${selected_views}"
        train_tar_file="${base_url}${num_classes}_${view_type}${file_ext}"
        val_tar_file="${val_tar_base}${num_classes}_eval${file_ext}"

        for model in "resnet18"; do
            if [[ ${model} == "resnet18" || ${model} == "resnet50" ]]; then
                lr=0.1
            elif [[ ${model} == "vgg11_bn" || ${model} == "vit_b_16" ]]; then
                lr=0.01
            fi
            echo "Model: ${model} / Selected LR: ${lr}"

            job_name=paperclip_v6_wds_cls_${num_classes}_${view_type}_${model}
            srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 \
                --mem=64G --kill-on-bad-exit --job-name ${job_name} --nice=100 --time 3-00:00:00 \
                --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui \
                --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.10-py3.sqsh \
                --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
                    /opt/conda/bin/python trainer.py \
                        --dataset paperclip_wds \
                        --model-name ${model} \
                        --train_tar_file ${train_tar_file} \
                        --val_tar_file ${val_tar_file} \
                        --output-dir ${output_dir}/${model}/ \
                        --optimizer sgd \
                        --optimizer-batch-size 128 \
                        --lr ${lr} \
                        --momentum 0.9 \
                        --wd 1e-4 \
                        --batch-size 128 \
                        --test-batch-size 128 \
                        --epochs 300 \
                        --lr-scheduler cosine \
                        --lr-steps "" \
                        --lr-decay 0.2 \
                        --num-workers 8 \
                        --seed 1 \
                        --no-eval \
                        --clip-grad 10.0 \
                        --multirot-stride 10 \
                    > ./${log_dir}/${job_name}.log 2>&1 &
        done
    done
done
