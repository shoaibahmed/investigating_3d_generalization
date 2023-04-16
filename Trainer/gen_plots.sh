#!/bin/bash

srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=0 --cpus-per-task=4 \
    --mem=24G --kill-on-bad-exit --job-name paperclip_plots --nice=100 --time 3-00:00:00 \
    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.10-py3.sqsh \
    --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    python log_parsers/parse_logs_v6.py "paperclips" "normal" && python log_parsers/parse_logs_v6.py "paperclips" "bg_aug" && \
    python log_parsers/parse_logs_v6.py "paperclips" "rot_aug" && python log_parsers/parse_logs_v6.py "chairs" "normal" && \
    python log_parsers/parse_logs_v6_coords.py "images" && python log_parsers/parse_logs_v6_coords.py "array" && \
    python log_parsers/parse_logs_v6_selected_views.py "plot_view_range" && python log_parsers/parse_logs_v6_selected_views.py "plot_num_views"
