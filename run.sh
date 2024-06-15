export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=4

currenttime=`date "+%Y%m%d_%H%M%S"`
mkdir -p $1/server_log
mkdir -p $1/train_log

#nohup python tools/setup_server.py --config-file configs/NF.yaml > $1/server_log/${currenttime}.log &
python launch.py --config configs/cars96_sgps.yaml --train --gpu 4
