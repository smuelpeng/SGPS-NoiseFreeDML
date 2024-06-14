export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

python launch.py --config configs/textmass_vitb32.yaml --train --gpu 0
