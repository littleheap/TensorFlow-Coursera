#!/bin/bash
# Recommend syntax for setting an infinite while loop
# read -p "Enter GPU number [ 0 - 3 ] " GPU_NUMBER
GPU_NUMBER=$1
while :
do
	source ~/.bash_profile
	kill -9 $(nvidia-smi -g $GPU_NUMBER | awk '$2=="Processes:" {p=1} p && $3 > 0 {print $3}')
	rm -r /scratch/jjylee/temp/compiledir_Linux-3.13--generic-x86_64-with-Ubuntu-14.04-trusty-x86_64-2.7.6-64/lock_dir
	sleep 3s
	CUDA_VISIBLE_DEVICES=$GPU_NUMBER python3.5 main.py
done
