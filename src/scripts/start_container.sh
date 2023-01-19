srun --container-image=nvcr.io#nvidia/pytorch:22.02-py3 -N1 -q a100 -c 30 --gres=gpu:a100:1 --mem=100GB --pty bash
