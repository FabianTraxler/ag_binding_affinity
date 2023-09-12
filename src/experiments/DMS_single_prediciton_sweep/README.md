DMS single prediction sweep
====================================

See also https://github.com/moritzschaefer/guided-protein-diffusion/issues/299 for more details.

The base arguments are defined in 'base_args.txt'.

I used plateau scheduling as the default as it is strictly at least as good as constant. The wandb_user argument needs to be changed as it is currently set to my own wandb account.

Note: This sweep/experiment was executed several times:

- Sweep 1: madan21 relaxed (accidentally) https://wandb.ai/dachdiffusion/abag_binding_affinity/sweeps/ea78mw7d
  - Corr: ~0.4
  - RMSE=~1.9
- Screen 2: single DMS (5 datasets), NOT "per-complex" https://wandb.ai/dachdiffusion/abag_binding_affinity/sweeps/q4hltnmx
  - All datasets “learned”. Pearson corr. between 0.3 and 0.5
  - Big datasets take up to 2 days for training. From log files:
    - Phillips21 (-log(Kd)) is learned "perfectly" (both RMSE and corr)
    - taft22 is learned "good" (correlation ~0.85, RMSE >2)
  - Note: The runs still exist, but their compilation in the sweep page does not work.
- Screen 3: single DMS, per-complex https://wandb.ai/dachdiffusion/abag_binding_affinity/sweeps/pfkhi693
  - TODO compare/understand the impact of the dms-specific output layers!
Screen 4: Add relative loss function to see whether correlation etc. improves https://wandb.ai/dachdiffusion/abag_binding_affinity/sweeps/wulmy999
  - Using 100 epochs should be fair to compare against the 200 epochs previously, because we sum two losses
