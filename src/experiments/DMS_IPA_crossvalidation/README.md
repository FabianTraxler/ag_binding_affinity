Here, we test how DMS dataset training helps prediction/validation on other DMS datasets

We test first all DMS combinations on a single configuration (L2 absolute loss)

We tested
- L2
- relative_L2

Both of which led to some positive correlation in some of the CVs (not sure if data leakage?). The signal is "strong enough to continue".

relative sweep: (crashed upon calculating benchmarks in the end)
https://wandb.ai/dachdiffusion/abag_binding_affinity?workspace=user-dachdiffusion

absolute sweeps: (most crashed)
- https://wandb.ai/dachdiffusion/abag_binding_affinity/sweeps/909mtmkn?workspace=user-dachdiffusion
- https://wandb.ai/dachdiffusion/abag_binding_affinity/sweeps/vora8zhg?workspace=user-dachdiffusion

Next up: test a single combination with an extensive configuration.
