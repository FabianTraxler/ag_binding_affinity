import wandb
from argparse import Namespace
from torch import nn
from typing import Tuple, Optional
import os


def configure(args: Namespace, model: Optional[nn.Module] = None) -> Tuple:
    """ Configure Weights&Bias with the respective parameters and project

    Returns:
        Tuple: Wandb class, wdb config, indicator if wdb is used, offline run, online run
    """
    use_wandb = False
    if args.use_wandb:
        run = wandb.init(project="abag_binding_affinity", mode=args.wandb_mode, settings=wandb.Settings(start_method="fork"))
        if args.wandb_mode == "online":
            wandb.run.name = args.wandb_name
            run_id = (args.wandb_user + "/abag_binding_affinity/{}").format(run.id)
            api = wandb.Api()
            api.run(run_id)
        use_wandb = True

    elif args.sweep_id is not None:
        run = wandb.init()
        use_wandb = True
    else:
        run = wandb.init(project="abag_binding_affinity", mode="disabled")

    update_config = {
        parameter: args.__dict__[parameter]
        for parameter in ["batch_size", "max_epochs", "learning_rate", "patience", "max_num_nodes", "node_type", "num_workers", "validation_set"]
    }
    try:
        update_config["slurm_jobid"] = os.environ["SLURM_JOBID"]
    except KeyError:
        pass
    update_config["hostname"] = os.popen("hostname").read().strip()

    wandb.config.update(update_config, allow_val_change=True)

    if model:
        wandb.watch(model, log_freq=1000 / args.batch_size, log="all")

    return wandb, wandb.config, use_wandb, run
