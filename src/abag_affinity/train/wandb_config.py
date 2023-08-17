import wandb
from argparse import Namespace
from typing import Tuple
import os


def configure(args: Namespace) -> Tuple:
    """ Configure Weights&Bias with the respective parameters and project

    Returns:
        Tuple: Wandb class, wdb config, indicator if wdb is used, offline run, online run
    """
    use_wandb = False
    if args.use_wandb:
        run = wandb.init(project="abag_binding_affinity", mode=args.wandb_mode)
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

    return wandb, wandb.config, use_wandb, run
