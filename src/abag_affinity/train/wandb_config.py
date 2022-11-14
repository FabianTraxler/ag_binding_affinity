import wandb
from argparse import Namespace
from typing import Tuple


def configure(args: Namespace) -> Tuple:
    """ Configure Weights&Bias with the respective parameters and project

    Returns:
        Tuple: Wandb class, wdb config, indicator if wdb is used, offline run, online run
    """
    if args.use_wandb:
        run = wandb.init(project="abag_binding_affinity", mode=args.wandb_mode)
        if args.wandb_mode == "online":
            wandb.run.name = args.wandb_name
            run_id = "fabian22/abag_binding_affinity/{}".format(run.id)
            api = wandb.Api()
            api.run(run_id)

    elif args.sweep_id is not None:
        run = wandb.init()
    else:
        run = wandb.init(project="abag_binding_affinity", mode="disabled")


    config = wandb.config
    for parameter in ["batch_size", "max_epochs", "learning_rate", "patience", "max_num_nodes", "node_type", "num_workers", "validation_set"]:
        config[parameter] = args.__dict__[parameter]

    return wandb, config, args.use_wandb, run
