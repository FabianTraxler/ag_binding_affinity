import wandb
from argparse import Namespace

def configure(args: Namespace):
    """ Configure Weights&Bias with the respective parameters and project

    Returns:
        Tuple of Wandb instances
    """
    if args.use_wandb:
        run = wandb.init(project="abab_binding_affinity")
        wandb.run.name = args.wandb_name
        run_id = "fabian22/abab_binding_affinity/{}".format(run.id)
        api = wandb.Api()

        this_run = api.run(run_id)
    else:
        run = wandb.init(project="abag_binding_affinity", mode="disabled")
        this_run = None


    config = wandb.config
    config.batch_size = args.batch_size
    config.max_epochs = args.max_epochs
    config.learning_rate = args.learning_rate
    config.patience = args.patience
    config.max_num_nodes = args.max_num_nodes
    config.node_type = args.node_type
    config.num_workers = args.num_workers
    config.dataset_type = args.data_type
    config.model_type = args.model_type
    config.train_strategy = args.train_strategy
    config.validation_set = args.validation_set

    return wandb, config, args.use_wandb, run, this_run
