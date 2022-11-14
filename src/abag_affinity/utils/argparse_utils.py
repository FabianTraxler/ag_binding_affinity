import sys
from argparse import Namespace, ArgumentParser, Action
from pathlib import Path
import torch

from .config import read_config
from ..dataset.data_loader import AffinityDataset

enforced_node_type = {
    "Binding_DDG": "residue",
    "DeepRefine": "atom"
}


class BooleanOptionalAction(Action):
    """Taken from argparse Python >= 3.9 and added here to suppoert python < 3.9"""
    def __init__(self,
                 option_strings,
                 dest,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):

        _option_strings = []
        for option_string in option_strings:
            _option_strings.append(option_string)

            if option_string.startswith('--'):
                option_string = '--no-' + option_string[2:]
                _option_strings.append(option_string)

        super().__init__(
            option_strings=_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            setattr(namespace, self.dest, not option_string.startswith('--no-'))

    def format_usage(self):
        return ' | '.join(self.option_strings)


def parse_args() -> Namespace:
    """ CLI arguments parsing functionality

    Parse all CLI arguments and set not available to default values

    Returns:
        Namespace: Class with all arguments
    """

    parser = ArgumentParser(description='CLI for using the abag_affinity module')

    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    optional.add_argument("-t", "--train_strategy", type=str, help='The training strategy to use',
                          choices=["bucket_train", "pretrain_model", "model_train", "cross_validation"],
                          default="model_train")
    optional.add_argument("-m", "--pretrained_model", type=str, help='Name of the pretrained model to use for node embeddings',
                          choices=["", "DeepRefine", "Binding_DDG"], default="")
    optional.add_argument("-b", "--batch_size", type=int, help="Batch size used for training", default=1)
    optional.add_argument("-e", "--max_epochs", type=int, help="Max number of training epochs", default=200)
    optional.add_argument("-lr", "--learning_rate", type=float, help="Initial learning rate", default=1e-4)
    optional.add_argument("-p", "--patience", type=int,
                          help="Number of epochs with no improvement until end of training",
                          default=10)
    optional.add_argument("-n", "--node_type", type=str, help="Type of nodes in the graphs", default="residue",
                          choices=["residue", "atom"])
    optional.add_argument("--max_num_nodes", type=int, help="Maximal number of nodes for fixed sized graphs",
                          default=None)
    optional.add_argument("--bucket_size_mode", type=int, help="Mode to determine the size of the training buckets",
                          default="min", choices=["min", "geometric_mean", "double_geometric_mean"])
    optional.add_argument("--interface_distance_cutoff", type=int, help="Max distance of nodes to be regarded as interface",
                          default=5)
    optional.add_argument("--interface_hull_size", type=int, help="Size of the extension from interface to generate interface hull",
                          default=7)
    optional.add_argument("--scale_values", action=BooleanOptionalAction, help="Scale affinity values between 0 and 1",
                          default=False)
    optional.add_argument("--loss_function", type=str, help="Type of Loss Function", default="L1",
                          choices=["L1", "L2"] )
    optional.add_argument("--layer_type", type=str, help="Type of GNN Layer", default="GAT",
                          choices=["GAT", "GCN"] )
    optional.add_argument("--gnn_type", type=str, help="Type of GNN Layer", default="proximity",
                          choices=["proximity", "guided"] )
    optional.add_argument("--max_edge_distance", type=int, help="Maximal distance of proximity edges", default=5)
    optional.add_argument("--num_gnn_layers", type=int, help="Number of GNN Layers", default=3)
    optional.add_argument("--size_halving", action=BooleanOptionalAction,
                          help="Indicator if after every layer the embedding size should be halved", default=False)
    optional.add_argument("--aggregation_method", type=str, help="Type aggregation method to get graph embeddings",
                          default="max",  choices=["max", "sum", "mean", "attention", "fixed_size", "edge"])
    optional.add_argument("--nonlinearity", type=str, help="Type of activation function", default="relu",
                          choices=["relu", "leaky", "gelu"])
    optional.add_argument("--num_fc_layers", type=int, help="Number of FullyConnected Layers in regression head",
                          default=3)
    optional.add_argument("-w", "--num_workers", type=int, help="Number of workers to use for data loading", default=0)
    optional.add_argument("-wdb", "--use_wandb", action=BooleanOptionalAction, help="Use Weight&Bias to log training process",
                          default=False)
    optional.add_argument("--wandb_mode", type=str, help="Mode of Weights&Bias Process", choices=["online", "offline"],
                          default="online")
    optional.add_argument("--wandb_name", type=str, help="Name of the Weight&Bias logs", default="")
    optional.add_argument("--init_sweep", action=BooleanOptionalAction, help="Use Weight&Bias sweep to search hyperparameter space",
                          default=False)
    optional.add_argument("--sweep_config", type=str, help="Path to the configuration file of the sweep",
                          default=(Path(__file__).resolve().parents[2] / "config.yaml").resolve())
    optional.add_argument("--sweep_id", type=str, help="The sweep ID to use for all runs")
    optional.add_argument("--sweep_runs", type=int, help="Number of runs to perform in this sweep instance",
                          default=30)
    optional.add_argument("-v", "--validation_set", type=int, help="Which validation set to use", default=1,
                          choices=[1, 2, 3])
    optional.add_argument("-c", "--config_file", type=str,
                          help="Path to config file for datasets and training strategies",
                          default=(Path(__file__).resolve().parents[2] / "config.yaml").resolve())
    optional.add_argument("--verbose", action=BooleanOptionalAction, help="Print verbose logging statements",
                          default=False)
    optional.add_argument("--preprocess_graph", action=BooleanOptionalAction,
                          help="Compute graphs beforehand to speedup training (especially for DeepRefine",
                          default=False)
    optional.add_argument("--save_graphs", action=BooleanOptionalAction, help="Saves computed graphs to speed up training in later epochs",
                          default=False)
    optional.add_argument("--force_recomputation", action=BooleanOptionalAction,
                          help="Force recomputation of graphs - deletes folder containing processed graphs",
                          default=False)
    optional.add_argument("--shuffle", action=BooleanOptionalAction,
                          help="Shuffle train-dataloader",
                          default=True)
    optional.add_argument("--test", action=BooleanOptionalAction,
                          help="Perform 1 iteration with only 1 sample in train and test sets",
                          default=False)
    optional.add_argument("--cuda", action=BooleanOptionalAction,
                          help="Use cuda resources for training if available",
                          default=True)

    args = parser.parse_args()
    args.config = read_config(args.config_file)
    args.tqdm_output = True  # enable tqdm output

    if args.wandb_name == "":
        args.wandb_name = f'{args.train_strategy}' \
                          f' -m {args.pretrained_model}' \
                          f' -n {args.node_type}'

    if args.test:
        args.max_epochs = 1
        args.shuffle = False

    # check arguments
    if args.pretrained_model in enforced_node_type and args.pretrained_model != enforced_node_type[args.pretrained_model]:
        args.__dict__["node_type"] = enforced_node_type[args.pretrained_model]

    args.__dict__["learning_rate"] = args.__dict__["learning_rate"] * args.__dict__["batch_size"]

    return args


def adapt_batch_size(args: Namespace, model: torch.nn.Module, dataset: AffinityDataset) -> int:
    """ Get maximal batch size to use with available gpu memory

    Args:
        args:
        model:
        dataset:

    Returns:

    """

    #TODO: Implement and use
    total_memory = torch.cuda.get_device_properties(0).total_memory

    model_size = sys.getsizeof(model)

    available_memory = total_memory - model_size

    datapoint_size = sys.getsizeof(dataset.__getitem__(0))


    return 1