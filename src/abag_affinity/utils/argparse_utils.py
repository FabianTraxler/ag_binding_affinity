import json
import sys
from argparse import Namespace, ArgumentParser, Action, ArgumentTypeError
import re
from pathlib import Path
import logging
import random
from .config import read_config

enforced_node_type = {
    "Binding_DDG": "residue",
    "DeepRefine": "atom",
    "IPA": "residue",
    "Diffusion": "residue"
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


def parse_args(artifical_args=None) -> Namespace:
    """ CLI arguments parsing functionality

    Parse all CLI arguments and set not available to default values

    Returns:
        Namespace: Class with all arguments
    """

    parser = ArgumentParser(description='CLI for using abag_affinity module', fromfile_prefix_chars='@')

    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    # train config arguments
    # datasets
    def validate_dataset_string(value):
        """
        NOTE: we could also add all possible dataset names
        """
        pattern = r'^[A-Za-z0-9-_.]+#(L1|L2|RL2|NLL|relative_L1|relative_L2|relative_RL2|relative_ce|relative_cdf)(-[0-9.]+)?(\+(L1|L2|RL2|NLL|relative_L1|relative_L2|relative_RL2|relative_ce|relative_cdf)(-[0-9.]+)?)*$'
        if not re.match(pattern, value):
            raise ArgumentTypeError(f"Invalid value: {value}. Expected format: DATASET#LOSS1-LAMBDA1+LOSS2-LAMBDA2")
        return value

    parser.add_argument("--target_dataset", type=validate_dataset_string, help='The datasize used for final and patience, Loss function is added after #',
                        default="abag_affinity#L1")
    parser.add_argument("-tld", "--transfer_learning_datasets", type=validate_dataset_string,
                        help='Datasets used for transfer-learning in addition to goal_dataset', default=[], nargs='+')

    optional.add_argument("--relaxed_pdbs", choices=["True", "False", "both"], help="Use the relaxed pdbs for training "
                                                                               "and validation", default="False")
    # -train strategy
    optional.add_argument("-t", "--train_strategy", type=str, help='The training strategy to use',
                          choices=["bucket_train", "train_transferlearnings_validate_target"],
                          default="bucket_train")
    optional.add_argument("--bucket_size_mode", type=str, help="Mode to determine the size of the training buckets",
                          default="geometric_mean", choices=["min", "geometric_mean", "double_geometric_mean"])
    optional.add_argument("-m", "--pretrained_model", type=str,
                          help='Name of the published/pretrained model to use for node embeddings',
                          choices=["", "DeepRefine", "Binding_DDG", "IPA", "Diffusion"], default="")
    optional.add_argument("--warm_up_epochs",
                          help='Fine-tune model components that have been frozen at the start of training (e.g. published/pretrained models or dataset-specific layers). Provide an integer to indicate the number of pre-training epochs',
                          type=int, default=0)
    optional.add_argument('--load_pretrained_weights', action=BooleanOptionalAction, help="Load pretrained weights for the pretrained model", default=True)
    optional.add_argument("--training_set_spikein", type=float,
                          help="Proportion of target_dataset to spike into training set, in mode `train_transferlearnings_validate_target`",
                          default=0.0)
    # -train config
    optional.add_argument("-b", "--batch_size", type=int, help="Batch size used for training", default=1)
    optional.add_argument("-e", "--max_epochs", type=int, help="Max number of training epochs", default=400)
    optional.add_argument("-lr", "--learning_rate", type=float, help="Initial learning rate", default=1e-3)
    optional.add_argument("-p", "--patience", type=int,
                          help="Number of epochs with no improvement until end of training",
                          default=30)  # this needs to be different from None if plateau is the default LR scheduling method
    optional.add_argument("--lr_scheduler", type=str, default="plateau",
                          choices=["constant", "plateau", "exponential"],
                          help="Type of learning rate scheduler",)
    optional.add_argument("--stop_at_learning_rate", type=float, help="Stop training after learning rate" +
                          "goes beneath this value", default=None)
    optional.add_argument("--lr_decay_factor", type=float, help="Factor to decay learning rate", default=0.5)
    # input graph config arguments
    optional.add_argument("-n", "--node_type", type=str, help="Type of nodes in the graphs", default="residue",
                          choices=["residue", "atom"])
    optional.add_argument("--max_num_nodes", type=int, help="Maximal number of nodes for fixed sized graphs",
                          default=None)
    optional.add_argument("--interface_distance_cutoff", type=int, help="Max distance of nodes to be regarded as interface",
                          default=5)
    optional.add_argument("--interface_hull_size", type=lambda x: None if x is None or x.lower() == 'none' else int(x),
                          help="Size of the extension from interface to generate interface hull. Provide None to include whole protein", default=None)
    optional.add_argument("--scale_values", action=BooleanOptionalAction, help="Scale affinity values between 0 and 1",
                          default=True)
    optional.add_argument("--scale_min", type=int, help="The minimal affinity value -> gets mapped to 0",
                          default=5)
    optional.add_argument("--scale_max", type=int, help="The maximal affinity value -> gets mapped to 1",
                          default=14)
    optional.add_argument("--max_edge_distance", type=int, help="Maximal distance of proximity edges", default=5)
    optional.add_argument("--add_neglogkd_labels_dataset", type=str, default=None, help="Include an additional dataset composed of only samples with -log(kd) labels. Only implemented for bucket_learning. Provide a criterion to be used for this dataset")
    # model config arguments
    optional.add_argument("--layer_type", type=str, help="Type of GNN Layer", default="GCN",
                          choices=["GAT", "GCN"] )
    optional.add_argument("--gnn_type", type=str, help="Type of GNN Layer", default="egnn",
                          choices=["proximity", "guided", "identity", "egnn"])
    optional.add_argument("--num_gnn_layers", type=int, help="Number of GNN Layers", default=5)
    optional.add_argument("--attention_heads", type=int, help="Number of attention heads for GAT layer type",
                          default=5)
    optional.add_argument("--channel_halving", action=BooleanOptionalAction,
                          help="Indicator if after every layer the embedding size should be halved", default=True)
    optional.add_argument("--channel_doubling", action=BooleanOptionalAction,
                          help="Indicator if after every layer the embedding size should be doubled", default=False)
    optional.add_argument("--egnn_dim", type=int, help="The number of EGNN nodes",
                          default=64)

    optional.add_argument("--aggregation_method", type=str, help="Type aggregation method to get graph embeddings",
                          default="interface_sum",  choices=["max", "sum", "mean", "attention", "fixed_size", "edge", "interface_sum", "interface_mean","interface_size"])
    optional.add_argument("--nonlinearity", type=str, help="Type of activation function", default="leaky",
                          choices=["relu", "leaky", "gelu", "silu"])
    optional.add_argument("--num_fc_layers", type=int, help="Number of FullyConnected Layers in regression head",
                          default=2)
    optional.add_argument("--fc_size_halving", action=BooleanOptionalAction,
                          help="Indicator if after every FC layer the embedding sizeshould be halved",
                          default=True)
    optional.add_argument("--dms_output_layer_type", choices=["identity", "bias_only", "regression", "regression_sigmoid", "positive_regression","positive_regression_sigmoid","mlp"],
                          help="Architecture of the DMS-specific output layers",
                          default="bias_only")

    # weight and bias arguments
    optional.add_argument("-wdb", "--use_wandb", action=BooleanOptionalAction, help="Use Weight&Bias to log training process",
                          default=False)
    optional.add_argument("--wandb_mode", type=str, help="Mode of Weights&Bias Process", choices=["online", "offline"],
                          default="offline")
    optional.add_argument("--wandb_name", type=str, help="Name of the Weight&Bias logs", default="")
    optional.add_argument("--wandb_user", type=str, help="Name of the Weight&Bias user", default="dachdiffusion")
    optional.add_argument("--model_path", type=str, help="Target filename for model. Default is defined in main.py", default=None)
    optional.add_argument("--init_sweep", action=BooleanOptionalAction,
                          help="Use Weight&Bias sweep to search hyperparameter space", default=False)
    optional.add_argument("--sweep_config", type=str, help="Path to the configuration file of the sweep",
                          default=None)
    optional.add_argument("--sweep_id", type=str, help="The sweep ID to use for all runs")
    optional.add_argument("--sweep_runs", type=int, help="Number of runs to perform in this sweep instance. Default: exhaustive search",
                          default=None)

    # general config
    optional.add_argument("-w", "--num_workers", type=int, help="Number of workers to use for data loading", default=0)
    optional.add_argument("--cross_validation", action=BooleanOptionalAction, help="Perform CV on all validation datasets", default=False)
    optional.add_argument("--number_cv_splits", type=int, help='The number of data splits for cross validation',
                          default=10)
    optional.add_argument("-v", "--validation_set", type=int, help="Which validation set to use", default=0,
                          choices=[0, 1, 2, 3, 4])
    optional.add_argument("-c", "--config_file", type=str,
                          help="Path to config file for datasets and training strategies",
                          default=(Path(__file__).resolve().parents[2] / "config.yaml").resolve())
    optional.add_argument("--verbose", action=BooleanOptionalAction, help="Print verbose logging statements",
                          default=False)
    optional.add_argument("--preprocess_graph", action=BooleanOptionalAction,
                          help="Compute graphs beforehand to speedup training (especially for DeepRefine).",
                          default=True)
    optional.add_argument("--preprocessed_to_scratch", type=str, default=None,
                          help="Provide target path to copy preprocessed files to a scratch space for minimized/optimized cluster network access.")
    optional.add_argument("--save_graphs", action=BooleanOptionalAction,
                          help="Saves computed graphs to speed up training in later epochs", default=True)
    optional.add_argument("--force_recomputation", action=BooleanOptionalAction,
                          help="Force recomputation of graphs - deletes folder containing processed graphs",
                          default=False)  # TODO enable this for safety. too many can changes can happen... disable before sweeps
    optional.add_argument("--shuffle", action=BooleanOptionalAction,
                          help="Shuffle train-dataloader",
                          default=True)
    optional.add_argument("--test", action=BooleanOptionalAction,
                          help="Perform 1 iteration with only 1 sample in train and test sets",
                          default=False)
    optional.add_argument("--cuda", action=BooleanOptionalAction,
                          help="Use cuda resources for training if available",
                          default=True)
    optional.add_argument("--tqdm_output", action=BooleanOptionalAction,
                          help="Use tqdm output to monitor epochs",
                          default=True)
    optional.add_argument("--args_file", type=str,
                          help="Specify the path to a file with additional arguments",
                          default=None)
    optional.add_argument("--embeddings_path", type=str, default=None, help="Path to embeddings file. Requires --embeddings_type to be set.")
    optional.add_argument("--embeddings_type", type=str, default="", choices=["", "rf", "of"], help="Type of embeddings to use.")
    optional.add_argument("--seed", type=int, default=42, help="Seed for random number generator")
    optional.add_argument("--debug", action=BooleanOptionalAction, default=False, help="Start debugger on a free port starting from 5678")
    optional.add_argument("--weight_decay", type=float, default=0, help="Weight Decay for Parameters")
    optional.add_argument("--uncertainty_temp", type=float, default=0.0, help="Uncertainty Temperature for cdf loss")

    args = parser.parse_args(artifical_args)
    args.config = read_config(args.config_file)

    args.relaxed_pdbs = eval(args.relaxed_pdbs.capitalize()) if args.relaxed_pdbs != "both" else "both"

    if args.wandb_name == "":
        args.wandb_name = f'{args.train_strategy}' \
                          f' -m {args.pretrained_model}' \
                          f' -n {args.node_type}'

    if args.test:
        args.max_epochs = 1
        args.shuffle = False

    if args.stop_at_learning_rate is None:
        args.stop_at_learning_rate = args.learning_rate / 100


    if args.args_file is not None:
        args = read_args_from_file(args)

    # Modify args that are incompatible
    args = check_and_complement_args(args, {})

    return args


def read_args_from_file(args: Namespace) -> Namespace:
    with open(args.args_file) as f:
        arg_dict = json.load(f)

    manually_passed_args = [arg[:2].replace("-", "") + arg[2:] for arg in sys.argv if "-" == arg[0]]

    for key, value in arg_dict.items():
        if value["value"] == "None":
            value["value"] = None
        if key in args.__dict__ and key not in manually_passed_args:
            args.__dict__[key] = value["value"]
            if key == "transfer_learning_datasets" and isinstance(value["value"], str):
                if ";" in value["value"]:
                    args.__dict__[key] = value["value"].split(";")
                else:
                    args.__dict__[key] = [value["value"]]
                continue

    return args

def check_and_complement_args(args: Namespace, args_dict: dict) -> Namespace:
    """
    Check args, enforcing specific constraints. Optionally integrate additional args (e.g. from a sweep)

    Args:
        args: Args to be modified
        config: Config to update args

    Returns:
    Cleaned and modified args
    """
    ATOM_NODES_MULTIPLIKATOR = 5

    new_args = Namespace(**vars(args))
    wandb_name = ""
    for param in args_dict.keys():
        if args_dict[param] == "None":
            param_value = None
        else:
            param_value = args_dict[param]
            wandb_name = wandb_name + str(param)[:5] + str(param_value)[:40]

        if param == "max_num_nodes" and param_value is None and "aggregation_method" in args_dict and args_dict[
            "aggregation_method"] == "fixed_size":
            continue  # ignore since it is manually overwritten below
        if param == "node_type" and "pretrained_model" in args_dict and args_dict[
            "pretrained_model"] in enforced_node_type:
            continue  # ignore since it is manually overwritten below
        if param == "transfer_learning_datasets" and isinstance(param_value, str):
            if ";" in param_value:
                new_args.__dict__[param] = param_value.split(";")
            else:
                new_args.__dict__[param] = [param_value]
            continue

        new_args.__dict__[param] = param_value
        if param == "pretrained_model":
            if args_dict[param] in enforced_node_type:
                new_args.__dict__["node_type"] = enforced_node_type[args_dict[param]]
                args_dict["node_type"] = enforced_node_type[args_dict[param]]

        if param == 'aggregation_method':
            if param_value == "fixed_size" and "max_num_nodes" in args_dict and args_dict["max_num_nodes"] == "None":
                max_num_nodes_values = [int(value) for value in
                                        new_args.config["HYPERPARAMETER_SEARCH"]["parameters"]["max_num_nodes"][
                                            "values"] if value != "None"]
                new_args.max_num_nodes = random.choice(max_num_nodes_values)
                args_dict["max_num_nodes"] = new_args.max_num_nodes

    # adapt hyperparameter based on node type
    if new_args.node_type == "atom" and new_args.max_num_nodes is not None:
        new_args.max_num_nodes = int(new_args.max_num_nodes * ATOM_NODES_MULTIPLIKATOR)

    # adapt batch size bases on node type
    if new_args.node_type == "atom":
        new_args.batch_size = int(new_args.batch_size / ATOM_NODES_MULTIPLIKATOR) + 1
    # check arguments
    if args.pretrained_model in enforced_node_type and args.pretrained_model != enforced_node_type[
        args.pretrained_model]:
        args.__dict__["node_type"] = enforced_node_type[args.pretrained_model]

    if new_args.pretrained_model in ["IPA", "Diffusion"]:
        logging.warning("Forcing batch_size to 1 for IPA model (learning-rate is reduced proportionally). Also forcing GNN type to 'identity', embeddings_type to 'of' and fine_tuning.")
        new_args.__dict__[
            "gnn_type"] = "identity"  # we could also test combination of IPA and GNN, but it adds combplexity
        # Adjusting learning rate for the reduced batch size
        new_args.__dict__["learning_rate"] = new_args.__dict__["learning_rate"] / new_args.__dict__["batch_size"]
        new_args.__dict__["batch_size"] = 1
        new_args.__dict__["embeddings_type"] = "of"

        # Enforce fine-tuning
        if not new_args.__dict__["warm_up_epochs"]:
            # Do 10% Finetuning
            new_args.__dict__["warm_up_epochs"] = new_args.__dict__["max_epochs"] // 10

    if args.preprocessed_to_scratch and not args.preprocess_graph:
        logging.warning("preprocessed_to_scratch only works with --preprocess_graph activated. Enabling forcefully...")
        args.__dict__["preprocess_graph"] = True

    new_args.tqdm_output = False  # disable tqdm output to reduce log syncing
    if wandb_name:
        # In a sweep, we update the config and thereby set the name to the updated config entries
        new_args.wandb_name = wandb_name
    return new_args
