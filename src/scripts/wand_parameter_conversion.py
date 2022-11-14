import json
import wandb


def get_run_config(run_path: str):
    api = wandb.Api()
    run = api.run(run_path)
    config_str = run.json_config
    config = json.loads(config_str)
    return config


def convert_to_args(run_config: dict):
    args = ""
    for k, v in run_config.items():
        if k == "num_workers":
            continue
        value = v["value"]
        if  (isinstance(value, bool) and not value) or (isinstance(value, str) and value == ""):
            continue
        elif isinstance(value, bool) and value:
            args += "--"
            args += k
            args += "\n"
        else:
            args += "--"
            args += k
            args += " "
            args += str(value)
            args += "\n"
    return args


if __name__ == "__main__":
    run_path = "fabian22/abag_binding_affinity/d8gpm66d"
    run_config = get_run_config(run_path)
    args = convert_to_args(run_config)
    args_line = args.replace("\n", " ")
    print(args)