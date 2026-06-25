from dataclasses import dataclass

from omegaconf import MISSING

from config.DatasetConfig import DataConfig
from config.ModelConfig import ModelConfig
from config.UserVariables import folder_path, wandb_entity

@dataclass
class LogConfig:
    # wandb
    wandb_entity: str = wandb_entity
    wandb_group: str = ""
    wandb_run_name: str = ""
    wandb_project_name: str = "multi_view_vae"
    wandb_log_freq: int = 50 # 50
    wandb_offline: bool = False
    wandb_local_instance: bool = False

    # logs
    dir_logs: str = folder_path + "logs"

    # logging frequencies
    downstream_logging_frequency: int = 50 
    coherence_logging_frequency: int = 50 
    img_plotting_frequency: int = 50 
    fid_logging_frequency: int = 50
    val_freq: int = 50

    # debug level wandb
    debug: bool = False

    # profiler
    profile: bool = False


@dataclass
class EvalConfig:
    # latent representation
    num_samples_train: int = 10000
    max_iteration: int = 10000
    eval_downstream_task: bool = True

    # coherence
    coherence: bool = True

    # fid
    path_inception_weights: str = (
        folder_path + "utils/pt_inception-2015-12-05-6726825d.pth"
    )


@dataclass
class ExperimentConfig:
    seed: int = 0
    checkpoint_metric: str = "val/loss/loss"
    # logger
    log: LogConfig = MISSING
    # dataset
    dataset: DataConfig = MISSING
    # model
    model: ModelConfig = MISSING
    # eval
    eval: EvalConfig = MISSING
