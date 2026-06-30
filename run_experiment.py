import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profilers import PyTorchProfiler
import wandb

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch # added for torch.set_float32_matmul_precision
from utils import dataset
from config.UserVariables import folder_path
torch.set_float32_matmul_precision('high')
print("[OPTIMISATION] TensorFloat32 (TF32) matmul precision enabled ('high').")
torch.backends.cudnn.benchmark = True
print("[OPTIMISATION] cuDNN Benchmark enabled (will profile algorithms on first batch).")
os.environ["WANDB__SERVICE_WAIT"] = "300"

# experiment configs 
from config.ExperimentConfig import ExperimentConfig
from config.ExperimentConfig import LogConfig
from config.ExperimentConfig import EvalConfig
# model configs
from config.ModelConfig import JointModelConfig
from config.ModelConfig import MixedPriorModelConfig
from config.ModelConfig import UnimodalModelConfig
from config.ModelConfig import JointPriorModelConfig
# dataset configs
from config.DatasetConfig import PMtranslatedData75Config
from config.DatasetConfig import CelebADataConfig
from config.DatasetConfig import scMNCDataConfig
# models
from mv_vaes.mv_joint_vae import MVJointVAE as MVJointVAE
from mv_vaes.mv_unimodal_vae import MVunimodalVAE as MVunimodalVAE
from mv_vaes.mv_mixedprior_vae import MVMixedPriorVAE as MVMixedPriorVAE
from mv_vaes.mv_jointprior_vae import MVJointPriorVAE as MVJointPriorVAE

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(group="log", name="log", node=LogConfig)
cs.store(group="model", name="joint", node=JointModelConfig)
cs.store(group="model", name="mixedprior", node=MixedPriorModelConfig)
cs.store(group="model", name="jointprior", node=JointPriorModelConfig)
cs.store(group="model", name="unimodal", node=UnimodalModelConfig)
cs.store(group="eval", name="eval", node=EvalConfig)
cs.store(group="dataset", name="PMtranslated75", node=PMtranslatedData75Config)
cs.store(group="dataset", name="CelebA", node=CelebADataConfig)
cs.store(group="dataset", name="scMNC", node=scMNCDataConfig)
cs.store(name="base_config", node=ExperimentConfig)

class BetaEarlyStopping(EarlyStopping):
    def __init__(self, *args, beta_threshold=1.0, tolerance=1e-6, **kwargs):
        kwargs.setdefault("check_on_train_epoch_end", False)
        super().__init__(*args, **kwargs)
        self.beta_threshold = beta_threshold
        self.tolerance = tolerance

    def on_validation_end(self, trainer, pl_module):
        beta = float(pl_module._get_current_beta_weight())
        enabled = abs(beta - self.beta_threshold) <= self.tolerance
        
        if enabled:
            # Call parent's validation end, which triggers early stopping check
            super().on_validation_end(trainer, pl_module)
        # else: do nothing, skip the parent call

    def on_train_epoch_end(self, trainer, pl_module):
        # Explicitly block any train-epoch early stopping checks
        pass

@hydra.main(version_base=None, config_path="config", config_name="config")
def run_experiment(cfg: ExperimentConfig):
    print(cfg)
    
    if cfg.log.wandb_local_instance:
        wandb.login(host=os.getenv("WANDB_LOCAL_URL"))
    elif not cfg.log.wandb_offline:
        wandb.login()
        
    pl.seed_everything(cfg.model.seed, workers=True)

    # get data loaders
    train_loader, train_dst, val_loader, _ = dataset.get_dataset(cfg)
    label_names = train_dst.label_names

    # if using the cpu, use cpu specific classifier directoryß
    if cfg.model.device != "cuda":
        cfg.dataset.dir_clf = cfg.dataset.dir_clf + "_cpu"
    # initialise model
    model = None
    if cfg.model.name == "joint":
        model = MVJointVAE(cfg)
    elif cfg.model.name == "mixedprior":
        model = MVMixedPriorVAE(cfg)
    elif cfg.model.name == "unimodal":
        model = MVunimodalVAE(cfg)
    elif cfg.model.name == "jointprior":
        model = MVJointPriorVAE(cfg)
    assert model is not None
    model.assign_label_names(label_names)

    # Convert model to channels_last for optimal GPU kernel dispatch
    if cfg.model.device == "cuda":
        model = model.to(memory_format=torch.channels_last)
        print("[OPTIMISATION] Model converted to channels_last memory format for Tensor Cores.")
    summary = ModelSummary(model, max_depth=2)
    print(summary)
    agg = getattr(cfg.model, "aggregation", "none") if "aggregation" in cfg.model else "none"
    alpha = getattr(cfg.model, "alpha_scalar", "none") if "alpha_scalar" in cfg.model else "none"
    filename = f'{cfg.model.name}_{agg}_{alpha}_{cfg.model.early_stop}_{cfg.model.beta_annealing}_{cfg.model.beta_M}_{cfg.model.seed}'
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.log.dir_logs,
        monitor=cfg.checkpoint_metric,
        mode="min",
        save_last=True,
        filename=filename
    )
    
    # early stopping specification
    early_stopping = BetaEarlyStopping(monitor="val/loss/loss", mode="min")
    wandb_logger = WandbLogger(
        name=cfg.log.wandb_run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        project=cfg.log.wandb_project_name,
        group=cfg.log.wandb_group,
        offline=cfg.log.wandb_offline,
        entity=cfg.log.wandb_entity,
        save_dir=cfg.log.dir_logs,
    )
    
    profiler = "simple"
    if getattr(cfg.log, "profile", False):
        profiler = PyTorchProfiler(
            dirpath=os.path.join(cfg.log.dir_logs, "profiler"),
            filename=f"profiler_{filename}",
            export_to_chrome=True,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )

    trainer = pl.Trainer(
        max_epochs=cfg.model.epochs,
        devices=1,
        accelerator="gpu" if cfg.model.device == "cuda" else cfg.model.device,
        precision="bf16-mixed" if cfg.model.device == "cuda" else "32-true",
        logger=wandb_logger,
        check_val_every_n_epoch=cfg.log.val_freq,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, early_stopping] if cfg.model.early_stop else [checkpoint_callback],
        profiler=profiler,
    )

    if cfg.log.debug:
        trainer.logger.watch(model, log="all")
        

    # train and evaluate model
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # log metrics
    model.logger.log_metrics({"final_scores/rec_loss": model.final_scores_rec_loss})
    model.logger.log_metrics(
        {"final_scores/cond_rec_loss": model.final_scores_cond_rec_loss}
    )
    model.logger.log_metrics(
      {"final_scores/cond_rec_loss_cov": model.final_scores_cond_rec_loss_cov }
    )
    model.logger.log_metrics({"best_scores/val_loss": model.best_val_loss})
    model.logger.log_metrics({"best_scores/rec_loss": model.best_scores_rec_loss})
    model.logger.log_metrics(
        {"best_scores/cond_rec_loss": model.best_scores_cond_rec_loss}
    )
    model.logger.log_metrics(
        {"best_scores/cond_rec_loss_cov": model.best_scores_cond_rec_loss_cov }
    )
      
    for m, key in enumerate(model.modality_names):
        model.logger.log_metrics(
            {
                f"final_scores/downstream_lr/aggregated/{key}": model.final_scores_lr_aggregated[
                    m
                ]
            }
        )
        model.logger.log_metrics(
            {
                f"final_scores/downstream_lr/unimodal/{key}": model.final_scores_lr_unimodal[
                    m
                ]
            }
        )
        model.logger.log_metrics(
            {
                f"best_scores/downstream_lr/aggregated/{key}": model.best_scores_lr_aggregated[
                    m
                ]
            }
        )
        model.logger.log_metrics(
            {
                f"best_scores/downstream_lr/unimodal/{key}": model.best_scores_lr_unimodal[
                    m
                ]
            }
        )
        if cfg.dataset.name == "celeba":
            for k, l_name in enumerate(label_names):
                model.logger.log_metrics(
                    {
                        f"final_scores/downstream_lr/aggregated/{key}/{l_name}": model.final_scores_lr_aggregated_alllabels[
                            m, k
                        ]
                    }
                )
                model.logger.log_metrics(
                    {
                        f"final_scores/downstream_lr/unimodal/{key}/{l_name}": model.final_scores_lr_unimodal_alllabels[
                            m, k
                        ]
                    }
                )
                model.logger.log_metrics(
                    {
                        f"best_scores/downstream_lr/aggregated/{key}/{l_name}": model.best_scores_lr_aggregated_alllabels[
                            m, k
                        ]
                    }
                )
                model.logger.log_metrics(
                    {
                        f"best_scores/downstream_lr/unimodal/{key}/{l_name}": model.best_scores_lr_unimodal_alllabels[
                            m, k
                        ]
                    }
                )

    for m, key in enumerate(model.modality_names):
        for m_tilde, key_tilde in enumerate(model.modality_names):
            model.logger.log_metrics(
                {
                    f"final_scores/coherence/{key}_to_{key_tilde}": model.final_scores_coh[
                        m, m_tilde, :
                    ].mean()
                }
            )
            model.logger.log_metrics(
                {
                    f"final_scores/coherence/{key}_to_{key_tilde}_cov": model.final_scores_coh_cov[
                        m, m_tilde, :
                    ].mean()
                }
            )
            model.logger.log_metrics(
                {
                    f"best_scores/coherence/{key}_to_{key_tilde}": model.best_scores_coh[
                        m, m_tilde, :
                    ].mean()
                }
            )
            model.logger.log_metrics(
                {
                    f"best_scores/coherence/{key}_to_{key_tilde}_cov": model.best_scores_coh_cov[
                        m, m_tilde, :
                    ].mean()
                }
            )


if __name__ == "__main__":
    run_experiment()
