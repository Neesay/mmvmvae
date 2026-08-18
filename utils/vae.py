import torch
from torch import nn
from config.ExperimentConfig import ExperimentConfig

from networks.CelebA import EncoderText, DecoderText, EncoderImg, DecoderImg
from networks.PolyMNIST import Encoder, Decoder
from networks.scMNC import scEncoder, scDecoder
from networks.PolyMNIST import ResnetEncoder, ResnetDecoder
from networks.JointPrior import OrthogMat
# from networks.NetworksRatsspike import Encoder as RatsEncoder
# from networks.NetworksRatsspike import Decoder as RatsDecoder


def _compile(module: nn.Module, cfg: ExperimentConfig) -> nn.Module:
    # Optionally torch.compile() a single encoder/decoder submodule for extra
    # kernel fusion / tensor-core throughput on Ampere+ GPUs. Applied per-module
    # (not to the LightningModule) since these are static nn.Sequential-style
    # stacks with fixed input shapes (drop_last=True everywhere), which is the
    # shape torch.compile handles best. Falls back to the uncompiled module if
    # compilation isn't available/fails, so turning the flag on can't hard-crash
    # a run in an environment without a working Triton/compiler backend.
    if not getattr(cfg.model, "compile_networks", False):
        return module
    try:
        return torch.compile(module)
    except Exception as exc:
        print(f"torch.compile failed for {module.__class__.__name__} ({exc}); continuing uncompiled.")
        return module


def get_networks(cfg: ExperimentConfig) -> list[nn.ModuleList]:
    if cfg.dataset.name.startswith("PM"):
        if not cfg.model.use_resnets:
            encoders = nn.ModuleList(
                [
                    _compile(Encoder(cfg.model.latent_dim).to(cfg.model.device), cfg)
                    for _ in range(cfg.dataset.num_views)
                ]
            )
            decoders = nn.ModuleList(
                [
                    _compile(Decoder(cfg.model.latent_dim).to(cfg.model.device), cfg)
                    for _ in range(cfg.dataset.num_views)
                ]
            )
        else:
            encoders = nn.ModuleList(
                [
                    _compile(ResnetEncoder(cfg).to(cfg.model.device), cfg)
                    for _ in range(cfg.dataset.num_views)
                ]
            )
            decoders = nn.ModuleList(
                [
                    _compile(ResnetDecoder(cfg).to(cfg.model.device), cfg)
                    for _ in range(cfg.dataset.num_views)
                ]
            )
    elif cfg.dataset.name.startswith("celeba"):
        encoders = nn.ModuleList(
            [
                _compile(EncoderImg(cfg).to(cfg.model.device), cfg),
                _compile(EncoderText(cfg).to(cfg.model.device), cfg),
            ]
        )
        decoders = nn.ModuleList(
            [
                _compile(DecoderImg(cfg).to(cfg.model.device), cfg),
                _compile(DecoderText(cfg).to(cfg.model.device), cfg),
            ]
        )
    elif cfg.dataset.name.startswith("sc"):
        original_dims = [1302, 39]
        encoders = nn.ModuleList(
            [
                _compile(
                    scEncoder(original_dims[m], cfg.model.latent_dim, cfg.model.hidden_dim).to(cfg.model.device),
                    cfg,
                )
                for m in range(cfg.dataset.num_views)
            ]
        )
        decoders = nn.ModuleList(
            [
                _compile(
                    scDecoder(original_dims[m], cfg.model.latent_dim, cfg.model.hidden_dim).to(cfg.model.device),
                    cfg,
                )
                for m in range(cfg.dataset.num_views)
            ]
        )
    else:
        raise NotImplementedError(
            "Unknown dataset/networks to create encoders and decoders for specified config"
        )

    if cfg.model.name == "jointprior":
      cov_mat = nn.ModuleList(
          [
            OrthogMat(cfg).to(cfg.model.device) for _ in range(cfg.dataset.num_views - 1)
          ]
        )
    else:
      # cov_mat is only ever indexed/used by the jointprior model (see
      # MVJointPriorVAE.compute_loss); avoid allocating an unused
      # latent_dim*num_views square identity matrix for the other model types.
      cov_mat = nn.ModuleList()

    covariance = torch.eye(cfg.model.latent_dim * cfg.dataset.num_views).to(cfg.model.device)
    mu = torch.zeros(cfg.model.latent_dim * cfg.dataset.num_views).to(cfg.model.device)

    return [encoders, decoders, cov_mat, covariance, mu]
