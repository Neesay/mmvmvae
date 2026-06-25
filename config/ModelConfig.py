from dataclasses import dataclass

@dataclass
class ModelConfig:
    device: str = "cuda"
    batch_size: int = 128 
    batch_size_eval: int = 64
    lr: float = 5e-4 
    epochs: int = 400 
    temp_annealing: str = "cosine" # alpha annealing procedure for mixed prior
    seed: int = 1
    latent_dim: int = 256
    hidden_dim: int = 512 # for encoders/decoders hidden layers for scMNC
    early_stop: bool = False
    compile: bool = True

    # loss hyperparameters
    beta_annealing: bool = True # True
    init_beta_value: float = 0 
    final_beta_value: float = 1.0 
    schedule: str = "cyclical"
    beta_annealing_steps: int = 250000
    beta_M: int = 4
    beta_R: float = 0.5

    # network architectures
    use_resnets: bool = True

@dataclass
class JointModelConfig(ModelConfig):
    name: str = "joint"
    aggregation: str = "poe"

@dataclass
class MixedPriorModelConfig(ModelConfig):
    name: str = "mixedprior"
    # weight on N(0,1) in mixed prior
    alpha_annealing: bool = True # True
    init_alpha_value: float = 1.0 # 1
    final_alpha_value: float = 0 # 0
    alpha_annealing_steps: int = 150000
    
@dataclass
class JointPriorModelConfig(ModelConfig):
    name: str = "jointprior"
    # weight on N(0,1) in mixed prior
    alpha_annealing: bool = True # True
    init_alpha_value: float = 1.0 # 1
    final_alpha_value: float = 0 # 0
    alpha_annealing_steps: int = 150000
    cov_scalar: float =  1 - 0.9 ** 2 # 1 - alpha^2
    alpha_scalar: float = 0.9


@dataclass
class UnimodalModelConfig(ModelConfig):
    name: str = "unimodal"