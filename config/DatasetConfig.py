from dataclasses import dataclass, field
from typing import List
from omegaconf import MISSING
from config.UserVariables import folder_path

# NCDE: never changed during experiments - this variable was never modified in the experiments, but it is included in the config for completeness and potential future use.
# note: just because a variable doesn't have NCDE beside it doesn't mean it was changed, this has just been added for clarity. 
@dataclass
class DataConfig:
    name: str = MISSING
    num_workers: int = 8
    # num views
    num_views: int = MISSING
    # Preload the decoded dataset into RAM once (as uint8) instead of
    # re-decoding every PNG on every epoch. Currently honoured by PolyMNIST
    # only; scMNC already holds its data in memory, and CelebA would need its
    # own implementation. Val is cached by default (~118 MB for 10k samples);
    # train is opt-in because it is 6x larger (~705 MB for 60k samples).
    cache_val: bool = True
    cache_train: bool = False


@dataclass
class PolyMNISTDataConfig(DataConfig):
    num_views: int = 5
    # index of the modality which should be treated as the first modality for JPVAE
    modalities_order: int = 0
    dir_data_base: str = folder_path + "data"
    dir_clfs_base: str = (
        folder_path + "trained_classifiers/polyMNIST"
    )
    n_clfs_outputs: int = 10
    # the number of corresponding labels for the image (default 1 for the digit label, but could be 2 if we also include the color label) (NCDE)
    num_labels: int = 1 


@dataclass
class PMtranslatedData75Config(PolyMNISTDataConfig):
    name: str = "PM_translated75"
    suffix_data_train: str = "MMNIST/train" 
    suffix_data_test: str = "MMNIST/test" 
    suffix_clfs: str = "translated75_resnet"


@dataclass
class CelebADataConfig(DataConfig):
    name: str = "celeba"
    num_views: int = 2
    dir_data: str = folder_path + "data/CelebA"
    dir_alphabet: str = (
        folder_path + "utils"
    )
    dir_clf: str = (
        folder_path + "trained_classifiers/celeba"
    )
    dir_clfs_base: str = (
        folder_path + "trained_classifiers/celeba"
    )
    
    ## none of the following variables were changed during the experiments
    # they are included in the config for completeness and potential future use
    
    # maximum length of sequence considered for the text modality 
    len_sequence: int = 256
    # can be used to select different versions of the text modality, by randomly shuffling the order of the attributes/indices
    random_text_ordering: bool = False
    random_text_startindex: bool = True
    # number of channels in the image modality (default 3 for RGB)
    image_channels: int = 3
    # size to which the original 218x178 image is cropped before resizing to img_size x img_size 
    crop_size_img: int = 148
    img_size: int = 64
    # number of outputs of the classifier trained on the CelebA dataset (default 40 for the 40 attributes) 
    n_clfs_outputs: int = 40
    num_labels: int = 40
    # length of the alphabet used for the text modality + 1 for the "end of sequence" token 
    num_features: int = 41
    
    
    # archictecture variables 
    num_layers_img: int = 5
    num_layers_text: int = 7
    filter_dim_img: int = 64
    filter_dim_text: int = 64
    skip_connections_img_weight_a: float = 1.0
    skip_connections_img_weight_b: float = 1.0
    skip_connections_text_weight_a: float = 1.0
    skip_connections_text_weight_b: float = 1.0

    use_rec_weight: bool = True
    include_channels_rec_weight: bool = False
    
@dataclass
class scMNCDataConfig(DataConfig):
    name: str = "scMNC"
    num_views: int = 2
    dir_data: str = folder_path + "data/scMNC"
    num_labels: int = 6
    dir_clf: str = (
        folder_path + "trained_classifiers/scMNC"
    )
    n_clfs_outputs: int = 6
    label_names: List[str] = field(
        default_factory=lambda: [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        ]
    )
