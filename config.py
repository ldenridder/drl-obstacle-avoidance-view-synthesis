from dataclasses import dataclass

@dataclass
class Paths:
    log: str
    data: str

@dataclass
class Settings:
    wandb: bool
    rendering: bool
    perceptual_loss: bool

@dataclass
class Files:
    train_data: str
    train_labels: str
    test_data: str
    test_labels: str
    val_data: str
    left_image_data: str
    right_image_data: str
    image_height: int
    image_width: int
    path: str

@dataclass
class HyperParams:
    epoch_count:int
    lr: float
    batch_size: int
    mean: list
    std: list

@dataclass
class ModelParams:
    kernel_size:int
    n_layers:int
    size_layer:int
    num_channels_1: int
    num_channels_2: int
    num_channels_3: int
    num_channels_4: int

@dataclass
class SwinParams:
    img_size: list

@dataclass
class KITTIConfig:
    files: Files
    hyperparams: HyperParams
    modelparams: ModelParams
    settings: Settings

@dataclass
class CityScapesConfig:
    files: Files
    hyperparams: HyperParams
    modelparams: ModelParams
    swinparams: SwinParams
    settings: Settings