from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # data
    data_dir: Path = Path("data")
    images_glob: str = "images/*.png"
    masks_glob: str = "masks/*.png"
    input_size: tuple = (256, 256)
    channels: int = 3  # 2.5D: prev, current, next slices stacked
    # training
    batch_size: int = 4
    epochs: int = 100
    learning_rate: float = 1e-3
    val_split: float = 0.2
    # output
    out_dir: Path = Path("outputs")
    model_name: str = "xagnet_unet"
    seed: int = 42
