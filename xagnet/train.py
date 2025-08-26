import argparse, random, os
import numpy as np
import tensorflow as tf
from pathlib import Path

from .config import Config
from .data_utils import load_dataset_from_dir
from .models import build_xagnet_unet, compile_model
from .losses import dice_coef, combined_validation_loss
from .viz import plot_and_save_history


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Train XAG-Net (TF/Keras) with explicit train/val directories (no testing)."
    )

    # path to directories
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Root dir for TRAIN set (contains images/, masks/)")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Root dir for VAL set (contains images/, masks/)")

    # Optional per-split glob overrides
    parser.add_argument("--train_images_glob", type=str, default="images/*.png",
                        help="Glob for TRAIN images")
    parser.add_argument("--train_masks_glob", type=str, default="masks/*.png",
                        help="Glob for TRAIN masks")
    parser.add_argument("--val_images_glob", type=str, default="images/*.png",
                        help="Glob for VAL images")
    parser.add_argument("--val_masks_glob", type=str, default="masks/*.png",
                        help="Glob for VAL masks")

    # General training params
    parser.add_argument("--out_dir", type=str, default=None, help="Where to save models/plots")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--input_h", type=int, default=None)
    parser.add_argument("--input_w", type=int, default=None)

    # Model selection behavior
    parser.add_argument("--monitor", type=str, default="val_dice_coef",
                        help="Metric to monitor for best checkpoint (e.g., val_dice_coef or val_loss)")
    parser.add_argument("--monitor_mode", type=str, default="max",
                        choices=["min", "max"], help="Direction for monitor metric")

    args = parser.parse_args()

    # Load config defaults, then apply CLI overrides
    cfg = Config()
    if args.out_dir:
        cfg.out_dir = Path(args.out_dir)
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.input_h and args.input_w:
        cfg.input_size = (args.input_w, args.input_h)

    set_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load datasets from directories ---
    X_tr, y_tr = load_dataset_from_dir(
        Path(args.train_dir),
        images_glob=args.train_images_glob,
        masks_glob=args.train_masks_glob,
        target_size=cfg.input_size,
    )
    X_va, y_va = load_dataset_from_dir(
        Path(args.val_dir),
        images_glob=args.val_images_glob,
        masks_glob=args.val_masks_glob,
        target_size=cfg.input_size,
    )

    # --- Build & compile model ---
    model = build_xagnet_unet(
        input_shape=(cfg.input_size[1], cfg.input_size[0], cfg.channels),
        num_classes=1
    )
    model = compile_model(
        model,
        learning_rate=cfg.learning_rate,
        loss=combined_validation_loss,   # change to "binary_crossentropy" if preferred
        metrics=[dice_coef],
    )

    # --- Callbacks ---
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(cfg.out_dir / f"{cfg.model_name}_best.h5"),
            monitor=args.monitor,
            mode=args.monitor_mode,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor=args.monitor, mode=args.monitor_mode, patience=10, restore_best_weights=True),
    ]

    # --- Train ---
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Save & Plot ---
    model.save(cfg.out_dir / f"{cfg.model_name}_final.h5")
    plot_and_save_history(history, cfg.out_dir)

if __name__ == "__main__":
    main()
