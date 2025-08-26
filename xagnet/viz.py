from pathlib import Path
import matplotlib.pyplot as plt

def plot_and_save_history(history, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(history.history.get("loss", []), label="loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.legend(); plt.title("Loss"); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.tight_layout(); plt.savefig(out_dir / "loss.png"); plt.close()

    # Dice (optional)
    if "dice_coef" in history.history or "val_dice_coef" in history.history:
        plt.figure(figsize=(6,4))
        if "dice_coef" in history.history:
            plt.plot(history.history["dice_coef"], label="dice_coef")
        if "val_dice_coef" in history.history:
            plt.plot(history.history["val_dice_coef"], label="val_dice_coef")
        plt.legend(); plt.title("Dice Coefficient"); plt.xlabel("epoch"); plt.ylabel("dice")
        plt.tight_layout(); plt.savefig(out_dir / "dice.png"); plt.close()
