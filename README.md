# XAG-Net: Cross-Slice Attention and Skip Gating Network for 2.5D Femur MRI Segmentation

Official implementation of **XAG-Net**, a novel 2.5D U-Net–based architecture with **Cross-Slice Attention (CSA)** and **Skip Attention Gating (AG)** modules.  
This repository accompanies the paper:

> **XAG-Net: A Cross-Slice Attention and Skip Gating Network for 2.5D Femur MRI Segmentation**  
> Byunghyun Ko, Anning Tian, Jeongkyu Lee  
> *Proc. of International Conference on Artificial Intelligence, Computer, Data Sciences and Applications (ACDSA 2025)*  

**arXiv preprint:** [arXiv:2508.06258](https://arxiv.org/abs/2508.06258)

---

## Key Contributions
- **2.5D Input Strategy**: Three adjacent axial MRI slices are stacked as input, capturing partial volumetric context while keeping computation efficient.
- **Cross-Slice Attention (CSA)**: Pixel-wise softmax normalization across slices at each spatial location for fine-grained inter-slice modeling.
- **Attention Gating (AG)**: Skip-level gating modules refine intra-slice features by suppressing irrelevant background.
- **Combined Loss**: Dice + Boundary loss for robust overlap and edge alignment.
- **Accuracy**: Achieves Dice ≈ **0.95**, surpassing 2D, 2.5D, and 3D baselines while using fewer FLOPs.

---

## 🗂️ Dataset Structure

Organize your dataset into separate directories for **train** and **val**.  
Each directory must contain `images/` and `masks/` subfolders with matching PNG files:

```
dataset_root/
├── train/
│   ├── images/
│   │   ├── TD01_S1_001.png
│   │   └── ...
│   └── masks/
│       ├── TD01_S1_001.png
│       └── ...
└── val/
    ├── images/
    └── masks/
```

- File naming convention should follow: `TD##_S#_##.png` (e.g., `TD01_S1_045.png`).  
- Each slice has a corresponding binary mask.  
- The loader automatically builds 2.5D stacks (`prev, current, next`) and pairs them with masks.

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/xag-net.git
cd xag-net
pip install -r requirements.txt
```

**Dependencies**:
- Python 3.8+
- TensorFlow 2.11+
- NumPy, OpenCV, Matplotlib, scikit-learn

---

## 🚀 Training

```bash
python -m xagnet.train \
  --train_dir /path/to/dataset/train \
  --val_dir /path/to/dataset/val \
  --out_dir outputs \
  --epochs 100 \
  --batch_size 4 \
  --lr 1e-4
```

**Options**:
- `--monitor` (default: `val_dice_coef`) – metric to monitor for best model.
- `--monitor_mode` (`min` or `max`) – whether lower or higher is better.
- `--input_h`, `--input_w` – override input resolution (default: 256×256).
- Per-split glob overrides (e.g., `--train_images_glob "images/*.png"`).

Models and plots are saved in `outputs/`:
- `xagnet_unet_best.h5` → checkpoint with best val score.  
- `xagnet_unet_final.h5` → final model after training.  
- `loss.png`, `dice.png` → training curves.

---

## 📊 Evaluation

To evaluate, load a saved model manually:

```python
import tensorflow as tf
from xagnet.losses import dice_coef

model = tf.keras.models.load_model("outputs/xagnet_unet_best.h5",
                                   custom_objects={"dice_coef": dice_coef})

# Use model.evaluate(X, y) on your evaluation dataset loaded with data_utils.load_dataset_from_dir
```

---

## 📈 Results (from the ACDSA paper)

On femur MRI datasets:

- **Full-scan Dice**: **0.9535** (↑12.3% over baselines)  
- **IoU**: 0.9160  
- **HD95**: 0.92 px  
- Outperforms 2D, 2.5D, and 3D U-Nets while using fewer parameters and FLOPs.

---

## 📄 Citation

If you use this code or paper, please cite:

```bibtex
@inproceedings{ko2025xagnet,
  title={XAG-Net: A Cross-Slice Attention and Skip Gating Network for 2.5D Femur MRI Segmentation},
  author={Ko, Byunghyun and Tian, Anning and Lee, Jeongkyu},
  booktitle={Proc. of International Conference on Artificial Intelligence, Computer, Data Sciences and Applications (ACDSA)},
  year={2025},
  organization={IEEE}
}
```

---
