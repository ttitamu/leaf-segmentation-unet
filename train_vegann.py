import os
import glob
import time
import copy
from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast  # mixed precision
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
import albumentations as A
from model import dice_loss, ResNetUNet
from preprocess import check_dir

# ---------------------------------------------------------------------------
# Anchor weight path to the script file location
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
WEIGHT_PATH = os.path.join(SCRIPT_DIR, "model", "pretrained")
check_dir(WEIGHT_PATH)

# ---------------------------------------------------------------------------
# Parallelism helpers
# ---------------------------------------------------------------------------
def get_num_workers():
    """
    Pick a safe number of DataLoader worker processes.

    - On Windows, 'spawn' multiprocessing means workers are slow to start,
      so we cap at 4.
    - On Linux / Mac, 'fork' is used and more workers are fine.
    - Always leave at least 1 CPU free for the main process.
    """
    import platform
    cpu_count = os.cpu_count() or 1
    if platform.system() == "Windows":
        # Windows uses 'spawn' — more than 4 rarely helps and can cause issues
        return min(4, max(1, cpu_count - 1))
    else:
        return max(1, cpu_count - 1)


def get_device_and_model(num_class=1):
    """
    Build the model and move it to the best available device.
    Uses all available GPUs via DataParallel if more than one is present.

    Returns:
        model  (nn.Module): Model ready for training.
        device (torch.device): Primary device.
    """
    if torch.cuda.is_available():
        device     = torch.device("cuda:0")
        model      = ResNetUNet(num_class).to(device)
        n_gpus     = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"Using {n_gpus} GPUs via DataParallel")
            # DataParallel splits each batch across all GPUs automatically
            model = torch.nn.DataParallel(model)
        else:
            print(f"Using 1 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        model  = ResNetUNet(num_class).to(device)
        print("No GPU found — running on CPU (training will be slow)")

    return model, device


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def read_imgs_and_masks(images_dir, masks_dir, display=False):
    """
    Read images and their corresponding masks from separate directories.
    Both folders must contain files with identical names (e.g. VegAnn_96.png).

    Args:
        images_dir (str):  Path to the folder containing raw images.
        masks_dir  (str):  Path to the folder containing annotation masks.
        display    (bool): If True, display up to 2 random image-mask pairs.

    Returns:
        valid_img_paths  (list): Sorted list of valid image file paths.
        valid_mask_paths (list): Corresponding mask file paths (same order).
    """
    for directory in [images_dir, masks_dir]:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

    img_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpg"))
    )

    if len(img_paths) == 0:
        print(f"Warning: No images found in {images_dir}")
        print(f"Files present: {os.listdir(images_dir)}")
        return [], []

    print(f"Found {len(img_paths)} images in {images_dir}")

    mask_paths = [
        os.path.join(masks_dir, os.path.basename(p))
        for p in img_paths
    ]

    valid_img_paths, valid_mask_paths = [], []
    missing_count = 0

    for img_path, mask_path in zip(img_paths, mask_paths):
        if os.path.exists(mask_path):
            valid_img_paths.append(img_path)
            valid_mask_paths.append(mask_path)
        else:
            print(f"Warning: No matching mask for: {img_path}")
            missing_count += 1

    if missing_count > 0:
        print(
            f"Warning: {missing_count} image(s) skipped — "
            f"{len(valid_img_paths)} valid pairs remaining."
        )

    if display and len(valid_img_paths) > 0:
        indices = np.random.randint(
            low=0, high=len(valid_img_paths),
            size=(min(2, len(valid_img_paths)),)
        )
        for i in indices:
            mask_img = cv2.imread(valid_mask_paths[i])
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            _, mask_img = cv2.threshold(mask_img, 5, 255, cv2.THRESH_BINARY)
            org_img = cv2.imread(valid_img_paths[i])
            org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
            _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
            ax1.set_title("Image")
            ax2.set_title("Mask")
            ax1.imshow(org_img)
            ax2.imshow(mask_img, cmap="gray")
            plt.tight_layout()
            plt.show()

    return valid_img_paths, valid_mask_paths


def split_train_val(img_paths, mask_paths, val_split=0.2, seed=42):
    """
    Randomly split paired image/mask lists into train and validation subsets.

    Args:
        img_paths  (list):  All image paths.
        mask_paths (list):  Corresponding mask paths.
        val_split  (float): Fraction reserved for validation.
        seed       (int):   Random seed.

    Returns:
        train_imgs, train_masks, val_imgs, val_masks (lists)
    """
    np.random.seed(seed)
    n         = len(img_paths)
    indices   = np.random.permutation(n)
    split_idx = int(n * (1 - val_split))

    train_idx = indices[:split_idx]
    val_idx   = indices[split_idx:]

    train_imgs  = [img_paths[i]  for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]
    val_imgs    = [img_paths[i]  for i in val_idx]
    val_masks   = [mask_paths[i] for i in val_idx]

    print(f"Split -> Train: {len(train_imgs)}, Val: {len(val_imgs)}")
    return train_imgs, train_masks, val_imgs, val_masks


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class parseDataset(Dataset):
    """
    PyTorch Dataset for VegAnn image-mask pairs.
    Caches resized images in memory after first load to speed up
    subsequent epochs.
    """

    def __init__(self, img_paths, mask_paths,
                 transform=None, augment=None,
                 img_size=224, cache=False):
        """
        Args:
            img_paths  (list):  Image file paths.
            mask_paths (list):  Mask file paths.
            transform  :        torchvision transform applied after augment.
            augment    :        albumentations pipeline.
            img_size   (int):   Resize target (square).  Default 224.
            cache      (bool):  If True, load all images into RAM on init.
                                Fast but requires enough memory.
        """
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.transform  = transform
        self.augment    = augment
        self.img_size   = img_size
        self.cache      = cache
        self._img_cache  = {}
        self._mask_cache = {}

        print(f"Dataset initialised with {len(self.img_paths)} image-mask pairs"
              f"{'  [caching enabled]' if cache else ''}")

        if self.cache:
            print("Pre-loading dataset into RAM...")
            for idx in range(len(self.img_paths)):
                self._load_and_cache(idx)
            print("Pre-loading complete.")

    def _load_and_cache(self, idx):
        """Load one image-mask pair, resize, and store in the cache dicts."""
        image = cv2.imread(self.img_paths[idx])
        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {self.img_paths[idx]}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Pre-resize to the target size to avoid repeated resizing per epoch
        image = cv2.resize(
            image, (self.img_size, self.img_size),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.imread(self.mask_paths[idx])
        if mask is None:
            raise FileNotFoundError(
                f"Could not read mask: {self.mask_paths[idx]}"
            )
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.resize(
            mask, (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST   # nearest-neighbour keeps binary values
        )

        self._img_cache[idx]  = image
        self._mask_cache[idx] = mask

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Retrieve from cache if available, otherwise load from disk
        if self.cache and idx in self._img_cache:
            image = self._img_cache[idx].copy()   # copy so augment doesn't mutate cache
            mask  = self._mask_cache[idx].copy()
        else:
            image = cv2.imread(self.img_paths[idx])
            if image is None:
                raise FileNotFoundError(
                    f"Could not read image: {self.img_paths[idx]}"
                )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(self.mask_paths[idx])
            if mask is None:
                raise FileNotFoundError(
                    f"Could not read mask: {self.mask_paths[idx]}"
                )
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Albumentations augmentation (training only)
        if self.augment:
            augmented = self.augment(image=image, mask=mask)
            image     = augmented["image"]
            mask      = augmented["mask"]

        if self.transform:
            image = self.transform(image)   # -> (3, H, W) float32 [0, 1]
            mask  = self.transform(mask)    # -> (1, H, W) float32 [0, 1]
            image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )(image)

        return image, mask


# ---------------------------------------------------------------------------
# Transforms & augmentations
# ---------------------------------------------------------------------------
trans = transforms.Compose([
    transforms.ToTensor()
])

aug = A.Compose([
    A.RandomScale(scale_limit=0.5, p=0.5),
    A.PadIfNeeded(min_height=224, min_width=224, p=1),
    A.RandomCrop(height=224, width=224, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.8),
    A.RandomGamma(p=0.8),
])


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def calc_loss(pred, target, metrics, bce_weight=0.5):
    """Combined BCE + Dice loss."""
    bce  = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics["bce"]  += bce.data.cpu().numpy()  * target.size(0)
    metrics["dice"] += dice.data.cpu().numpy() * target.size(0)
    metrics["loss"] += loss.data.cpu().numpy() * target.size(0)
    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = [
        "{}: {:4f}".format(k, metrics[k] / epoch_samples)
        for k in metrics.keys()
    ]
    print("{}: {}".format(phase, ", ".join(outputs)))


# ---------------------------------------------------------------------------
# Training loop  (with mixed-precision support)
# ---------------------------------------------------------------------------
def train_model(model, optimizer, scheduler, device,
                num_epochs=25, use_amp=True):
    """
    Train and validate the model.

    Key speed improvements over the original:
      • Mixed precision (AMP) — halves memory and speeds up GPU maths
      • GradScaler — keeps float16 gradients numerically stable
      • num_workers > 0 — parallel data loading fills the GPU pipeline

    Args:
        model      (nn.Module):   Segmentation model.
        optimizer  (Optimizer):   Optimiser.
        scheduler  (LRScheduler): Stepped after each training epoch.
        device     (torch.device):Primary compute device.
        num_epochs (int):         Total epochs.
        use_amp    (bool):        Enable automatic mixed precision (GPU only).

    Returns:
        model (nn.Module): Loaded with best validation weights.
    """
    # AMP is only useful on CUDA; disable silently on CPU
    use_amp        = use_amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss      = 1e10

    print(f"Mixed precision (AMP) : {'ON' if use_amp else 'OFF'}")
    print(f"DataLoader workers    : {NUM_WORKERS}")

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)
        since = time.time()

        for phase in ["train", "val"]:
            if phase == "train":
                for param_group in optimizer.param_groups:
                    print("LR", param_group["lr"])
                model.train()
            else:
                model.eval()

            metrics       = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()

                # autocast runs the forward pass in float16 where safe
                with torch.set_grad_enabled(phase == "train"):
                    with autocast("cuda", enabled=use_amp):
                        outputs = model(inputs)
                        loss    = calc_loss(outputs, labels, metrics)

                    if phase == "train":
                        # scaler handles gradient unscaling + optimizer step
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics["loss"] / epoch_samples

            if phase == "val" and epoch_loss < best_loss:
                print("Saving best model")
                best_loss      = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(
                    model.state_dict(),
                    os.path.join(WEIGHT_PATH, "best_val_weights.pth")
                )

            if phase == "train":
                scheduler.step()

        time_elapsed = time.time() - since
        print("{:.0f}m {:.0f}s\n".format(
            time_elapsed // 60, time_elapsed % 60
        ))

        torch.save(
            model.state_dict(),
            os.path.join(WEIGHT_PATH, "latest_weights.pth")
        )

    print("Best val loss: {:4f}".format(best_loss))
    model.load_state_dict(best_model_wts)
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # ── Set CPU thread count before anything else ────────────────────────
    num_cores = os.cpu_count() or 1
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(num_cores)
    print(f"CPU cores available : {num_cores}")
    print(f"PyTorch threads     : {torch.get_num_threads()}")

    IMAGES_DIR = os.path.join(SCRIPT_DIR, "dataset", "VegAnn_dataset", "images")
    MASKS_DIR  = os.path.join(SCRIPT_DIR, "dataset", "VegAnn_dataset", "annotations")

    print(f"Script location  : {SCRIPT_DIR}")
    print(f"Images directory : {IMAGES_DIR}  (exists: {os.path.exists(IMAGES_DIR)})")
    print(f"Masks  directory : {MASKS_DIR}   (exists: {os.path.exists(MASKS_DIR)})")

    all_img_paths, all_mask_paths = read_imgs_and_masks(
        IMAGES_DIR, MASKS_DIR, display=False
    )

    if len(all_img_paths) == 0:
        raise ValueError(
            f"No valid image-mask pairs found.\n"
            f"Expected images in : {IMAGES_DIR}\n"
            f"Expected masks in  : {MASKS_DIR}"
        )

    # Optional quick-test mode
    QUICK_TEST  = False
    SUBSET_SIZE = 200
    if QUICK_TEST:
        print(f"QUICK TEST MODE — using first {SUBSET_SIZE} images")
        all_img_paths  = all_img_paths[:SUBSET_SIZE]
        all_mask_paths = all_mask_paths[:SUBSET_SIZE]

    train_img_paths, train_mask_paths, \
    val_img_paths,   val_mask_paths = split_train_val(
        all_img_paths, all_mask_paths, val_split=0.2
    )

    IMAGE_SIZE    = 128    # reduce from 224 for faster CPU training
    CACHE_DATASET = True   # load everything into RAM once

    train_set = parseDataset(
        train_img_paths, train_mask_paths,
        transform=trans, augment=aug,
        img_size=IMAGE_SIZE, cache=CACHE_DATASET
    )
    val_set = parseDataset(
        val_img_paths, val_mask_paths,
        transform=trans,
        img_size=IMAGE_SIZE, cache=CACHE_DATASET
    )

    # On CPU, 4 workers is a safe default
    # On Windows, more than 4 rarely helps
    NUM_WORKERS = min(4, max(1, num_cores - 1))

    batch_size  = 16    # smaller batch is faster per step on CPU
    dataloaders = {
        "train": DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=False,          # pin_memory does nothing on CPU
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
        ),
        "val": DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
        ),
    }

    model, device = get_device_and_model(num_class=1)

    # Freeze backbone to reduce computation
    print("\nFreezing backbone layers...")
    frozen_count = trained_count = 0
    for name, param in model.named_parameters():
        if any(l in name for l in [
            "base_layers", "layer0", "layer1", "layer2", "layer3", "layer4"
        ]):
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True
            trained_count += 1
    print(f"  Frozen     : {frozen_count} parameter tensors")
    print(f"  Trainable  : {trained_count} parameter tensors")

    optimizer_ft     = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=15, gamma=0.1
    )

    print(f"\n{'='*50}")
    print("TRAINING CONFIGURATION (CPU MODE)")
    print(f"{'='*50}")
    print(f"  Total images       : {len(all_img_paths)}")
    print(f"  Training images    : {len(train_img_paths)}")
    print(f"  Validation images  : {len(val_img_paths)}")
    print(f"  Image size         : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Batch size         : {batch_size}")
    print(f"  Epochs             : 30")
    print(f"  CPU cores          : {num_cores}")
    print(f"  PyTorch threads    : {torch.get_num_threads()}")
    print(f"  DataLoader workers : {NUM_WORKERS}")
    print(f"  Dataset caching    : {'ON' if CACHE_DATASET else 'OFF'}")
    print(f"  Backbone frozen    : YES")
    print(f"  Quick test mode    : {'ON' if QUICK_TEST else 'OFF'}")
    print(f"{'='*50}\n")

    model = train_model(
        model, optimizer_ft, exp_lr_scheduler,
        device, num_epochs=30, use_amp=False
    )

    print("\nTraining complete.")
    print(f"Best weights   : {os.path.join(WEIGHT_PATH, 'best_val_weights.pth')}")
    print(f"Latest weights : {os.path.join(WEIGHT_PATH, 'latest_weights.pth')}")