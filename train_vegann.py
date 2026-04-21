import os
import glob
import time
import copy
from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import albumentations as A
from model import dice_loss, ResNetUNet
from preprocess import check_dir

WEIGHT_PATH = "./model/pretrained"
check_dir(WEIGHT_PATH)


def read_imgs_and_masks(images_dir, masks_dir, display=False):
    """
    Read images and their corresponding masks from separate directories.
    Both folders should contain files with identical names (e.g., VegAnn_96.png).

    Args:
        images_dir (str): Path to the folder containing images
        masks_dir  (str): Path to the folder containing masks
        display   (bool): Optionally display random image-mask pairs

    Returns:
        img_paths  (list): Sorted list of image file paths
        mask_paths (list): Sorted list of mask file paths (same order as img_paths)
    """
    # --- 1. Validate that both directories exist ---
    for directory in [images_dir, masks_dir]:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

    # --- 2. Collect image paths (support both .jpg and .png) ---
    img_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpg"))
    )

    if len(img_paths) == 0:
        print(f"Warning: No images found in {images_dir}")
        print(f"Files present: {os.listdir(images_dir)}")
        return [], []

    print(f"Found {len(img_paths)} images in {images_dir}")

    # --- 3. Build mask paths by reusing the image filename in the masks directory ---
    # Example: "VegAnn_dataset/images/VegAnn_96.png"
    #       -> "VegAnn_dataset/annotations/VegAnn_96.png"
    mask_paths = [
        os.path.join(masks_dir, os.path.basename(img_path))
        for img_path in img_paths
    ]

    # --- 4. Keep only pairs where both the image AND the mask actually exist ---
    valid_img_paths, valid_mask_paths = [], []
    missing_count = 0

    for img_path, mask_path in zip(img_paths, mask_paths):
        if os.path.exists(mask_path):
            valid_img_paths.append(img_path)
            valid_mask_paths.append(mask_path)
        else:
            print(f"Warning: No matching mask for image: {img_path}")
            missing_count += 1

    if missing_count > 0:
        print(
            f"Warning: {missing_count} image(s) skipped due to missing masks. "
            f"{len(valid_img_paths)} valid pairs remaining."
        )

    # --- 5. Optionally display a couple of random image-mask pairs ---
    if display and len(valid_img_paths) > 0:
        indices = np.random.randint(
            low=0, high=len(valid_img_paths),
            size=(min(2, len(valid_img_paths)),)
        )
        for i in indices:
            mask_img = cv2.imread(valid_mask_paths[i])
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            # Binarise: any non-zero pixel becomes 255
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
    Split paired lists into train/val subsets.
    Useful when the dataset is not pre-split into separate folders.

    Args:
        img_paths  (list): All image paths
        mask_paths (list): Corresponding mask paths
        val_split  (float): Fraction of data used for validation (default 0.2)
        seed       (int):   Random seed for reproducibility

    Returns:
        Four lists: train_imgs, train_masks, val_imgs, val_masks
    """
    np.random.seed(seed)
    n = len(img_paths)
    indices = np.random.permutation(n)

    split_idx = int(n * (1 - val_split))
    train_idx = indices[:split_idx]
    val_idx   = indices[split_idx:]

    train_imgs  = [img_paths[i]  for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]
    val_imgs    = [img_paths[i]  for i in val_idx]
    val_masks   = [mask_paths[i] for i in val_idx]

    print(f"Split -> Train: {len(train_imgs)}, Val: {len(val_imgs)}")
    return train_imgs, train_masks, val_imgs, val_masks


class parseDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None, augment=None):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.transform  = transform
        self.augment    = augment
        print(f"Dataset initialised with {len(self.img_paths)} image-mask pairs")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # --- Load image (H x W x 3, uint8) ---
        image = cv2.imread(self.img_paths[idx])
        if image is None:
            raise FileNotFoundError(f"Could not read image: {self.img_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Load mask and binarise to {0, 255} (H x W, uint8) ---
        mask = cv2.imread(self.mask_paths[idx])
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {self.mask_paths[idx]}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # Threshold > 1 so any annotation pixel becomes 255
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # --- Optional albumentations augmentation ---
        # albumentations expects both image (H x W x C) and mask (H x W)
        if self.augment:
            augmented = self.augment(image=image, mask=mask)
            image     = augmented["image"]
            mask      = augmented["mask"]

        # --- ToTensor transform (applied separately to image and mask) ---
        if self.transform:
            image = self.transform(image)   # -> (3, H, W) float [0, 1]

            # transforms.ToTensor() expects HxWxC or HxW;
            # expand mask to HxWx1 so ToTensor gives (1, H, W)
            mask = self.transform(mask)     # -> (1, H, W) float [0, 1]

            # Normalise image channels only (not the mask)
            image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )(image)

        return image, mask


# ---------------------------------------------------------------------------
# Transforms & augmentations
# ---------------------------------------------------------------------------

# Basic transform: numpy array -> torch tensor
trans = transforms.Compose([
    transforms.ToTensor()   # converts uint8 HxWxC -> float32 CxHxW in [0,1]
])

aug = A.Compose([
    A.RandomScale(scale_limit=0.5, p=0.5),
    A.PadIfNeeded(min_height=224, min_width=224, p=1),
    A.RandomCrop(height=224, width=224, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.8),
    A.RandomGamma(p=0.8)
])


# ---------------------------------------------------------------------------
# Loss & training utilities (unchanged)
# ---------------------------------------------------------------------------

def calc_loss(pred, target, metrics, bce_weight=0.5):
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


def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

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
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = calc_loss(outputs, labels, metrics)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

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
        print("{:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60))
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
    # --- Paths for the VegAnn dataset ---
    IMAGES_DIR = "VegAnn_dataset/images"
    MASKS_DIR  = "VegAnn_dataset/annotations"

    print(f"Images directory : {IMAGES_DIR}  (exists: {os.path.exists(IMAGES_DIR)})")
    print(f"Masks  directory : {MASKS_DIR}   (exists: {os.path.exists(MASKS_DIR)})")

    # Load all paired paths from the two directories
    all_img_paths, all_mask_paths = read_imgs_and_masks(
        IMAGES_DIR, MASKS_DIR, display=False
    )

    if len(all_img_paths) == 0:
        raise ValueError(
            f"No valid image-mask pairs found. "
            f"Check that {IMAGES_DIR} and {MASKS_DIR} are correct."
        )

    # Split into train / validation sets (80 / 20 by default)
    train_img_paths, train_mask_paths, \
    val_img_paths,   val_mask_paths   = split_train_val(
        all_img_paths, all_mask_paths, val_split=0.2
    )

    # Build Dataset objects
    train_set = parseDataset(
        train_img_paths, train_mask_paths,
        transform=trans, augment=aug
    )
    val_set = parseDataset(
        val_img_paths, val_mask_paths,
        transform=trans            # no augmentation for validation
    )

    batch_size  = 25
    dataloaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0),
        "val":   DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0),
    }

    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    num_class = 1
    model     = ResNetUNet(num_class).to(device)

    optimizer_ft       = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    exp_lr_scheduler   = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=30)