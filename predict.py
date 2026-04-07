import os
from imutils import paths
import imageio
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
from model import ResNetUNet
from utilities import reverse_transform, reverse_transform_mask
from preprocess import check_dir

WEIGHT_PATH = "./model/pretrained"
USE_BEST_VAL = True
DISPLAY_PLOTS = False
TEST_DIR = "./dataset/DenseLeaves/test/"
SAVE_PATH = "./dataset/DenseLeaves/test/outputs"
PREFIX = "seg_"
SAVE_MASK_ONLY = False  # Save mask visualization
SAVE_OVERLAY = False  # Save overlay visualization
SAVE_METRICS = True  # Save coverage metrics to CSV

# Important: Make sure SAVE_PATH is different from TEST_DIR!
if os.path.abspath(SAVE_PATH) == os.path.abspath(TEST_DIR):
    print("ERROR: SAVE_PATH cannot be the same as TEST_DIR!")
    print("This will cause recursive processing of output files.")
    SAVE_PATH = os.path.join(TEST_DIR, "output")
    print(f"Changed SAVE_PATH to: {SAVE_PATH}")

trans = transforms.Compose([
    transforms.ToTensor()
])

class parseTestset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform
        print(f"Test dataset initialized with {len(self.img_paths)} images")
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        _, filename = os.path.split(image_path)
        if self.transform:
            image = self.transform(image)  # ToTensor
            image = transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])(image)
        return image, filename

def filter_image_paths(img_paths):
    """Filter out mask files, overlay files, and already processed files"""
    filtered = []
    excluded = []
    
    for path in img_paths:
        filename = os.path.basename(path)
        
        should_skip = any([
            '_seg.' in filename,
            filename.startswith('seg_'),
            '_mask.' in filename,
            '_overlay.' in filename,
            'mask' in filename.lower(),
            'overlay' in filename.lower(),
        ])
        
        if should_skip:
            excluded.append(filename)
            continue
            
        filtered.append(path)
    
    if excluded:
        print(f"\nExcluded {len(excluded)} processed/mask/overlay files:")
        for f in excluded[:10]:
            print(f"  - {f}")
        if len(excluded) > 10:
            print(f"  ... and {len(excluded) - 10} more")
    
    return filtered

def calculate_coverage(seg_binary):
    """
    Calculate the percentage of image covered by leaves
    
    Args:
        seg_binary: Binary segmentation mask (2D array with 0s and 1s)
    
    Returns:
        dict with coverage metrics
    """
    total_pixels = seg_binary.size
    leaf_pixels = np.sum(seg_binary)
    background_pixels = total_pixels - leaf_pixels
    
    coverage_percentage = (leaf_pixels / total_pixels) * 100
    
    return {
        'total_pixels': total_pixels,
        'leaf_pixels': int(leaf_pixels),
        'background_pixels': int(background_pixels),
        'coverage_percentage': coverage_percentage
    }

if __name__ == "__main__":
    # Check if test directory exists
    print(f"Checking test directory: {TEST_DIR}")
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Test directory does not exist: {TEST_DIR}")
    print(f"Test directory exists: {os.path.exists(TEST_DIR)}")
    
    # Check if weight file exists
    weight_file = "best_val_weights.pth" if USE_BEST_VAL else "latest_weights.pth"
    weight_path = os.path.join(WEIGHT_PATH, weight_file)
    print(f"Checking weight file: {weight_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file does not exist: {weight_path}")
    print(f"Weight file exists: {os.path.exists(weight_path)}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'running on: {device}')
    num_class = 1
    model = ResNetUNet(num_class).to(device)
    
    # Load weights
    print(f"Loading weights from: {weight_file}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    
    # Get test image paths
    all_img_paths = list(paths.list_images(TEST_DIR))
    print(f'Found {len(all_img_paths)} total files in {TEST_DIR}')
    
    # Filter out mask and segmentation files
    test_img_paths = filter_image_paths(all_img_paths)
    print(f'\nFiltered to {len(test_img_paths)} actual images for processing')
    
    if len(test_img_paths) == 0:
        print(f"Warning: No valid images found in {TEST_DIR}")
        print(f"All files might be processed outputs or masks.")
        print(f"\nLooking for files ending with: .jpg, .jpeg, .png, .bmp")
        print(f"Excluding files containing: seg_, _seg., _mask., _overlay.")
        raise ValueError(f"No test images found in {TEST_DIR}")
    
    # Print first few image paths for debugging
    print(f"\nSample image paths to process:")
    for i, path in enumerate(test_img_paths[:5]):
        print(f"  {i+1}. {os.path.basename(path)}")
    
    # small batch_size if you are testing on 1 or 2 images
    b_size = min(25, len(test_img_paths))
    print(f"\nUsing batch size: {b_size}")
    
    test_set = parseTestset(test_img_paths, transform=trans)
    test_loader = DataLoader(test_set, batch_size=b_size,
                             shuffle=False, num_workers=0)
    
    # Create output directory
    check_dir(SAVE_PATH)
    print(f"Output will be saved to: {SAVE_PATH}")
    print(f"NOTE: Output directory is separate from input to avoid recursive processing\n")
    
    model.eval()
    total_processed = 0
    
    # Store metrics for all images
    all_metrics = []
    
    with torch.no_grad():
        for i, batch_pair in enumerate(tqdm(test_loader, desc="Processing images")):
            img_batch = batch_pair[0].to(device)
            img_names = batch_pair[1]
            seg_batch = model(img_batch)
            seg_batch = torch.sigmoid(seg_batch)
            
            for img, seg, filename in zip(img_batch, seg_batch, img_names):
                # Get segmentation values
                seg_np = seg.cpu().detach().numpy()
                
                # Use a lower threshold based on your data
                threshold = 0.3  # Adjusted based on your max values (0.4-0.7 range)
                seg_binary = np.where(seg_np[0] > threshold, 1, 0)
                
                # Calculate coverage metrics
                metrics = calculate_coverage(seg_binary)
                metrics['filename'] = filename
                metrics['threshold'] = threshold
                all_metrics.append(metrics)
                
                # Get original image
                img_np = img.cpu()
                img_np = reverse_transform(img_np)
                
                # Ensure mask has correct dimensions
                if len(seg_binary.shape) == 2:
                    seg_binary_3d = np.expand_dims(seg_binary, axis=-1)
                else:
                    seg_binary_3d = seg_binary
                
                # Create masked image (leaf only, black background)
                prod_img = np.multiply(seg_binary_3d, img_np).astype("uint8")
                
                # Prepare filenames
                base_filename = os.path.splitext(filename)[0]
                ext = os.path.splitext(filename)[1]
                
                try:
                    # 1. Save segmented leaf (main output)
                    if len(PREFIX) > 0:
                        seg_filename = PREFIX + filename
                    else:
                        seg_filename = f"{base_filename}_segmented{ext}"
                    seg_savename = os.path.join(SAVE_PATH, seg_filename)
                    imageio.imwrite(seg_savename, prod_img)
                    
                    # 2. Save binary mask (optional)
                    if SAVE_MASK_ONLY:
                        mask_filename = f"{base_filename}_mask{ext}"
                        mask_savename = os.path.join(SAVE_PATH, mask_filename)
                        mask_visual = (seg_binary * 255).astype("uint8")
                        imageio.imwrite(mask_savename, mask_visual)
                    
                    # 3. Save overlay visualization (optional)
                    if SAVE_OVERLAY:
                        overlay_filename = f"{base_filename}_overlay{ext}"
                        overlay_savename = os.path.join(SAVE_PATH, overlay_filename)
                        overlay = img_np.copy().astype(float)
                        # Create red overlay where mask is positive
                        red_overlay = np.zeros_like(img_np)
                        red_overlay[:, :, 0] = 255  # Red channel
                        # Blend original image with red overlay
                        mask_3d = np.repeat(seg_binary[:, :, np.newaxis], 3, axis=2)
                        overlay = np.where(mask_3d, 0.6 * img_np + 0.4 * red_overlay, img_np)
                        imageio.imwrite(overlay_savename, overlay.astype("uint8"))
                    
                    total_processed += 1
                except Exception as e:
                    print(f"Error saving {filename}: {e}")
    
    # Save metrics to CSV
    if SAVE_METRICS and all_metrics:
        csv_path = os.path.join(SAVE_PATH, "foliage_coverage_metrics.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'coverage_percentage', 'leaf_pixels', 'background_pixels', 
                         'total_pixels', 'threshold']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for metrics in all_metrics:
                writer.writerow(metrics)
        
        print(f"\n{'='*60}")
        print(f"Coverage metrics saved to: {csv_path}")
        print(f"{'='*60}")
        
        # Print summary statistics
        coverages = [m['coverage_percentage'] for m in all_metrics]
        print(f"\nFoliage Coverage Statistics:")
        print(f"  Total images processed: {len(all_metrics)}")
        print(f"  Average coverage: {np.mean(coverages):.2f}%")
        print(f"  Min coverage: {np.min(coverages):.2f}%")
        print(f"  Max coverage: {np.max(coverages):.2f}%")
        print(f"  Median coverage: {np.median(coverages):.2f}%")
        
        # Show top 5 and bottom 5
        sorted_metrics = sorted(all_metrics, key=lambda x: x['coverage_percentage'], reverse=True)
        print(f"\nTop 5 images with highest foliage coverage:")
        for i, m in enumerate(sorted_metrics[:5], 1):
            print(f"  {i}. {m['filename']}: {m['coverage_percentage']:.2f}%")
        
        print(f"\nBottom 5 images with lowest foliage coverage:")
        for i, m in enumerate(sorted_metrics[-5:], 1):
            print(f"  {i}. {m['filename']}: {m['coverage_percentage']:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Total images processed: {total_processed}/{len(test_img_paths)}")
    print(f"Results saved to: {SAVE_PATH}")
    print(f"\nOutput files for each input image:")
    print(f"  1. {PREFIX}[original_name]{ext} - Leaf only (black background)")
    if SAVE_MASK_ONLY:
        print(f"  2. [original_name]_mask{ext} - Binary mask (white/black)")
    if SAVE_OVERLAY:
        print(f"  3. [original_name]_overlay{ext} - Original with red highlight")
    if SAVE_METRICS:
        print(f"  4. foliage_coverage_metrics.csv - Coverage percentages for all images")