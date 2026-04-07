TEST_DIR = "./dataset/DenseLeaves/test/"
SAVE_PATH = "./dataset/DenseLeaves/test/outputs"

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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels import robust
import statsmodels.api as sm
from statsmodels.stats import diagnostic
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
GENERATE_PLOTS = True  # Generate statistical plots
PERFORM_ADVANCED_STATS = True  # Perform advanced statistical analysis

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

def perform_statistical_analysis(coverages, save_path):
    """
    Perform comprehensive statistical analysis on foliage coverage data
    
    Args:
        coverages: List of coverage percentages
        save_path: Path to save analysis results
    """
    coverages = np.array(coverages)
    
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE STATISTICAL ANALYSIS OF FOLIAGE COVERAGE")
    print(f"{'='*70}")
    
    # ==================== DESCRIPTIVE STATISTICS ====================
    print(f"\n{'─'*70}")
    print("1. DESCRIPTIVE STATISTICS")
    print(f"{'─'*70}")
    
    desc_stats = {
        'Count': len(coverages),
        'Mean': np.mean(coverages),
        'Median': np.median(coverages),
        'Mode': stats.mode(coverages, keepdims=True)[0][0] if len(coverages) > 0 else np.nan,
        'Std Dev': np.std(coverages, ddof=1),
        'Variance': np.var(coverages, ddof=1),
        'Min': np.min(coverages),
        'Max': np.max(coverages),
        'Range': np.max(coverages) - np.min(coverages),
        'Q1 (25%)': np.percentile(coverages, 25),
        'Q2 (50%)': np.percentile(coverages, 50),
        'Q3 (75%)': np.percentile(coverages, 75),
        'IQR': np.percentile(coverages, 75) - np.percentile(coverages, 25),
        'Skewness': stats.skew(coverages),
        'Kurtosis': stats.kurtosis(coverages),
        'CV (Coefficient of Variation)': (np.std(coverages, ddof=1) / np.mean(coverages)) * 100 if np.mean(coverages) > 0 else np.nan,
        'Standard Error': stats.sem(coverages),
        'MAD (Median Absolute Deviation)': robust.mad(coverages),
    }
    
    for key, value in desc_stats.items():
        if isinstance(value, (int, np.integer)):
            print(f"  {key:<35}: {value}")
        else:
            print(f"  {key:<35}: {value:.4f}")
    
    # ==================== CONFIDENCE INTERVALS ====================
    print(f"\n{'─'*70}")
    print("2. CONFIDENCE INTERVALS")
    print(f"{'─'*70}")
    
    # 95% CI for mean
    ci_95 = stats.t.interval(0.95, len(coverages)-1, 
                             loc=np.mean(coverages), 
                             scale=stats.sem(coverages))
    print(f"  95% CI for Mean: ({ci_95[0]:.4f}, {ci_95[1]:.4f})")
    
    # 99% CI for mean
    ci_99 = stats.t.interval(0.99, len(coverages)-1, 
                             loc=np.mean(coverages), 
                             scale=stats.sem(coverages))
    print(f"  99% CI for Mean: ({ci_99[0]:.4f}, {ci_99[1]:.4f})")
    
    # Bootstrap CI (non-parametric)
    from scipy.stats import bootstrap
    rng = np.random.default_rng(42)
    res = bootstrap((coverages,), np.mean, n_resamples=10000, 
                    confidence_level=0.95, random_state=rng)
    print(f"  95% Bootstrap CI: ({res.confidence_interval.low:.4f}, {res.confidence_interval.high:.4f})")
    
    # ==================== NORMALITY TESTS ====================
    print(f"\n{'─'*70}")
    print("3. NORMALITY TESTS")
    print(f"{'─'*70}")
    
    # Shapiro-Wilk test
    if len(coverages) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(coverages)
        print(f"  Shapiro-Wilk Test:")
        print(f"    Statistic: {shapiro_stat:.4f}")
        print(f"    P-value: {shapiro_p:.4f}")
        print(f"    Result: {'Data appears NORMAL' if shapiro_p > 0.05 else 'Data appears NON-NORMAL'} (α=0.05)")
    
    # Kolmogorov-Smirnov test
    if len(coverages) >= 3:
        ks_stat, ks_p = stats.kstest(coverages, 'norm', 
                                     args=(np.mean(coverages), np.std(coverages, ddof=1)))
        print(f"\n  Kolmogorov-Smirnov Test:")
        print(f"    Statistic: {ks_stat:.4f}")
        print(f"    P-value: {ks_p:.4f}")
        print(f"    Result: {'Data appears NORMAL' if ks_p > 0.05 else 'Data appears NON-NORMAL'} (α=0.05)")
    
    # Anderson-Darling test
    if len(coverages) >= 3:
        anderson_result = stats.anderson(coverages, dist='norm')
        print(f"\n  Anderson-Darling Test:")
        print(f"    Statistic: {anderson_result.statistic:.4f}")
        for i, (sig_level, crit_val) in enumerate(zip(anderson_result.significance_level, 
                                                       anderson_result.critical_values)):
            result = "REJECT normality" if anderson_result.statistic > crit_val else "ACCEPT normality"
            print(f"    At {sig_level}% significance: {crit_val:.4f} -> {result}")
    
    # D'Agostino's K-squared test
    if len(coverages) >= 8:
        dagostino_stat, dagostino_p = stats.normaltest(coverages)
        print(f"\n  D'Agostino's K-squared Test:")
        print(f"    Statistic: {dagostino_stat:.4f}")
        print(f"    P-value: {dagostino_p:.4f}")
        print(f"    Result: {'Data appears NORMAL' if dagostino_p > 0.05 else 'Data appears NON-NORMAL'} (α=0.05)")
    
    # ==================== OUTLIER DETECTION ====================
    print(f"\n{'─'*70}")
    print("4. OUTLIER DETECTION")
    print(f"{'─'*70}")
    
    # IQR method
    Q1 = np.percentile(coverages, 25)
    Q3 = np.percentile(coverages, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = coverages[(coverages < lower_bound) | (coverages > upper_bound)]
    print(f"  IQR Method (1.5×IQR rule):")
    print(f"    Lower bound: {lower_bound:.4f}")
    print(f"    Upper bound: {upper_bound:.4f}")
    print(f"    Number of outliers: {len(outliers_iqr)}")
    if len(outliers_iqr) > 0:
        print(f"    Outlier values: {outliers_iqr}")
    
    # Z-score method
    z_scores = np.abs(stats.zscore(coverages))
    outliers_z = coverages[z_scores > 3]
    print(f"\n  Z-Score Method (|z| > 3):")
    print(f"    Number of outliers: {len(outliers_z)}")
    if len(outliers_z) > 0:
        print(f"    Outlier values: {outliers_z}")
    
    # Modified Z-score (using MAD)
    median = np.median(coverages)
    mad = robust.mad(coverages)
    modified_z_scores = 0.6745 * (coverages - median) / mad if mad > 0 else np.zeros_like(coverages)
    outliers_mad = coverages[np.abs(modified_z_scores) > 3.5]
    print(f"\n  Modified Z-Score Method (MAD-based, |z| > 3.5):")
    print(f"    Number of outliers: {len(outliers_mad)}")
    if len(outliers_mad) > 0:
        print(f"    Outlier values: {outliers_mad}")
    
    # ==================== DISTRIBUTION FITTING ====================
    print(f"\n{'─'*70}")
    print("5. DISTRIBUTION FITTING")
    print(f"{'─'*70}")
    
    # Fit various distributions and compare
    distributions = ['norm', 'lognorm', 'gamma', 'beta', 'weibull_min']
    best_dist = None
    best_aic = np.inf
    
    print(f"  Testing distributions (using AIC for comparison):")
    for dist_name in distributions:
        try:
            dist = getattr(stats, dist_name)
            params = dist.fit(coverages)
            
            # Calculate log-likelihood
            log_likelihood = np.sum(dist.logpdf(coverages, *params))
            
            # Calculate AIC (Akaike Information Criterion)
            k = len(params)  # number of parameters
            aic = 2 * k - 2 * log_likelihood
            
            print(f"    {dist_name:15s}: AIC = {aic:.4f}")
            
            if aic < best_aic:
                best_aic = aic
                best_dist = dist_name
        except:
            print(f"    {dist_name:15s}: Failed to fit")
    
    print(f"\n  Best fitting distribution: {best_dist} (lowest AIC)")
    
    # ==================== PERCENTILE ANALYSIS ====================
    print(f"\n{'─'*70}")
    print("6. PERCENTILE ANALYSIS")
    print(f"{'─'*70}")
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"  Percentile breakdown:")
    for p in percentiles:
        value = np.percentile(coverages, p)
        print(f"    {p:3d}th percentile: {value:7.4f}%")
    
    # ==================== HYPOTHESIS TESTS ====================
    print(f"\n{'─'*70}")
    print("7. HYPOTHESIS TESTS")
    print(f"{'─'*70}")
    
    # One-sample t-test (test if mean is significantly different from a hypothetical value)
    hypothetical_mean = 20.0  # You can change this
    t_stat, t_p = stats.ttest_1samp(coverages, hypothetical_mean)
    print(f"  One-sample t-test (H0: μ = {hypothetical_mean}%):")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    P-value: {t_p:.4f}")
    print(f"    Result: {'REJECT H0' if t_p < 0.05 else 'FAIL TO REJECT H0'} (α=0.05)")
    print(f"    Interpretation: Mean coverage is {'significantly different from' if t_p < 0.05 else 'not significantly different from'} {hypothetical_mean}%")
    
    # Test if median is significantly different from hypothetical value
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(coverages - hypothetical_mean)
    print(f"\n  Wilcoxon signed-rank test (H0: median = {hypothetical_mean}%):")
    print(f"    Statistic: {wilcoxon_stat:.4f}")
    print(f"    P-value: {wilcoxon_p:.4f}")
    print(f"    Result: {'REJECT H0' if wilcoxon_p < 0.05 else 'FAIL TO REJECT H0'} (α=0.05)")
    
    # ==================== SUMMARY REPORT ====================
    print(f"\n{'='*70}")
    print("SUMMARY INTERPRETATION")
    print(f"{'='*70}")
    
    print(f"\n  Dataset Characteristics:")
    print(f"    • Sample size: {len(coverages)} images")
    print(f"    • Average foliage coverage: {np.mean(coverages):.2f}% ± {np.std(coverages, ddof=1):.2f}%")
    print(f"    • Coverage ranges from {np.min(coverages):.2f}% to {np.max(coverages):.2f}%")
    print(f"    • Variability (CV): {desc_stats['CV (Coefficient of Variation)']:.2f}%")
    
    if desc_stats['Skewness'] > 0.5:
        print(f"    • Distribution is RIGHT-SKEWED (tail towards higher values)")
    elif desc_stats['Skewness'] < -0.5:
        print(f"    • Distribution is LEFT-SKEWED (tail towards lower values)")
    else:
        print(f"    • Distribution is approximately SYMMETRIC")
    
    if desc_stats['Kurtosis'] > 0:
        print(f"    • Distribution has HEAVY TAILS (more outliers than normal)")
    else:
        print(f"    • Distribution has LIGHT TAILS (fewer outliers than normal)")
    
    print(f"    • {len(outliers_iqr)} potential outliers detected")
    
    # Save detailed statistics to file
    stats_file = os.path.join(save_path, "statistical_analysis.txt")
    with open(stats_file, 'w') as f:
        f.write("FOLIAGE COVERAGE - DETAILED STATISTICAL ANALYSIS\n")
        f.write("="*70 + "\n\n")
        f.write("DESCRIPTIVE STATISTICS:\n")
        for key, value in desc_stats.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nBest fitting distribution: {best_dist}\n")
        f.write(f"Number of outliers (IQR method): {len(outliers_iqr)}\n")
    
    print(f"\n  Detailed statistics saved to: {stats_file}")
    
    return desc_stats

def generate_statistical_plots(coverages, all_metrics, save_path):
    """
    Generate comprehensive statistical visualizations
    """
    coverages = np.array(coverages)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Histogram with KDE
    ax1 = plt.subplot(3, 3, 1)
    sns.histplot(coverages, kde=True, bins=20, color='skyblue', edgecolor='black', ax=ax1)
    ax1.axvline(np.mean(coverages), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(coverages):.2f}%')
    ax1.axvline(np.median(coverages), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(coverages):.2f}%')
    ax1.set_xlabel('Foliage Coverage (%)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Distribution of Foliage Coverage', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2 = plt.subplot(3, 3, 2)
    box_parts = ax2.boxplot(coverages, vert=True, patch_artist=True, 
                            notch=True, showmeans=True,
                            boxprops=dict(facecolor='lightblue', alpha=0.7),
                            meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    ax2.set_ylabel('Foliage Coverage (%)', fontsize=10)
    ax2.set_title('Box Plot with Outliers', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Q-Q Plot
    ax3 = plt.subplot(3, 3, 3)
    stats.probplot(coverages, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Violin plot
    ax4 = plt.subplot(3, 3, 4)
    parts = ax4.violinplot([coverages], positions=[1], showmeans=True, showmedians=True, widths=0.7)
    for pc in parts['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)
    ax4.set_ylabel('Foliage Coverage (%)', fontsize=10)
    ax4.set_title('Violin Plot', fontsize=12, fontweight='bold')
    ax4.set_xticks([1])
    ax4.set_xticklabels(['Coverage'])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Cumulative Distribution
    ax5 = plt.subplot(3, 3, 5)
    sorted_coverages = np.sort(coverages)
    cumulative = np.arange(1, len(sorted_coverages) + 1) / len(sorted_coverages) * 100
    ax5.plot(sorted_coverages, cumulative, linewidth=2, color='darkblue')
    ax5.set_xlabel('Foliage Coverage (%)', fontsize=10)
    ax5.set_ylabel('Cumulative Probability (%)', fontsize=10)
    ax5.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(50, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    ax5.legend()
    
    # 6. Coverage by image index (time series style)
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(range(len(coverages)), coverages, marker='o', linestyle='-', linewidth=1, markersize=4)
    ax6.axhline(np.mean(coverages), color='red', linestyle='--', linewidth=2, alpha=0.7, label='Mean')
    ax6.fill_between(range(len(coverages)), 
                      np.mean(coverages) - np.std(coverages), 
                      np.mean(coverages) + np.std(coverages), 
                      alpha=0.2, color='red', label='±1 SD')
    ax6.set_xlabel('Image Index', fontsize=10)
    ax6.set_ylabel('Foliage Coverage (%)', fontsize=10)
    ax6.set_title('Coverage Across Images', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Percentile bars
    ax7 = plt.subplot(3, 3, 7)
    percentiles = [10, 25, 50, 75, 90]
    percentile_values = [np.percentile(coverages, p) for p in percentiles]
    bars = ax7.bar([str(p) for p in percentiles], percentile_values, color='teal', alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Percentile', fontsize=10)
    ax7.set_ylabel('Coverage (%)', fontsize=10)
    ax7.set_title('Coverage at Key Percentiles', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, value in zip(bars, percentile_values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 8. Density comparison with fitted normal
    ax8 = plt.subplot(3, 3, 8)
    sns.kdeplot(coverages, fill=True, color='blue', alpha=0.5, label='Actual', ax=ax8)
    # Fit and plot normal distribution
    mu, sigma = np.mean(coverages), np.std(coverages, ddof=1)
    x = np.linspace(coverages.min(), coverages.max(), 100)
    ax8.plot(x, stats.norm.pdf(x, mu, sigma) * len(coverages) * (coverages.max() - coverages.min()) / 20, 
             'r-', linewidth=2, label='Fitted Normal', alpha=0.7)
    ax8.set_xlabel('Foliage Coverage (%)', fontsize=10)
    ax8.set_ylabel('Density', fontsize=10)
    ax8.set_title('Actual vs Normal Distribution', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('tight')
    ax9.axis('off')
    
    summary_data = [
        ['Metric', 'Value'],
        ['Count', f'{len(coverages)}'],
        ['Mean', f'{np.mean(coverages):.2f}%'],
        ['Median', f'{np.median(coverages):.2f}%'],
        ['Std Dev', f'{np.std(coverages, ddof=1):.2f}%'],
        ['Min', f'{np.min(coverages):.2f}%'],
        ['Max', f'{np.max(coverages):.2f}%'],
        ['Range', f'{np.max(coverages) - np.min(coverages):.2f}%'],
        ['Skewness', f'{stats.skew(coverages):.3f}'],
        ['Kurtosis', f'{stats.kurtosis(coverages):.3f}'],
    ]
    
    table = ax9.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax9.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, "foliage_coverage_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Statistical plots saved to: {plot_path}")

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
        
        print(f"\n{'='*70}")
        print(f"Coverage metrics saved to: {csv_path}")
        print(f"{'='*70}")
        
        # Extract coverage percentages
        coverages = [m['coverage_percentage'] for m in all_metrics]
        
        # Perform advanced statistical analysis
        if PERFORM_ADVANCED_STATS and len(coverages) >= 3:
            desc_stats = perform_statistical_analysis(coverages, SAVE_PATH)
        
        # Generate statistical plots
        if GENERATE_PLOTS and len(coverages) >= 3:
            print(f"\nGenerating statistical visualizations...")
            generate_statistical_plots(coverages, all_metrics, SAVE_PATH)
    
    print(f"\n{'='*70}")
    print(f"Processing complete!")
    print(f"{'='*70}")
    print(f"Total images processed: {total_processed}/{len(test_img_paths)}")
    print(f"Results saved to: {SAVE_PATH}")
    print(f"\nGenerated files:")
    print(f"  • Segmented images: {PREFIX}[filename]")
    if SAVE_MASK_ONLY:
        print(f"  • Binary masks: [filename]_mask")
    if SAVE_OVERLAY:
        print(f"  • Overlay visualizations: [filename]_overlay")
    if SAVE_METRICS:
        print(f"  • Coverage metrics CSV: foliage_coverage_metrics.csv")
        print(f"  • Statistical analysis report: statistical_analysis.txt")
    if GENERATE_PLOTS:
        print(f"  • Statistical plots: foliage_coverage_analysis.png")