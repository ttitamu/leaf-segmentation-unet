import os
import glob
from imutils import paths
import imageio
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels import robust
from model import ResNetUNet
from utilities import reverse_transform, reverse_transform_mask
from preprocess import check_dir

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WEIGHT_PATH = "./model/pretrained"
USE_BEST_VAL = True
DISPLAY_PLOTS = False

# ---------------------------------------------------------------------------
# Build all paths relative to THIS script file, not the working directory.
# This ensures the script works regardless of where you launch it from.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# VegAnn dataset paths - adjust these folder names to match your actual layout
IMAGES_DIR = os.path.join(SCRIPT_DIR, "dataset", "VegAnn_dataset", "images")
MASKS_DIR  = os.path.join(SCRIPT_DIR, "dataset", "VegAnn_dataset", "annotations")
SAVE_PATH  = os.path.join(SCRIPT_DIR, "dataset", "VegAnn_dataset", "outputs")

# Safety check: output must not overwrite the input images
if os.path.abspath(SAVE_PATH) == os.path.abspath(IMAGES_DIR):
    print("ERROR: SAVE_PATH cannot be the same as IMAGES_DIR!")
    SAVE_PATH = os.path.join(IMAGES_DIR, "output")
    print(f"Changed SAVE_PATH to: {SAVE_PATH}")

PREFIX                 = "seg_"
SAVE_MASK_ONLY         = False
SAVE_OVERLAY           = False
SAVE_METRICS           = True
GENERATE_PLOTS         = True
PERFORM_ADVANCED_STATS = True

# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------
trans = transforms.Compose([
    transforms.ToTensor()
])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class parseTestset(Dataset):
    """
    Loads raw images from IMAGES_DIR for inference.
    No masks are loaded here — the model predicts them.
    """
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform
        print(f"Test dataset initialised with {len(self.img_paths)} images")

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
            image = self.transform(image)
            image = transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )(image)

        return image, filename


# ---------------------------------------------------------------------------
# Path collection
# ---------------------------------------------------------------------------
def collect_vegann_image_paths(images_dir):
    """
    Collect all .png / .jpg image paths from the VegAnn images directory.
    Skips any files that look like previous outputs (seg_, _mask, _overlay).

    Args:
        images_dir (str): Path to the folder containing raw images.

    Returns:
        list[str]: Sorted list of valid image paths.
    """
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    raw_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpg"))
    )

    print(f"Found {len(raw_paths)} files in {images_dir}")

    valid, skipped = [], []
    for p in raw_paths:
        fname = os.path.basename(p)
        is_output = any([
            fname.startswith("seg_"),
            "_mask."      in fname,
            "_overlay."   in fname,
            "_segmented." in fname,
        ])
        if is_output:
            skipped.append(fname)
        else:
            valid.append(p)

    if skipped:
        print(f"Skipped {len(skipped)} output/processed file(s):")
        for f in skipped[:10]:
            print(f"  - {f}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    print(f"Valid images for inference: {len(valid)}")
    return valid


# ---------------------------------------------------------------------------
# Coverage metric
# ---------------------------------------------------------------------------
def calculate_coverage(seg_binary):
    """
    Calculate the percentage of the image covered by vegetation.

    Args:
        seg_binary: 2-D binary array (0 = background, 1 = vegetation)

    Returns:
        dict with pixel counts and coverage percentage
    """
    total_pixels      = seg_binary.size
    leaf_pixels       = int(np.sum(seg_binary))
    background_pixels = total_pixels - leaf_pixels
    coverage_pct      = (leaf_pixels / total_pixels) * 100

    return {
        'total_pixels':        total_pixels,
        'leaf_pixels':         leaf_pixels,
        'background_pixels':   background_pixels,
        'coverage_percentage': coverage_pct,
    }


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------
def perform_statistical_analysis(coverages, save_path):
    """
    Perform comprehensive statistical analysis on foliage coverage data.

    Args:
        coverages (list): Coverage percentages for every processed image.
        save_path (str):  Directory to write the text report into.

    Returns:
        dict: Descriptive statistics.
    """
    coverages = np.array(coverages)

    print(f"\n{'='*70}")
    print("COMPREHENSIVE STATISTICAL ANALYSIS OF FOLIAGE COVERAGE")
    print(f"{'='*70}")

    # 1. Descriptive statistics
    print(f"\n{'─'*70}")
    print("1. DESCRIPTIVE STATISTICS")
    print(f"{'─'*70}")

    desc_stats = {
        'Count':                           len(coverages),
        'Mean':                            np.mean(coverages),
        'Median':                          np.median(coverages),
        'Mode':                            stats.mode(coverages, keepdims=True)[0][0],
        'Std Dev':                         np.std(coverages, ddof=1),
        'Variance':                        np.var(coverages, ddof=1),
        'Min':                             np.min(coverages),
        'Max':                             np.max(coverages),
        'Range':                           np.max(coverages) - np.min(coverages),
        'Q1 (25%)':                        np.percentile(coverages, 25),
        'Q2 (50%)':                        np.percentile(coverages, 50),
        'Q3 (75%)':                        np.percentile(coverages, 75),
        'IQR':                             np.percentile(coverages, 75) - np.percentile(coverages, 25),
        'Skewness':                        stats.skew(coverages),
        'Kurtosis':                        stats.kurtosis(coverages),
        'CV (Coefficient of Variation)':   (np.std(coverages, ddof=1) / np.mean(coverages)) * 100
                                           if np.mean(coverages) > 0 else np.nan,
        'Standard Error':                  stats.sem(coverages),
        'MAD (Median Absolute Deviation)': robust.mad(coverages),
    }

    for key, value in desc_stats.items():
        if isinstance(value, (int, np.integer)):
            print(f"  {key:<35}: {value}")
        else:
            print(f"  {key:<35}: {value:.4f}")

    # 2. Confidence intervals
    print(f"\n{'─'*70}")
    print("2. CONFIDENCE INTERVALS")
    print(f"{'─'*70}")

    ci_95 = stats.t.interval(0.95, len(coverages) - 1,
                              loc=np.mean(coverages),
                              scale=stats.sem(coverages))
    ci_99 = stats.t.interval(0.99, len(coverages) - 1,
                              loc=np.mean(coverages),
                              scale=stats.sem(coverages))
    print(f"  95% CI for Mean : ({ci_95[0]:.4f}, {ci_95[1]:.4f})")
    print(f"  99% CI for Mean : ({ci_99[0]:.4f}, {ci_99[1]:.4f})")

    from scipy.stats import bootstrap
    rng = np.random.default_rng(42)
    res = bootstrap(
        (coverages,), np.mean,
        n_resamples=10000, confidence_level=0.95, random_state=rng
    )
    print(f"  95% Bootstrap CI: ({res.confidence_interval.low:.4f}, "
          f"{res.confidence_interval.high:.4f})")

    # 3. Normality tests
    print(f"\n{'─'*70}")
    print("3. NORMALITY TESTS")
    print(f"{'─'*70}")

    if len(coverages) >= 3:
        sw_stat, sw_p = stats.shapiro(coverages)
        print(f"  Shapiro-Wilk Test:")
        print(f"    Statistic : {sw_stat:.4f}")
        print(f"    P-value   : {sw_p:.4f}")
        print(f"    Result    : {'NORMAL' if sw_p > 0.05 else 'NON-NORMAL'} (α=0.05)")

        ks_stat, ks_p = stats.kstest(
            coverages, 'norm',
            args=(np.mean(coverages), np.std(coverages, ddof=1))
        )
        print(f"\n  Kolmogorov-Smirnov Test:")
        print(f"    Statistic : {ks_stat:.4f}")
        print(f"    P-value   : {ks_p:.4f}")
        print(f"    Result    : {'NORMAL' if ks_p > 0.05 else 'NON-NORMAL'} (α=0.05)")

        ad = stats.anderson(coverages, dist='norm')
        print(f"\n  Anderson-Darling Test:")
        print(f"    Statistic : {ad.statistic:.4f}")
        for sig, crit in zip(ad.significance_level, ad.critical_values):
            verdict = "REJECT" if ad.statistic > crit else "ACCEPT"
            print(f"    At {sig}%: critical={crit:.4f} -> {verdict} normality")

    if len(coverages) >= 8:
        dag_stat, dag_p = stats.normaltest(coverages)
        print(f"\n  D'Agostino's K-squared Test:")
        print(f"    Statistic : {dag_stat:.4f}")
        print(f"    P-value   : {dag_p:.4f}")
        print(f"    Result    : {'NORMAL' if dag_p > 0.05 else 'NON-NORMAL'} (α=0.05)")

    # 4. Outlier detection
    print(f"\n{'─'*70}")
    print("4. OUTLIER DETECTION")
    print(f"{'─'*70}")

    Q1, Q3  = np.percentile(coverages, 25), np.percentile(coverages, 75)
    IQR     = Q3 - Q1
    lb, ub  = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    out_iqr = coverages[(coverages < lb) | (coverages > ub)]
    print(f"  IQR Method (1.5×IQR):")
    print(f"    Lower bound : {lb:.4f}")
    print(f"    Upper bound : {ub:.4f}")
    print(f"    Outliers    : {len(out_iqr)}")
    if len(out_iqr):
        print(f"    Values      : {out_iqr}")

    z_scores = np.abs(stats.zscore(coverages))
    out_z    = coverages[z_scores > 3]
    print(f"\n  Z-Score Method (|z| > 3):")
    print(f"    Outliers : {len(out_z)}")
    if len(out_z):
        print(f"    Values   : {out_z}")

    median  = np.median(coverages)
    mad     = robust.mad(coverages)
    mod_z   = (0.6745 * (coverages - median) / mad) if mad > 0 else np.zeros_like(coverages)
    out_mad = coverages[np.abs(mod_z) > 3.5]
    print(f"\n  Modified Z-Score (MAD, |z| > 3.5):")
    print(f"    Outliers : {len(out_mad)}")
    if len(out_mad):
        print(f"    Values   : {out_mad}")

    # 5. Distribution fitting
    print(f"\n{'─'*70}")
    print("5. DISTRIBUTION FITTING")
    print(f"{'─'*70}")

    best_dist, best_aic = None, np.inf
    for dname in ['norm', 'lognorm', 'gamma', 'beta', 'weibull_min']:
        try:
            d      = getattr(stats, dname)
            params = d.fit(coverages)
            ll     = np.sum(d.logpdf(coverages, *params))
            aic    = 2 * len(params) - 2 * ll
            print(f"    {dname:15s}: AIC = {aic:.4f}")
            if aic < best_aic:
                best_aic, best_dist = aic, dname
        except Exception:
            print(f"    {dname:15s}: failed to fit")
    print(f"\n  Best fitting distribution: {best_dist}")

    # 6. Percentile analysis
    print(f"\n{'─'*70}")
    print("6. PERCENTILE ANALYSIS")
    print(f"{'─'*70}")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    {p:3d}th percentile: {np.percentile(coverages, p):7.4f}%")

    # 7. Hypothesis tests
    print(f"\n{'─'*70}")
    print("7. HYPOTHESIS TESTS")
    print(f"{'─'*70}")

    hyp_mean    = 20.0
    t_stat, t_p = stats.ttest_1samp(coverages, hyp_mean)
    print(f"  One-sample t-test (H0: μ = {hyp_mean}%):")
    print(f"    t-statistic : {t_stat:.4f}")
    print(f"    P-value     : {t_p:.4f}")
    print(f"    Result      : {'REJECT H0' if t_p < 0.05 else 'FAIL TO REJECT H0'} (α=0.05)")

    w_stat, w_p = stats.wilcoxon(coverages - hyp_mean)
    print(f"\n  Wilcoxon signed-rank test (H0: median = {hyp_mean}%):")
    print(f"    Statistic   : {w_stat:.4f}")
    print(f"    P-value     : {w_p:.4f}")
    print(f"    Result      : {'REJECT H0' if w_p < 0.05 else 'FAIL TO REJECT H0'} (α=0.05)")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY INTERPRETATION")
    print(f"{'='*70}")
    print(f"  Sample size : {len(coverages)}")
    print(f"  Mean ± SD   : {np.mean(coverages):.2f}% ± {np.std(coverages, ddof=1):.2f}%")
    print(f"  Range       : {np.min(coverages):.2f}% – {np.max(coverages):.2f}%")
    print(f"  CV          : {desc_stats['CV (Coefficient of Variation)']:.2f}%")
    skew = desc_stats['Skewness']
    if skew > 0.5:
        print("  Distribution: RIGHT-SKEWED (tail towards higher values)")
    elif skew < -0.5:
        print("  Distribution: LEFT-SKEWED (tail towards lower values)")
    else:
        print("  Distribution: approximately SYMMETRIC")
    kurt = desc_stats['Kurtosis']
    print(f"  Kurtosis    : {'HEAVY TAILS' if kurt > 0 else 'LIGHT TAILS'}")
    print(f"  Outliers    : {len(out_iqr)} (IQR method)")
    print(f"  Best dist.  : {best_dist}")

    # Save report
    stats_file = os.path.join(save_path, "statistical_analysis.txt")
    with open(stats_file, 'w') as f:
        f.write("VEGANN FOLIAGE COVERAGE — DETAILED STATISTICAL ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write("DESCRIPTIVE STATISTICS:\n")
        for key, value in desc_stats.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nBest fitting distribution : {best_dist}\n")
        f.write(f"Outliers (IQR method)     : {len(out_iqr)}\n")

    print(f"\n  Detailed statistics saved to: {stats_file}")
    return desc_stats


# ---------------------------------------------------------------------------
# Statistical plots
# ---------------------------------------------------------------------------
def generate_statistical_plots(coverages, all_metrics, save_path):
    """
    Generate 9 statistical subplots and save as a single PNG.

    Args:
        coverages   (list): Coverage percentages.
        all_metrics (list): Full per-image metric dicts.
        save_path   (str):  Directory to write the PNG into.
    """
    coverages = np.array(coverages)
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 12))

    # ── 1. Histogram + KDE ───────────────────────────────────────────────
    ax1 = plt.subplot(3, 3, 1)
    sns.histplot(coverages, kde=True, bins=20,
                 color='skyblue', edgecolor='black', ax=ax1)
    ax1.axvline(np.mean(coverages),   color='red',   linestyle='--', lw=2,
                label=f'Mean: {np.mean(coverages):.2f}%')
    ax1.axvline(np.median(coverages), color='green', linestyle='--', lw=2,
                label=f'Median: {np.median(coverages):.2f}%')
    ax1.set_xlabel('Foliage Coverage (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Foliage Coverage', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── 2. Box plot ──────────────────────────────────────────────────────
    ax2 = plt.subplot(3, 3, 2)
    ax2.boxplot(
        coverages,
        vert=True,
        patch_artist=True,
        notch=True,
        showmeans=True,
        boxprops=dict(facecolor='lightblue', alpha=0.7),
        meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
    )
    ax2.set_ylabel('Foliage Coverage (%)')
    ax2.set_title('Box Plot with Outliers', fontweight='bold')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['Coverage'])
    ax2.grid(True, alpha=0.3, axis='y')

    # ── 3. Q-Q plot ──────────────────────────────────────────────────────
    ax3 = plt.subplot(3, 3, 3)
    stats.probplot(coverages, dist='norm', plot=ax3)
    ax3.set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # ── 4. Violin plot ───────────────────────────────────────────────────
    ax4 = plt.subplot(3, 3, 4)
    parts = ax4.violinplot(
        [coverages], positions=[1],
        showmeans=True, showmedians=True, widths=0.7
    )
    for pc in parts['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)
    ax4.set_ylabel('Foliage Coverage (%)')
    ax4.set_title('Violin Plot', fontweight='bold')
    ax4.set_xticks([1])
    ax4.set_xticklabels(['Coverage'])
    ax4.grid(True, alpha=0.3, axis='y')

    # ── 5. Cumulative distribution ───────────────────────────────────────
    ax5 = plt.subplot(3, 3, 5)
    sorted_cov = np.sort(coverages)
    cumulative = np.arange(1, len(sorted_cov) + 1) / len(sorted_cov) * 100
    ax5.plot(sorted_cov, cumulative, linewidth=2, color='darkblue')
    ax5.axhline(50, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    ax5.set_xlabel('Foliage Coverage (%)')
    ax5.set_ylabel('Cumulative Probability (%)')
    ax5.set_title('Cumulative Distribution Function', fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── 6. Coverage across images (time-series style) ────────────────────
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(range(len(coverages)), coverages,
             marker='o', linestyle='-', linewidth=1, markersize=4)
    ax6.axhline(np.mean(coverages), color='red', linestyle='--', lw=2,
                alpha=0.7, label='Mean')
    ax6.fill_between(
        range(len(coverages)),
        np.mean(coverages) - np.std(coverages),
        np.mean(coverages) + np.std(coverages),
        alpha=0.2, color='red', label='±1 SD'
    )
    ax6.set_xlabel('Image Index')
    ax6.set_ylabel('Foliage Coverage (%)')
    ax6.set_title('Coverage Across Images', fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # ── 7. Key-percentile bar chart ──────────────────────────────────────
    ax7 = plt.subplot(3, 3, 7)
    pcts       = [10, 25, 50, 75, 90]
    pct_values = [np.percentile(coverages, p) for p in pcts]
    bars = ax7.bar(
        [str(p) for p in pcts], pct_values,
        color='teal', alpha=0.7, edgecolor='black'
    )
    for bar, value in zip(bars, pct_values):
        ax7.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f'{value:.1f}%',
            ha='center', va='bottom', fontsize=9
        )
    ax7.set_xlabel('Percentile')
    ax7.set_ylabel('Coverage (%)')
    ax7.set_title('Coverage at Key Percentiles', fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')

    # ── 8. Actual density vs fitted normal ───────────────────────────────
    ax8 = plt.subplot(3, 3, 8)
    sns.kdeplot(coverages, fill=True, color='blue', alpha=0.5,
                label='Actual', ax=ax8)
    mu, sigma = np.mean(coverages), np.std(coverages, ddof=1)
    x = np.linspace(coverages.min(), coverages.max(), 200)
    # Scale the normal PDF so its area matches the KDE
    ax8.plot(x, stats.norm.pdf(x, mu, sigma),
             'r-', linewidth=2, label='Fitted Normal', alpha=0.7)
    ax8.set_xlabel('Foliage Coverage (%)')
    ax8.set_ylabel('Density')
    ax8.set_title('Actual vs Normal Distribution', fontweight='bold')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # ── 9. Summary statistics table ──────────────────────────────────────
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('tight')
    ax9.axis('off')

    table_data = [
        ['Metric',    'Value'],
        ['Count',     f'{len(coverages)}'],
        ['Mean',      f'{np.mean(coverages):.2f}%'],
        ['Median',    f'{np.median(coverages):.2f}%'],
        ['Std Dev',   f'{np.std(coverages, ddof=1):.2f}%'],
        ['Min',       f'{np.min(coverages):.2f}%'],
        ['Max',       f'{np.max(coverages):.2f}%'],
        ['Range',     f'{np.max(coverages) - np.min(coverages):.2f}%'],
        ['Skewness',  f'{stats.skew(coverages):.3f}'],
        ['Kurtosis',  f'{stats.kurtosis(coverages):.3f}'],
    ]

    tbl = ax9.table(
        cellText=table_data,
        cellLoc='left',
        loc='center',
        colWidths=[0.5, 0.5]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2)

    # Style header row
    for col in range(2):
        tbl[(0, col)].set_facecolor('#4CAF50')
        tbl[(0, col)].set_text_props(weight='bold', color='white')

    # Alternate row shading
    for row in range(1, len(table_data)):
        for col in range(2):
            if row % 2 == 0:
                tbl[(row, col)].set_facecolor('#f0f0f0')

    ax9.set_title('Summary Statistics', fontweight='bold', pad=20)

    plt.tight_layout()
    plot_path = os.path.join(save_path, "foliage_coverage_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Statistical plots saved to: {plot_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # ── Directory checks ─────────────────────────────────────────────────
    print(f"Script location   : {SCRIPT_DIR}")
    print(f"Images directory  : {IMAGES_DIR}")
    print(f"  exists          : {os.path.exists(IMAGES_DIR)}")
    print(f"Masks  directory  : {MASKS_DIR}")
    print(f"  exists          : {os.path.exists(MASKS_DIR)}")

    # Print the actual contents of SCRIPT_DIR to help diagnose path issues
    print(f"\nContents of script directory ({SCRIPT_DIR}):")
    for item in sorted(os.listdir(SCRIPT_DIR)):
        item_path = os.path.join(SCRIPT_DIR, item)
        kind = "DIR " if os.path.isdir(item_path) else "FILE"
        print(f"  [{kind}] {item}")

    if not os.path.exists(IMAGES_DIR):
        # Check if VegAnn_dataset folder exists at all and show its contents
        vegann_root = os.path.join(SCRIPT_DIR, "VegAnn_dataset")
        if os.path.exists(vegann_root):
            print(f"\nVegAnn_dataset folder found but 'images' subfolder is missing.")
            print(f"Contents of {vegann_root}:")
            for item in sorted(os.listdir(vegann_root)):
                print(f"  {item}")
            print(
                f"\nPlease update IMAGES_DIR in the config section to match "
                f"one of the folders listed above."
            )
        else:
            print(f"\nVegAnn_dataset folder not found at: {vegann_root}")
            print(
                "Please either:\n"
                "  1. Move your dataset into the script directory, OR\n"
                "  2. Update IMAGES_DIR in the config section to the full "
                "absolute path of your images folder.\n"
                "  Example:\n"
                "    IMAGES_DIR = r'C:\\Users\\YourName\\Downloads\\VegAnn_dataset\\images'"
            )
        raise FileNotFoundError(
            f"Images directory does not exist: {IMAGES_DIR}"
        )

    # ── Weight file check ────────────────────────────────────────────────
    weight_file = "best_val_weights.pth" if USE_BEST_VAL else "latest_weights.pth"
    weight_path = os.path.join(WEIGHT_PATH, weight_file)
    print(f"\nChecking weight file: {weight_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"Weight file does not exist: {weight_path}"
        )
    print(f"  Weight file found: {weight_file}")

    # ── Device & model ───────────────────────────────────────────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on: {device}")

    num_class = 1
    model = ResNetUNet(num_class).to(device)
    print(f"Loading weights from: {weight_file}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # ── Collect image paths ──────────────────────────────────────────────
    test_img_paths = collect_vegann_image_paths(IMAGES_DIR)

    if len(test_img_paths) == 0:
        raise ValueError(
            f"No valid images found in {IMAGES_DIR}. "
            "Check that the folder contains .png or .jpg files."
        )

    print(f"\nSample paths to process:")
    for i, p in enumerate(test_img_paths[:5]):
        print(f"  {i + 1}. {os.path.basename(p)}")

    # ── DataLoader ───────────────────────────────────────────────────────
    b_size      = min(25, len(test_img_paths))
    print(f"\nUsing batch size: {b_size}")

    test_set    = parseTestset(test_img_paths, transform=trans)
    test_loader = DataLoader(
        test_set, batch_size=b_size, shuffle=False, num_workers=0
    )

    # ── Output directory ─────────────────────────────────────────────────
    check_dir(SAVE_PATH)
    print(f"Output will be saved to: {SAVE_PATH}\n")

    # ── Inference loop ───────────────────────────────────────────────────
    total_processed = 0
    all_metrics     = []

    with torch.no_grad():
        for batch_pair in tqdm(test_loader, desc="Processing images"):
            img_batch  = batch_pair[0].to(device)
            img_names  = batch_pair[1]
            seg_batch  = model(img_batch)
            seg_batch  = torch.sigmoid(seg_batch)

            for img, seg, filename in zip(img_batch, seg_batch, img_names):
                seg_np     = seg.cpu().detach().numpy()
                threshold  = 0.3
                seg_binary = np.where(seg_np[0] > threshold, 1, 0)

                metrics             = calculate_coverage(seg_binary)
                metrics['filename'] = filename
                metrics['threshold']= threshold
                all_metrics.append(metrics)

                img_np        = reverse_transform(img.cpu())
                seg_binary_3d = np.expand_dims(seg_binary, axis=-1)
                prod_img      = np.multiply(seg_binary_3d, img_np).astype("uint8")

                base_name = os.path.splitext(filename)[0]
                ext       = os.path.splitext(filename)[1]

                try:
                    seg_filename = PREFIX + filename if PREFIX else f"{base_name}_segmented{ext}"
                    imageio.imwrite(os.path.join(SAVE_PATH, seg_filename), prod_img)

                    if SAVE_MASK_ONLY:
                        mask_visual = (seg_binary * 255).astype("uint8")
                        imageio.imwrite(
                            os.path.join(SAVE_PATH, f"{base_name}_mask{ext}"),
                            mask_visual
                        )

                    if SAVE_OVERLAY:
                        red_overlay          = np.zeros_like(img_np)
                        red_overlay[:, :, 0] = 255
                        mask_3d  = np.repeat(seg_binary[:, :, np.newaxis], 3, axis=2)
                        overlay  = np.where(
                            mask_3d,
                            0.6 * img_np + 0.4 * red_overlay,
                            img_np
                        )
                        imageio.imwrite(
                            os.path.join(SAVE_PATH, f"{base_name}_overlay{ext}"),
                            overlay.astype("uint8")
                        )

                    total_processed += 1

                except Exception as e:
                    print(f"Error saving {filename}: {e}")

    # ── Save CSV metrics ─────────────────────────────────────────────────
    if SAVE_METRICS and all_metrics:
        csv_path   = os.path.join(SAVE_PATH, "foliage_coverage_metrics.csv")
        fieldnames = [
            'filename', 'coverage_percentage',
            'leaf_pixels', 'background_pixels',
            'total_pixels', 'threshold'
        ]
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for m in all_metrics:
                writer.writerow(m)

        print(f"\n{'='*70}")
        print(f"Coverage metrics saved to: {csv_path}")
        print(f"{'='*70}")

        coverages = [m['coverage_percentage'] for m in all_metrics]

        if PERFORM_ADVANCED_STATS and len(coverages) >= 3:
            perform_statistical_analysis(coverages, SAVE_PATH)

        if GENERATE_PLOTS and len(coverages) >= 3:
            print("\nGenerating statistical visualisations...")
            generate_statistical_plots(coverages, all_metrics, SAVE_PATH)

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Processing complete!")
    print(f"{'='*70}")
    print(f"Total images processed : {total_processed}/{len(test_img_paths)}")
    print(f"Results saved to       : {SAVE_PATH}")
    print("\nGenerated files:")
    print(f"  • Segmented images        : {PREFIX}[filename]")
    if SAVE_MASK_ONLY:
        print("  • Binary masks            : [filename]_mask")
    if SAVE_OVERLAY:
        print("  • Overlay visualisations  : [filename]_overlay")
    if SAVE_METRICS:
        print("  • Coverage CSV            : foliage_coverage_metrics.csv")
        print("  • Statistical report      : statistical_analysis.txt")
    if GENERATE_PLOTS:
        print("  • Statistical plots       : foliage_coverage_analysis.png")