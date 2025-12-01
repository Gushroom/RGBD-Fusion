"""
Fusion Method Performance Analysis
Compares different fusion methods across RGB lighting conditions.
Generates heatmaps, bar charts, and summary statistics.

Usage:
    1. Place this script in your project directory
    2. Update the paths for results.txt and RGB images if needed
    3. Run: python fusion_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Update these paths as needed
# =============================================================================
RESULTS_FILE = "results.txt"
RGB_IMAGE_DIR = "MM5_SEG"  # Contains RGB1, RGB2, ..., RGB8 folders
RGB_IMAGE_NAME = "1.png"   # Image filename within each RGB folder

# =============================================================================
# DATA PARSING
# =============================================================================
def parse_results(filepath):
    """Parse results.txt and return a structured DataFrame."""
    data = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':')
            key = key.strip()
            value = float(value.strip())
            data[key] = value
    
    # Extract method names and RGB conditions
    methods = ['rgb_baseline', 'early_fusion', 'mid_fusion', 'late_fusion', 'se_fusion', 'attn_fusion']
    rgb_conditions = [f'RGB{i}' for i in range(1, 9)]
    
    # Build DataFrame
    df_data = []
    for method in methods:
        for rgb in rgb_conditions:
            key = f"{method}_{rgb}"
            if key in data:
                df_data.append({
                    'method': method,
                    'rgb': rgb,
                    'score': data[key]
                })
    
    df = pd.DataFrame(df_data)
    return df

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def compute_statistics(df):
    """Compute mean, std, min, max for each method."""
    stats = df.groupby('method')['score'].agg(['mean', 'std', 'min', 'max']).reset_index()
    stats.columns = ['Method', 'Mean', 'Std', 'Min', 'Max']
    return stats

def compute_improvement(df):
    """Compute improvement over RGB baseline for each method and condition."""
    # Pivot to get method x rgb matrix
    pivot = df.pivot(index='method', columns='rgb', values='score')
    
    # Get baseline row
    baseline = pivot.loc['rgb_baseline']
    
    # Compute improvement (difference from baseline)
    improvement = pivot.subtract(baseline, axis=1)
    
    return pivot, improvement

def compute_pct_improvement(df):
    """Compute percentage improvement over RGB baseline."""
    pivot = df.pivot(index='method', columns='rgb', values='score')
    baseline = pivot.loc['rgb_baseline']
    pct_improvement = ((pivot - baseline) / baseline) * 100
    return pct_improvement

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def load_rgb_images(rgb_dir, image_name, rgb_conditions):
    """Try to load RGB sample images."""
    images = {}
    for rgb in rgb_conditions:
        img_path = Path(rgb_dir) / rgb / image_name
        if img_path.exists():
            try:
                images[rgb] = mpimg.imread(str(img_path))
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")
                images[rgb] = None
        else:
            images[rgb] = None
    return images

def create_heatmap_with_images(pivot, improvement, rgb_images, save_path="fusion_heatmap.png"):
    """Create heatmap with RGB sample images on the right side.
    
    Rotated layout: RGB conditions as rows (y-axis), methods as columns (x-axis).
    This aligns RGB labels with sample images on the right.
    """
    
    # Check if we have any images
    has_images = any(img is not None for img in rgb_images.values())
    
    if has_images:
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.02)
        ax_heat = fig.add_subplot(gs[0])
        ax_imgs = fig.add_subplot(gs[1])
    else:
        fig, ax_heat = plt.subplots(figsize=(12, 8))
    
    # Prepare improvement data (exclude baseline from heatmap)
    improvement_no_baseline = improvement.drop('rgb_baseline', errors='ignore')
    
    # Reorder methods for better visualization
    method_order = ['early_fusion', 'mid_fusion', 'late_fusion', 'se_fusion', 'attn_fusion']
    improvement_ordered = improvement_no_baseline.reindex(method_order)
    
    # Reorder RGB columns
    rgb_order = [f'RGB{i}' for i in range(1, 9)]
    improvement_ordered = improvement_ordered[rgb_order]
    
    # TRANSPOSE: Now RGB conditions are rows, methods are columns
    improvement_transposed = improvement_ordered.T
    
    # Create heatmap with diverging colormap
    vmax = max(abs(improvement_transposed.values.min()), abs(improvement_transposed.values.max()))
    sns.heatmap(
        improvement_transposed,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        ax=ax_heat,
        cbar_kws={'label': 'Improvement over RGB Baseline (%)', 'shrink': 0.8}
    )
    
    # Customize heatmap
    ax_heat.set_title('Performance Improvement vs RGB Baseline\nAcross Different Lighting Conditions', 
                      fontsize=14, fontweight='bold', pad=20)
    ax_heat.set_xlabel('Fusion Method', fontsize=12)
    ax_heat.set_ylabel('Lighting Condition', fontsize=12)
    
    # Clean up method names for x-axis
    method_labels = [m.replace('_', ' ').title() for m in method_order]
    ax_heat.set_xticklabels(method_labels, rotation=45, ha='right')
    ax_heat.set_yticklabels(rgb_order, rotation=0)
    
    # Add RGB sample images if available (now aligned with rows)
    if has_images:
        n_images = len(rgb_order)
        
        for i, rgb in enumerate(rgb_order):
            img = rgb_images.get(rgb)
            # Position images to align with heatmap rows
            y_pos = (n_images - i - 1) / n_images
            img_height = 1.0 / n_images
            
            if img is not None:
                ax_small = ax_imgs.inset_axes([0.05, y_pos + 0.01, 0.9, img_height - 0.02])
                ax_small.imshow(img)
                ax_small.axis('off')
            else:
                ax_imgs.text(0.5, y_pos + img_height/2, f'{rgb}', 
                           ha='center', va='center', fontsize=9, color='gray',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        ax_imgs.set_xlim(0, 1)
        ax_imgs.set_ylim(0, 1)
        ax_imgs.axis('off')
        ax_imgs.set_title('Samples', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    return fig

def create_performance_comparison(pivot, save_path="fusion_performance.png"):
    """Create bar chart comparing all methods across conditions."""
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Prepare data
    methods = ['rgb_baseline', 'early_fusion', 'mid_fusion', 'late_fusion', 'se_fusion', 'attn_fusion']
    rgb_conditions = [f'RGB{i}' for i in range(1, 9)]
    pivot_ordered = pivot.reindex(methods)[rgb_conditions]
    
    # Set up bar positions
    x = np.arange(len(rgb_conditions))
    width = 0.12
    
    # Color palette
    colors = sns.color_palette("husl", len(methods))
    
    # Plot bars for each method
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, pivot_ordered.loc[method], width, 
                     label=method.replace('_', ' ').title(), color=colors[i], alpha=0.85)
    
    ax.set_xlabel('Lighting Condition', fontsize=12)
    ax.set_ylabel('Performance Score (%)', fontsize=12)
    ax.set_title('Performance Comparison Across Fusion Methods and Lighting Conditions', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rgb_conditions)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title='Method')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    return fig

def create_summary_stats_plot(stats, save_path="fusion_summary.png"):
    """Create summary statistics visualization."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Reorder methods
    method_order = ['rgb_baseline', 'early_fusion', 'mid_fusion', 'late_fusion', 'se_fusion', 'attn_fusion']
    stats_ordered = stats.set_index('Method').reindex(method_order).reset_index()
    
    # Left: Mean with error bars
    ax1 = axes[0]
    colors = sns.color_palette("husl", len(method_order))
    x = np.arange(len(method_order))
    
    bars = ax1.bar(x, stats_ordered['Mean'], yerr=stats_ordered['Std'], 
                   capsize=5, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(stats_ordered['Mean'], stats_ordered['Std'])):
        ax1.text(i, mean + std + 1.5, f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('Fusion Method', fontsize=12)
    ax1.set_ylabel('Mean Performance (%)', fontsize=12)
    ax1.set_title('Mean Performance ± Std Dev', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in method_order], rotation=45, ha='right')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Performance range (min-max)
    ax2 = axes[1]
    for i, (method, row) in enumerate(stats_ordered.iterrows()):
        ax2.plot([i, i], [row['Min'], row['Max']], color=colors[i], linewidth=3, solid_capstyle='round')
        ax2.scatter([i], [row['Mean']], color=colors[i], s=100, zorder=5, edgecolor='white', linewidth=2)
        ax2.scatter([i, i], [row['Min'], row['Max']], color=colors[i], s=50, marker='_', linewidth=3)
    
    ax2.set_xlabel('Fusion Method', fontsize=12)
    ax2.set_ylabel('Performance Score (%)', fontsize=12)
    ax2.set_title('Performance Range (Min - Mean - Max)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('_', ' ').title() for m in method_order], rotation=45, ha='right')
    ax2.set_ylim(40, 100)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    return fig

def create_radar_chart(pivot, save_path="fusion_radar.png"):
    """Create radar chart comparing methods across RGB conditions."""
    
    methods = ['rgb_baseline', 'early_fusion', 'mid_fusion', 'late_fusion', 'se_fusion', 'attn_fusion']
    rgb_conditions = [f'RGB{i}' for i in range(1, 9)]
    
    # Number of variables
    num_vars = len(rgb_conditions)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = sns.color_palette("husl", len(methods))
    
    for i, method in enumerate(methods):
        values = pivot.loc[method][rgb_conditions].tolist()
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method.replace('_', ' ').title(), 
                color=colors[i], alpha=0.8)
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(rgb_conditions, fontsize=11)
    ax.set_ylim(40, 100)
    ax.set_title('Fusion Method Performance Across Lighting Conditions', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    return fig

def print_summary_table(stats, improvement):
    """Print formatted summary tables to console."""
    
    print("\n" + "="*70)
    print("FUSION METHOD PERFORMANCE ANALYSIS")
    print("="*70)
    
    print("\n📊 SUMMARY STATISTICS (across all RGB conditions)")
    print("-"*60)
    stats_display = stats.copy()
    stats_display['Mean'] = stats_display['Mean'].map('{:.2f}'.format)
    stats_display['Std'] = stats_display['Std'].map('{:.2f}'.format)
    stats_display['Min'] = stats_display['Min'].map('{:.2f}'.format)
    stats_display['Max'] = stats_display['Max'].map('{:.2f}'.format)
    print(stats_display.to_string(index=False))
    
    print("\n📈 MEAN IMPROVEMENT OVER RGB BASELINE")
    print("-"*60)
    improvement_no_baseline = improvement.drop('rgb_baseline', errors='ignore')
    mean_improvement = improvement_no_baseline.mean(axis=1).sort_values(ascending=False)
    for method, imp in mean_improvement.items():
        sign = "+" if imp > 0 else ""
        emoji = "🟢" if imp > 0 else "🔴"
        print(f"{emoji} {method.replace('_', ' ').title():20s}: {sign}{imp:.2f}%")
    
    print("\n🏆 BEST METHOD PER RGB CONDITION")
    print("-"*60)
    for rgb in improvement.columns:
        best_method = improvement[rgb].idxmax()
        best_value = improvement[rgb].max()
        print(f"{rgb}: {best_method.replace('_', ' ').title()} (+{best_value:.2f}%)")
    
    print("\n" + "="*70)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    # Parse results
    print("Loading results...")
    df = parse_results(RESULTS_FILE)
    
    # Compute statistics
    stats = compute_statistics(df)
    pivot, improvement = compute_improvement(df)
    
    # Print summary to console
    print_summary_table(stats, improvement)
    
    # Try to load RGB images
    rgb_conditions = [f'RGB{i}' for i in range(1, 9)]
    rgb_images = load_rgb_images(RGB_IMAGE_DIR, RGB_IMAGE_NAME, rgb_conditions)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    create_heatmap_with_images(pivot, improvement, rgb_images, "fusion_heatmap.png")
    create_performance_comparison(pivot, "fusion_performance.png")
    create_summary_stats_plot(stats, "fusion_summary.png")
    create_radar_chart(pivot, "fusion_radar.png")
    
    print("\n✅ Analysis complete! Generated files:")
    print("   - fusion_heatmap.png    (improvement heatmap with RGB samples)")
    print("   - fusion_performance.png (grouped bar chart)")
    print("   - fusion_summary.png    (mean ± std and range)")
    print("   - fusion_radar.png      (radar chart)")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()