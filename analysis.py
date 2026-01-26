#!/usr/bin/env python3
"""
Statistical Analysis for Post-Quantum Cryptography Benchmarking
Generates publication-quality figures for Q1 conference papers
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, kruskal, mannwhitneyu, f_oneway
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PUBLICATION-QUALITY PLOT SETTINGS (IEEE/ACM Conference Style)
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette - professional and colorblind-friendly
COLORS = {
    'standalone': '#2E86AB',  # Steel blue
    'hybrid': '#A23B72',  # Deep magenta
    'classical': '#F18F01',  # Amber
    'pqc': '#C73E1D',  # Vermillion
    'ascon': '#3B7A57',  # Forest green
    'highlight': '#E63946',  # Red highlight
    'neutral': '#6C757D',  # Gray
}

MODEL_COLORS = {
    'X25519': '#F18F01',
    'Kyber512': '#2E86AB',
    'BIKE_L1': '#A23B72',
    'HQC_128': '#C73E1D',
    'FrodoKEM_640_AES': '#3B7A57',
    'ClassicMcEliece_348864': '#6A4C93',
    'Ascon_80pq': '#1D3557',
    'Hybrid_HQC_128_Ascon128a': '#E76F51',
    'Hybrid_FrodoKEM_640_AES_Ascon128a': '#2A9D8F',
    'Hybrid_ClassicMcEliece_348864_Ascon128a': '#9B5DE5',
}

IOT_MARKERS = {
    '1Core512MB': 'o',
    '2Core1GB': 's',
    '4Core2GB': '^'
}
IOT_COLORS = {
    '1Core512MB': '#E63946',
    '2Core1GB': '#457B9D',
    '4Core2GB': '#2D6A4F'
}

OUTPUT_DIR = 'analysis_output'

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_performance_data(base_path, subfolder):
    """Load all performance CSV files from a subfolder."""
    folder_path = os.path.join(base_path, subfolder)
    all_data = []

    for csv_file in glob.glob(os.path.join(folder_path, '*.csv')):
        filename = os.path.basename(csv_file)
        # Skip entropy files and combined files
        if '_ent' in filename or filename in ['testing_process.csv', 'testing_process_ent.csv']:
            continue

        try:
            df = pd.read_csv(csv_file)
            if 'Model' in df.columns and 'Total_ns' in df.columns:
                all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_entropy_data(base_path, subfolder):
    """Load entropy test CSV files."""
    folder_path = os.path.join(base_path, subfolder)
    all_data = []

    for csv_file in glob.glob(os.path.join(folder_path, '*_ent_*.csv')):
        try:
            df = pd.read_csv(csv_file)
            if 'Model' in df.columns and 'Entropy_Bits_Per_Byte' in df.columns:
                all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_iot_data(base_path):
    """Load IoT simulation data with configuration extraction."""
    folder_path = os.path.join(base_path, 'IoT_simulation')
    all_data = []

    for csv_file in glob.glob(os.path.join(folder_path, '*.csv')):
        filename = os.path.basename(csv_file)
        # Skip combined files
        if filename.startswith('testing_process_') and filename.count('_') <= 2:
            continue

        try:
            df = pd.read_csv(csv_file)
            if 'Model' in df.columns and 'Total_ns' in df.columns:
                # Extract IoT configuration from filename
                for config in ['1Core512MB', '2Core1GB', '4Core2GB']:
                    if config in filename:
                        df['IoT_Config'] = config
                        break
                all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def compute_descriptive_stats(df, group_col='Model', value_col='Total_ns'):
    """Compute descriptive statistics grouped by model."""
    stats_df = df.groupby(group_col)[value_col].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('q25', lambda x: x.quantile(0.25)),
        ('median', 'median'),
        ('q75', lambda x: x.quantile(0.75)),
        ('max', 'max'),
        ('cv', lambda x: (x.std() / x.mean()) * 100 if x.mean() != 0 else 0),  # CV%
        ('iqr', lambda x: x.quantile(0.75) - x.quantile(0.25)),
    ]).round(4)

    # Add 95% CI
    stats_df['ci_95'] = df.groupby(group_col)[value_col].apply(
        lambda x: 1.96 * x.std() / np.sqrt(len(x))
    ).round(4)

    return stats_df


def test_normality(df, group_col='Model', value_col='Total_ns'):
    """Perform Shapiro-Wilk normality test for each group."""
    results = {}
    for model in df[group_col].unique():
        data = pd.to_numeric(df[df[group_col] == model][value_col], errors='coerce').dropna()
        if len(data) >= 3:
            # Use subset for large samples (Shapiro-Wilk limited to 5000)
            sample = data.sample(min(5000, len(data)), random_state=42)
            stat, p_value = shapiro(sample.values)
            results[model] = {
                'statistic': stat,
                'p_value': p_value,
                'normal': p_value > 0.05
            }
    return pd.DataFrame(results).T


def perform_kruskal_wallis(df, group_col='Model', value_col='Total_ns'):
    """Perform Kruskal-Wallis H-test (non-parametric ANOVA)."""
    groups = [np.asarray(group[value_col].values, dtype=np.float64) 
              for name, group in df.groupby(group_col)]
    if len(groups) >= 2:
        stat, p_value = kruskal(*groups)
        return {'statistic': float(stat), 'p_value': float(p_value), 'significant': p_value < 0.05}
    return None


def pairwise_mannwhitney(df, group_col='Model', value_col='Total_ns'):
    """Perform pairwise Mann-Whitney U tests with Bonferroni correction."""
    models = df[group_col].unique()
    n_comparisons = len(models) * (len(models) - 1) // 2
    results = []

    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            data1 = pd.to_numeric(df[df[group_col] == model1][value_col], errors='coerce').dropna().values
            data2 = pd.to_numeric(df[df[group_col] == model2][value_col], errors='coerce').dropna().values

            if len(data1) >= 3 and len(data2) >= 3:
                stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                corrected_p = min(p_value * n_comparisons, 1.0)  # Bonferroni
                results.append({
                    'model_1': model1,
                    'model_2': model2,
                    'u_statistic': stat,
                    'p_value': p_value,
                    'corrected_p': corrected_p,
                    'significant': corrected_p < 0.05
                })

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_output_dir():
    """Create output directory for figures."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_model_display_name(model):
    """Convert model name to display-friendly format."""
    replacements = {
        'Standalone_': '',
        'Hybrid_': 'Hybrid\n',
        '_Ascon128a': '\n+Ascon',
        '_': '-',
        'ClassicMcEliece': 'McEliece',
        'FrodoKEM': 'Frodo',
    }
    name = model
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name


def plot_execution_time_comparison(df, title_suffix='', filename='exec_time_comparison.pdf'):
    """Bar chart comparing mean execution times with error bars."""
    fig, ax = plt.subplots(figsize=(7, 4))

    stats = df.groupby('Model')['Total_ns'].agg(['mean', 'std', 'count'])
    stats['se'] = stats['std'] / np.sqrt(stats['count'])
    stats['ci_95'] = 1.96 * stats['se']
    stats = stats.sort_values('mean')

    # Convert to microseconds for better readability
    stats['mean_us'] = stats['mean'] / 1000
    stats['ci_95_us'] = stats['ci_95'] / 1000

    colors = [MODEL_COLORS.get(m.replace('Standalone_', '').replace('Hybrid_', 'Hybrid_'), COLORS['neutral'])
              for m in stats.index]

    bars = ax.barh(range(len(stats)), stats['mean_us'], xerr=stats['ci_95_us'],
                   color=colors, edgecolor='black', linewidth=0.5, capsize=3, alpha=0.85)

    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels([get_model_display_name(m) for m in stats.index])
    ax.set_xlabel('Execution Time (μs)')
    ax.set_title(f'Mean Execution Time Comparison{title_suffix}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Add value labels
    for i, (idx, row) in enumerate(stats.iterrows()):
        ax.text(row['mean_us'] + row['ci_95_us'] + stats['mean_us'].max() * 0.02, i,
                f'{row["mean_us"]:,.1f}', va='center', fontsize=7)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_operation_breakdown(df, filename='operation_breakdown.pdf'):
    """Stacked bar chart showing breakdown of cryptographic operations."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ops = ['KeyGen_ns', 'Encaps_ns', 'Decaps_ns', 'KDF_ns', 'Encryption_ns', 'Decryption_ns']
    op_labels = ['Key Generation', 'Encapsulation', 'Decapsulation', 'KDF', 'Encryption', 'Decryption']
    op_colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51', '#9B2226']

    means = df.groupby('Model')[ops].mean()
    # Ensure numeric values to avoid object dtype issues
    means = means.apply(pd.to_numeric, errors='coerce')
    means = means / 1000  # Convert to μs
    # Sort by total execution time
    means['_total'] = means.sum(axis=1)
    means = means.sort_values('_total', ascending=False).drop('_total', axis=1)

    bottom = np.zeros(len(means))
    for op, label, color in zip(ops, op_labels, op_colors):
        values = means[op].values
        ax.barh(range(len(means)), values, left=bottom, label=label,
                color=color, edgecolor='white', linewidth=0.3)
        bottom += values

    ax.set_yticks(range(len(means)))
    ax.set_yticklabels([get_model_display_name(m) for m in means.index])
    ax.set_xlabel('Time (μs)')
    ax.set_title('Cryptographic Operation Breakdown')
    ax.legend(loc='lower right', framealpha=0.9, ncol=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_boxplot_comparison(df, filename='boxplot_comparison.pdf'):
    """Box plot for distribution comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Sort models by median
    model_order = df.groupby('Model')['Total_ns'].median().sort_values().index.tolist()

    # Prepare data for boxplot
    data_to_plot = [df[df['Model'] == m]['Total_ns'].values / 1000 for m in model_order]

    bp = ax.boxplot(data_to_plot, vert=False, patch_artist=True,
                    widths=0.6, showfliers=False)

    # Color boxes
    for i, (patch, model) in enumerate(zip(bp['boxes'], model_order)):
        base_model = model.replace('Standalone_', '').replace('Hybrid_', 'Hybrid_')
        color = MODEL_COLORS.get(base_model, COLORS['neutral'])
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')

    # Style whiskers and medians
    for element in ['whiskers', 'caps']:
        for line in bp[element]:
            line.set_color('black')
            line.set_linewidth(0.8)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)

    ax.set_yticks(range(1, len(model_order) + 1))
    ax.set_yticklabels([get_model_display_name(m) for m in model_order])
    ax.set_xlabel('Execution Time (μs)')
    ax.set_title('Execution Time Distribution')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_memory_comparison(df, filename='memory_comparison.pdf'):
    """Compare peak memory allocation across models."""
    if 'Peak_Alloc_KB' not in df.columns:
        print("Skipping memory comparison: Peak_Alloc_KB not found")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    stats = df.groupby('Model')['Peak_Alloc_KB'].agg(['mean', 'std'])
    stats = stats.sort_values('mean')

    colors = [MODEL_COLORS.get(m.replace('Standalone_', '').replace('Hybrid_', 'Hybrid_'), COLORS['neutral'])
              for m in stats.index]

    bars = ax.barh(range(len(stats)), stats['mean'],
                   color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels([get_model_display_name(m) for m in stats.index])
    ax.set_xlabel('Peak Memory Allocation (KB)')
    ax.set_title('Memory Usage Comparison')

    # Add value labels
    for i, (idx, row) in enumerate(stats.iterrows()):
        ax.text(row['mean'] + stats['mean'].max() * 0.02, i,
                f'{row["mean"]:,.0f}', va='center', fontsize=7)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_entropy_analysis(df, filename='entropy_analysis.pdf'):
    """Visualize entropy test results."""
    if df.empty or 'Entropy_Bits_Per_Byte' not in df.columns:
        print("Skipping entropy analysis: No data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Entropy bits per byte (should be close to 8 for good randomness)
    ax1 = axes[0]
    models = df['Model'].unique()
    positions = range(len(models))

    for i, model in enumerate(models):
        data = df[df['Model'] == model]['Entropy_Bits_Per_Byte']
        bp = ax1.boxplot([data.values], positions=[i], widths=0.6,
                         patch_artist=True, showfliers=False)
        color = MODEL_COLORS.get(model.replace('Standalone_', ''), COLORS['neutral'])
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)

    ax1.axhline(y=8.0, color='red', linestyle='--', linewidth=1, label='Ideal (8.0)')
    ax1.set_xticks(positions)
    ax1.set_xticklabels([get_model_display_name(m) for m in models], rotation=45, ha='right')
    ax1.set_ylabel('Entropy (bits/byte)')
    ax1.set_title('Entropy Quality')
    ax1.set_ylim(7.9999, 8.0001)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Serial correlation coefficient (should be close to 0)
    ax2 = axes[1]
    for i, model in enumerate(models):
        data = df[df['Model'] == model]['Serial_Correlation_Coefficient']
        bp = ax2.boxplot([data.values], positions=[i], widths=0.6,
                         patch_artist=True, showfliers=False)
        color = MODEL_COLORS.get(model.replace('Standalone_', ''), COLORS['neutral'])
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)

    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Ideal (0.0)')
    ax2.set_xticks(positions)
    ax2.set_xticklabels([get_model_display_name(m) for m in models], rotation=45, ha='right')
    ax2.set_ylabel('Serial Correlation Coefficient')
    ax2.set_title('Randomness Quality')
    ax2.legend(loc='lower right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_iot_comparison(df, filename='iot_comparison.pdf'):
    """Compare performance across IoT configurations."""
    if df.empty or 'IoT_Config' not in df.columns:
        print("Skipping IoT comparison: No data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Get unique models (without config suffix)
    df['Base_Model'] = df['Model'].str.replace('Standalone_', '').str.replace('Hybrid_', 'Hybrid_')

    # Plot 1: Grouped bar chart - execution time by config
    ax1 = axes[0]
    configs = ['1Core512MB', '2Core1GB', '4Core2GB']
    models = df['Base_Model'].unique()
    x = np.arange(len(models))
    width = 0.25

    for i, config in enumerate(configs):
        config_data = df[df['IoT_Config'] == config]
        means = config_data.groupby('Base_Model')['Total_ns'].mean() / 1000  # μs
        means = means.reindex(models).fillna(0)
        bars = ax1.bar(x + i * width, means.values, width, label=config.replace('Core', ' Core, ').replace('MB', ' MB').replace('GB', ' GB'),
                       color=IOT_COLORS[config], edgecolor='black', linewidth=0.5, alpha=0.85)

    ax1.set_xticks(x + width)
    ax1.set_xticklabels([get_model_display_name(m) for m in models], rotation=45, ha='right')
    ax1.set_ylabel('Execution Time (μs)')
    ax1.set_title('Performance by IoT Configuration')
    ax1.legend(title='Configuration', loc='upper right')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: CPU utilization comparison
    ax2 = axes[1]
    if 'Cpu_Pct' in df.columns:
        for i, config in enumerate(configs):
            config_data = df[df['IoT_Config'] == config]
            means = config_data.groupby('Base_Model')['Cpu_Pct'].mean()
            means = means.reindex(models).fillna(0)
            bars = ax2.bar(x + i * width, means.values, width, label=config.replace('Core', ' Core, ').replace('MB', ' MB').replace('GB', ' GB'),
                           color=IOT_COLORS[config], edgecolor='black', linewidth=0.5, alpha=0.85)

    ax2.set_xticks(x + width)
    ax2.set_xticklabels([get_model_display_name(m) for m in models], rotation=45, ha='right')
    ax2.set_ylabel('CPU Utilization (%)')
    ax2.set_title('CPU Utilization by Configuration')
    ax2.legend(title='Configuration', loc='upper right')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_iot_scaling(df, filename='iot_scaling.pdf'):
    """Line plot showing performance scaling across IoT tiers."""
    if df.empty or 'IoT_Config' not in df.columns:
        print("Skipping IoT scaling: No data available")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    df['Base_Model'] = df['Model'].str.replace('Standalone_', '').str.replace('Hybrid_', 'Hybrid_')
    configs = ['1Core512MB', '2Core1GB', '4Core2GB']
    config_labels = ['1 Core\n512 MB', '2 Cores\n1 GB', '4 Cores\n2 GB']

    for model in df['Base_Model'].unique():
        model_data = df[df['Base_Model'] == model]
        means = []
        for config in configs:
            config_mean = model_data[model_data['IoT_Config'] == config]['Total_ns'].mean() / 1000
            means.append(config_mean if not np.isnan(config_mean) else 0)

        color = MODEL_COLORS.get(model, COLORS['neutral'])
        marker = 'o' if 'Hybrid' not in model else 's'
        ax.plot(range(len(configs)), means, marker=marker, label=get_model_display_name(model),
                color=color, linewidth=1.5, markersize=6)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(config_labels)
    ax.set_xlabel('IoT Device Configuration')
    ax.set_ylabel('Execution Time (μs)')
    ax.set_title('Performance Scaling Across IoT Tiers')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_iot_vs_base_ratio(base_df, iot_df, filename='iot_vs_base_ratio.pdf'):
    """Bar chart of IoT mean time ratio vs base machine for common models."""
    if base_df.empty or iot_df.empty:
        print("Skipping IoT vs base ratio: Missing data")
        return
    fig, ax = plt.subplots(figsize=(7, 4))

    base_means = base_df.groupby('Model')['Total_ns'].mean()
    iot_means = iot_df.groupby('Model')['Total_ns'].mean()
    common_models = base_means.index.intersection(iot_means.index)
    if len(common_models) == 0:
        print("Skipping IoT vs base ratio: No common models")
        plt.close()
        return

    ratio = (iot_means.loc[common_models] / base_means.loc[common_models]).sort_values()
    colors = [MODEL_COLORS.get(m.replace('Standalone_', '').replace('Hybrid_', 'Hybrid_'), COLORS['neutral'])
              for m in ratio.index]

    ax.barh(range(len(ratio)), ratio.values, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.set_yticks(range(len(ratio)))
    ax.set_yticklabels([get_model_display_name(m) for m in ratio.index])
    ax.set_xlabel('IoT / Base Mean Time (x)')
    ax.set_title('Relative Slowdown: IoT Simulation vs Base Machine')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.2f}'))
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, v in enumerate(ratio.values):
        ax.text(v + 0.02, i, f'{v:.2f}x', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_violin_comparison(df, filename='violin_comparison.pdf'):
    """Violin plot for detailed distribution comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    model_order = df.groupby('Model')['Total_ns'].median().sort_values().index.tolist()
    filtered_models = []
    data_to_plot = []
    for model in model_order:
        data = pd.to_numeric(df[df['Model'] == model]['Total_ns'], errors='coerce').dropna().values / 1000
        # Violinplot requires at least two points for KDE
        if len(data) >= 2:
            filtered_models.append(model)
            data_to_plot.append(data)

    if not data_to_plot:
        print("Skipping violin plot: insufficient data")
        plt.close()
        return

    parts = ax.violinplot(data_to_plot, positions=range(len(model_order)),
                          showmeans=True, showmedians=True, widths=0.7)

    # Color violins
    for i, (pc, model) in enumerate(zip(parts['bodies'], filtered_models)):
        base_model = model.replace('Standalone_', '').replace('Hybrid_', 'Hybrid_')
        color = MODEL_COLORS.get(base_model, COLORS['neutral'])
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')

    parts['cmeans'].set_color('red')
    parts['cmedians'].set_color('black')

    ax.set_xticks(range(len(filtered_models)))
    ax.set_xticklabels([get_model_display_name(m) for m in filtered_models], rotation=45, ha='right')
    ax.set_ylabel('Execution Time (μs)')
    ax.set_title('Execution Time Distribution (Violin Plot)')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add legend for mean/median
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='red', linewidth=2, label='Mean'),
                       Line2D([0], [0], color='black', linewidth=2, label='Median')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_performance_heatmap(df, filename='performance_heatmap.pdf'):
    """Heatmap of normalized performance metrics."""
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = ['KeyGen_ns', 'Encaps_ns', 'Decaps_ns', 'Encryption_ns', 'Decryption_ns']
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        print("Skipping heatmap: No metrics available")
        plt.close()
        return

    pivot = df.groupby('Model')[available_metrics].mean()
    # Ensure numeric values
    pivot = pivot.apply(pd.to_numeric, errors='coerce')

    # Normalize each column (0-1 scale)
    normalized = (pivot - pivot.min()) / (pivot.max() - pivot.min())
    normalized = normalized.fillna(0).astype(np.float64)

    # Sort by total normalized score
    normalized = normalized.loc[normalized.sum(axis=1).sort_values().index]

    im = ax.imshow(normalized.values.astype(np.float64), cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(available_metrics)))
    ax.set_xticklabels([m.replace('_ns', '').replace('_', ' ') for m in available_metrics], rotation=45, ha='right')
    ax.set_yticks(range(len(normalized)))
    ax.set_yticklabels([get_model_display_name(m) for m in normalized.index])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Time\n(0=fastest, 1=slowest)', fontsize=8)

    # Add text annotations
    for i in range(len(normalized)):
        for j in range(len(available_metrics)):
            value = normalized.iloc[i, j]
            color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=color, fontsize=7)

    ax.set_title('Normalized Performance Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_standalone_vs_hybrid(base_df, filename='standalone_vs_hybrid.pdf'):
    """Compare standalone PQC vs hybrid constructions."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Categorize models
    standalone = base_df[base_df['Model'].str.startswith('Standalone_')]
    hybrid = base_df[base_df['Model'].str.startswith('Hybrid_')]

    if standalone.empty or hybrid.empty:
        print("Skipping standalone vs hybrid: Missing data")
        return

    # Calculate means for comparison
    standalone_means = standalone.groupby('Model')['Total_ns'].mean() / 1000
    hybrid_means = hybrid.groupby('Model')['Total_ns'].mean() / 1000

    # Create comparison data
    categories = ['Standalone\nPQC', 'Hybrid\nPQC+AEAD']
    all_standalone = pd.to_numeric(standalone['Total_ns'], errors='coerce').dropna().values / 1000
    all_hybrid = pd.to_numeric(hybrid['Total_ns'], errors='coerce').dropna().values / 1000

    bp = ax.boxplot([all_standalone, all_hybrid], labels=categories,
                    patch_artist=True, widths=0.5, showfliers=False)

    bp['boxes'][0].set_facecolor(COLORS['standalone'])
    bp['boxes'][1].set_facecolor(COLORS['hybrid'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
        box.set_edgecolor('black')

    ax.set_ylabel('Execution Time (μs)')
    ax.set_title('Standalone vs Hybrid Construction Performance')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistical annotation
    stat, p_value = mannwhitneyu(all_standalone, all_hybrid)
    sig_text = f'Mann-Whitney U: p = {p_value:.2e}'
    ax.text(0.5, 0.95, sig_text, transform=ax.transAxes, ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_cv_comparison(df, filename='cv_comparison.pdf'):
    """Bar chart of coefficient of variation (timing consistency)."""
    fig, ax = plt.subplots(figsize=(7, 4))

    cv_data = df.groupby('Model')['Total_ns'].agg(
        lambda x: (x.std() / x.mean()) * 100 if x.mean() != 0 else 0
    ).sort_values()

    colors = [MODEL_COLORS.get(m.replace('Standalone_', '').replace('Hybrid_', 'Hybrid_'), COLORS['neutral'])
              for m in cv_data.index]

    bars = ax.barh(range(len(cv_data)), cv_data.values,
                   color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_yticks(range(len(cv_data)))
    ax.set_yticklabels([get_model_display_name(m) for m in cv_data.index])
    ax.set_xlabel('Coefficient of Variation (%)')
    ax.set_title('Timing Consistency (Lower = More Consistent)')

    # Add value labels
    for i, v in enumerate(cv_data.values):
        ax.text(v + cv_data.max() * 0.02, i, f'{v:.1f}%', va='center', fontsize=7)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_latex_table(stats_df, caption, label):
    """Generate LaTeX table from statistics DataFrame."""
    latex = stats_df.to_latex(
        float_format='%.2f',
        caption=caption,
        label=label,
        column_format='l' + 'r' * len(stats_df.columns)
    )
    return latex


def save_statistics_report(base_stats, iot_stats, entropy_stats, kw_results, pairwise_results,
                           base_perf=None, iot_data=None, entropy_data=None):
    """Save comprehensive statistics report with separate tables for each metric category."""
    report_path = os.path.join(OUTPUT_DIR, 'statistical_report.txt')

    def compute_metric_stats(df, group_col, value_col):
        """Compute descriptive statistics for a specific metric."""
        if df is None or df.empty or value_col not in df.columns:
            return None
        data = df.groupby(group_col)[value_col].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('median', 'median'),
            ('max', 'max'),
            ('cv', lambda x: (x.std() / x.mean()) * 100 if x.mean() != 0 else 0),
        ]).round(4)
        return data

    with open(report_path, 'w') as f:
        def format_summary(stats_df, label, is_time=True):
            if stats_df is None or stats_df.empty:
                return [f"{label}: No data available."]

            fastest = stats_df['mean'].idxmin()
            slowest = stats_df['mean'].idxmax()
            if is_time:
                fastest_mean = stats_df.loc[fastest, 'mean'] / 1000
                slowest_mean = stats_df.loc[slowest, 'mean'] / 1000
                mean_range = (stats_df['mean'].max() - stats_df['mean'].min()) / 1000
                unit = "us"
            else:
                fastest_mean = stats_df.loc[fastest, 'mean']
                slowest_mean = stats_df.loc[slowest, 'mean']
                mean_range = stats_df['mean'].max() - stats_df['mean'].min()
                unit = ""
            cv_max_model = stats_df['cv'].idxmax()
            cv_max = stats_df.loc[cv_max_model, 'cv']

            return [
                f"{label}:",
                f"  Fastest mean: {fastest} at {fastest_mean:,.2f} {unit}".strip(),
                f"  Slowest mean: {slowest} at {slowest_mean:,.2f} {unit}".strip(),
                f"  Mean range: {mean_range:,.2f} {unit}".strip(),
                f"  Highest variability: {cv_max_model} at {cv_max:.2f} percent CV",
            ]

        f.write("=" * 100 + "\n")
        f.write("STATISTICAL ANALYSIS REPORT - PQC BENCHMARKING\n")
        f.write("=" * 100 + "\n\n")

        f.write("METRICS RECORDED AND UNITS\n")
        f.write("-" * 50 + "\n")
        f.write("Performance metrics (time): KeyGen_ns, Encaps_ns, Decaps_ns, KDF_ns,\n")
        f.write("  Encryption_ns, Decryption_ns, Total_ns (nanoseconds; summaries also shown in microseconds).\n")
        f.write("Resource metrics: Peak_Alloc_KB (kilobytes), Peak_RSS_KB (kilobytes, IoT only),\n")
        f.write("  Cpu_Pct (percent, IoT only).\n")
        f.write("Entropy metrics: Entropy_Bits_Per_Byte (ideal 8.0), Serial_Correlation_Coefficient (ideal 0.0).\n")
        f.write("Comparisons in tables: base machine and IoT simulation compare models within each environment.\n\n")

        section_num = 1

        # =====================================================================
        # SECTION: BASE MACHINE STATISTICS
        # =====================================================================
        f.write("=" * 100 + "\n")
        f.write(f"{section_num}. BASE MACHINE STATISTICS\n")
        f.write("=" * 100 + "\n\n")

        # 1.1 Total Execution Time (summary)
        f.write(f"  {section_num}.1 TOTAL EXECUTION TIME (Total_ns)\n")
        f.write("  " + "-" * 50 + "\n")
        if base_stats is not None and not base_stats.empty:
            f.write(base_stats.to_string())
            f.write("\n\n")
        else:
            f.write("  No data available.\n\n")

        # 1.2 Individual Performance Metrics
        time_metrics = ['KeyGen_ns', 'Encaps_ns', 'Decaps_ns', 'KDF_ns', 'Encryption_ns', 'Decryption_ns']
        sub_section = 2
        for metric in time_metrics:
            if base_perf is not None and metric in base_perf.columns:
                metric_stats = compute_metric_stats(base_perf, 'Model', metric)
                if metric_stats is not None and not metric_stats.empty:
                    metric_label = metric.replace('_ns', '').replace('_', ' ')
                    f.write(f"  {section_num}.{sub_section} {metric_label.upper()} ({metric})\n")
                    f.write("  " + "-" * 50 + "\n")
                    f.write(metric_stats.to_string())
                    f.write("\n\n")
                    sub_section += 1

        # 1.x Resource Metrics - Peak Memory
        if base_perf is not None and 'Peak_Alloc_KB' in base_perf.columns:
            memory_stats = compute_metric_stats(base_perf, 'Model', 'Peak_Alloc_KB')
            if memory_stats is not None and not memory_stats.empty:
                f.write(f"  {section_num}.{sub_section} PEAK MEMORY ALLOCATION (Peak_Alloc_KB)\n")
                f.write("  " + "-" * 50 + "\n")
                f.write(memory_stats.to_string())
                f.write("\n\n")
                sub_section += 1

        section_num += 1

        # =====================================================================
        # SECTION: IoT SIMULATION STATISTICS
        # =====================================================================
        f.write("=" * 100 + "\n")
        f.write(f"{section_num}. IoT SIMULATION STATISTICS\n")
        f.write("=" * 100 + "\n\n")

        # 2.1 Total Execution Time
        f.write(f"  {section_num}.1 TOTAL EXECUTION TIME (Total_ns)\n")
        f.write("  " + "-" * 50 + "\n")
        if iot_stats is not None and not iot_stats.empty:
            f.write(iot_stats.to_string())
            f.write("\n\n")
        else:
            f.write("  No IoT data available.\n\n")

        # 2.2 Individual Performance Metrics for IoT
        sub_section = 2
        for metric in time_metrics:
            if iot_data is not None and metric in iot_data.columns:
                metric_stats = compute_metric_stats(iot_data, 'Model', metric)
                if metric_stats is not None and not metric_stats.empty:
                    metric_label = metric.replace('_ns', '').replace('_', ' ')
                    f.write(f"  {section_num}.{sub_section} {metric_label.upper()} ({metric})\n")
                    f.write("  " + "-" * 50 + "\n")
                    f.write(metric_stats.to_string())
                    f.write("\n\n")
                    sub_section += 1

        # 2.x Resource Metrics for IoT
        resource_metrics_iot = ['Peak_Alloc_KB', 'Peak_RSS_KB', 'Cpu_Pct']
        for metric in resource_metrics_iot:
            if iot_data is not None and metric in iot_data.columns:
                resource_stats = compute_metric_stats(iot_data, 'Model', metric)
                if resource_stats is not None and not resource_stats.empty:
                    if metric == 'Peak_Alloc_KB':
                        label = "PEAK MEMORY ALLOCATION (Peak_Alloc_KB)"
                    elif metric == 'Peak_RSS_KB':
                        label = "PEAK RSS MEMORY (Peak_RSS_KB)"
                    else:
                        label = "CPU UTILIZATION (Cpu_Pct)"
                    f.write(f"  {section_num}.{sub_section} {label}\n")
                    f.write("  " + "-" * 50 + "\n")
                    f.write(resource_stats.to_string())
                    f.write("\n\n")
                    sub_section += 1

        section_num += 1

        # =====================================================================
        # SECTION: ENTROPY STATISTICS
        # =====================================================================
        f.write("=" * 100 + "\n")
        f.write(f"{section_num}. ENTROPY TEST STATISTICS\n")
        f.write("=" * 100 + "\n\n")

        # 3.1 Entropy Bits Per Byte
        f.write(f"  {section_num}.1 ENTROPY BITS PER BYTE (Entropy_Bits_Per_Byte)\n")
        f.write("  " + "-" * 50 + "\n")
        f.write("  Ideal value: 8.0 bits/byte (maximum entropy)\n")
        if entropy_stats is not None and not entropy_stats.empty:
            f.write(entropy_stats.to_string())
            f.write("\n\n")
        else:
            f.write("  No entropy data available.\n\n")

        # 3.2 Serial Correlation Coefficient
        if entropy_data is not None and 'Serial_Correlation_Coefficient' in entropy_data.columns:
            serial_stats = compute_metric_stats(entropy_data, 'Model', 'Serial_Correlation_Coefficient')
            if serial_stats is not None and not serial_stats.empty:
                f.write(f"  {section_num}.2 SERIAL CORRELATION COEFFICIENT (Serial_Correlation_Coefficient)\n")
                f.write("  " + "-" * 50 + "\n")
                f.write("  Ideal value: 0.0 (no correlation between consecutive bytes)\n")
                f.write(serial_stats.to_string())
                f.write("\n\n")

        section_num += 1

        # =====================================================================
        # SECTION: STATISTICAL TESTS
        # =====================================================================
        f.write("=" * 100 + "\n")
        f.write(f"{section_num}. STATISTICAL HYPOTHESIS TESTS\n")
        f.write("=" * 100 + "\n\n")

        # Kruskal-Wallis test
        f.write(f"  {section_num}.1 KRUSKAL-WALLIS H-TEST (Non-parametric ANOVA)\n")
        f.write("  " + "-" * 50 + "\n")
        if kw_results:
            f.write(f"  H-statistic: {kw_results['statistic']:.4f}\n")
            f.write(f"  p-value: {kw_results['p_value']:.2e}\n")
            f.write(f"  Significant difference: {'Yes' if kw_results['significant'] else 'No'}\n")
            f.write("  Interpretation: Tests whether all models have the same median execution time.\n\n")
        else:
            f.write("  Test not performed.\n\n")

        # Pairwise comparisons
        f.write(f"  {section_num}.2 PAIRWISE MANN-WHITNEY U TESTS (Top 10 significant)\n")
        f.write("  " + "-" * 50 + "\n")
        f.write("  Bonferroni-corrected p-values for multiple comparisons.\n")
        if pairwise_results is not None and not pairwise_results.empty:
            sig_results = pairwise_results[pairwise_results['significant']].head(10)
            if not sig_results.empty:
                f.write(sig_results.to_string(index=False))
            else:
                f.write("  No significant pairwise differences found.")
            f.write("\n\n")
        else:
            f.write("  No pairwise tests performed.\n\n")

        section_num += 1

        # =====================================================================
        # SECTION: COMPARATIVE ANALYSIS
        # =====================================================================
        f.write("=" * 100 + "\n")
        f.write(f"{section_num}. COMPARATIVE ANALYSIS (Base Machine vs IoT Simulation)\n")
        f.write("=" * 100 + "\n\n")

        for line in format_summary(base_stats, "Base machine results (Total_ns)"):
            f.write(line + "\n")
        f.write("\n")
        for line in format_summary(iot_stats, "IoT simulation results (Total_ns)"):
            f.write(line + "\n")
        f.write("\n")

        if base_stats is not None and iot_stats is not None and not base_stats.empty and not iot_stats.empty:
            common_models = base_stats.index.intersection(iot_stats.index)
            if len(common_models) > 0:
                ratios = (iot_stats.loc[common_models, 'mean'] / base_stats.loc[common_models, 'mean']).dropna()
                if len(ratios) > 0:
                    f.write("IoT vs Base Machine Performance Ratio (common models):\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"  Median ratio: {ratios.median():.2f}x slower on IoT\n")
                    f.write(f"  Min ratio: {ratios.min():.2f}x\n")
                    f.write(f"  Max ratio: {ratios.max():.2f}x\n\n")

                    # Per-model ratio table
                    f.write("Per-Model IoT/Base Ratio:\n")
                    ratio_df = pd.DataFrame({
                        'Base_Mean_us': base_stats.loc[common_models, 'mean'] / 1000,
                        'IoT_Mean_us': iot_stats.loc[common_models, 'mean'] / 1000,
                        'Ratio': ratios
                    }).round(2)
                    ratio_df = ratio_df.sort_values('Ratio')
                    f.write(ratio_df.to_string())
                    f.write("\n\n")

        f.write("=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")

    print(f"Saved: statistical_report.txt")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 60)
    print("PQC BENCHMARKING STATISTICAL ANALYSIS")
    print("=" * 60 + "\n")

    base_path = 'saved_output'

    # Create output directory
    create_output_dir()

    # Load data
    print("Loading data...")
    base_perf = load_performance_data(base_path, 'base_machine')
    entropy_data = load_entropy_data(base_path, 'base_machine')
    iot_data = load_iot_data(base_path)

    print(f"  Base machine: {len(base_perf)} records")
    print(f"  Entropy tests: {len(entropy_data)} records")
    print(f"  IoT simulation: {len(iot_data)} records")

    # Compute statistics
    print("\nComputing statistics...")
    base_stats = None
    iot_stats = None
    entropy_stats = None
    kw_results = None
    pairwise_results = None

    if not base_perf.empty:
        base_stats = compute_descriptive_stats(base_perf, 'Model', 'Total_ns')
        print("  Base machine statistics computed")

        # Statistical tests
        kw_results = perform_kruskal_wallis(base_perf)
        if kw_results:
            print(f"  Kruskal-Wallis: H={kw_results['statistic']:.2f}, p={kw_results['p_value']:.2e}")

        pairwise_results = pairwise_mannwhitney(base_perf)
        print(f"  Pairwise Mann-Whitney: {len(pairwise_results)} comparisons")

    if not iot_data.empty:
        iot_stats = compute_descriptive_stats(iot_data, 'Model', 'Total_ns')
        print("  IoT statistics computed")

    if not entropy_data.empty:
        entropy_stats = compute_descriptive_stats(entropy_data, 'Model', 'Entropy_Bits_Per_Byte')
        print("  Entropy statistics computed")

    # Generate visualizations
    print("\nGenerating visualizations...")

    if not base_perf.empty:
        plot_execution_time_comparison(base_perf, ' (Base Machine)', 'exec_time_base.pdf')
        plot_operation_breakdown(base_perf, 'operation_breakdown.pdf')
        plot_boxplot_comparison(base_perf, 'boxplot_base.pdf')
        plot_violin_comparison(base_perf, 'violin_base.pdf')
        plot_memory_comparison(base_perf, 'memory_comparison.pdf')
        plot_performance_heatmap(base_perf, 'performance_heatmap.pdf')
        plot_standalone_vs_hybrid(base_perf, 'standalone_vs_hybrid.pdf')
        plot_cv_comparison(base_perf, 'cv_comparison.pdf')

    if not entropy_data.empty:
        plot_entropy_analysis(entropy_data, 'entropy_analysis.pdf')

    if not iot_data.empty:
        plot_iot_comparison(iot_data, 'iot_comparison.pdf')
        plot_iot_scaling(iot_data, 'iot_scaling.pdf')
        if not base_perf.empty:
            plot_iot_vs_base_ratio(base_perf, iot_data, 'iot_vs_base_ratio.pdf')

    # Save statistics report
    print("\nSaving statistical report...")
    save_statistics_report(base_stats, iot_stats, entropy_stats, kw_results, pairwise_results,
                           base_perf=base_perf, iot_data=iot_data, entropy_data=entropy_data)

    # Print summary table
    if base_stats is not None:
        print("\n" + "=" * 60)
        print("SUMMARY: BASE MACHINE PERFORMANCE (μs)")
        print("=" * 60)
        summary = base_stats[['mean', 'std', 'median', 'cv']].copy()
        summary['mean'] = summary['mean'] / 1000
        summary['std'] = summary['std'] / 1000
        summary['median'] = summary['median'] / 1000
        summary = summary.sort_values('mean')
        summary.columns = ['Mean (μs)', 'Std (μs)', 'Median (μs)', 'CV (%)']
        print(summary.round(2).to_string())

    print("\n" + "=" * 60)
    print(f"Analysis complete. Output saved to: {OUTPUT_DIR}/")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()