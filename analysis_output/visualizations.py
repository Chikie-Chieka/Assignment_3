#!/usr/bin/env python3
"""
Visualization Module for PQC Benchmarking Analysis
Separated plotting functions for maintainability
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu

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
    'Hybrid_BIKE_L1_Ascon128a': '#E9C46A',
    'Hybrid_Kyber512_Ascon128a': '#264653',
    'Hybrid_X25519_Ascon128a': '#F4A261',
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

# gem5 specific colors
GEM5_COLORS = {
    'cycles': '#264653',
    'cpi': '#2A9D8F',
    'ipc': '#E9C46A',
    'sim_seconds': '#F4A261',
    'instructions': '#E76F51',
}


def get_model_display_name(model, short=False):
    """Convert model name to display-friendly format.
    
    Args:
        model: Model name string
        short: If True, return abbreviated version to prevent overlapping
    """
    if short:
        # Short abbreviations for tight spaces
        abbrev = {
            'Standalone_Kyber512': 'Kyber',
            'Standalone_BIKE_L1': 'BIKE',
            'Standalone_HQC_128': 'HQC',
            'Standalone_FrodoKEM_640_AES': 'Frodo',
            'Standalone_ClassicMcEliece_348864': 'McEliece',
            'Standalone_X25519': 'X25519',
            'Standalone_Ascon_80pq': 'Ascon',
            'Hybrid_Kyber512_Ascon128a': 'H-Kyber',
            'Hybrid_BIKE_L1_Ascon128a': 'H-BIKE',
            'Hybrid_HQC_128_Ascon128a': 'H-HQC',
            'Hybrid_FrodoKEM_640_AES_Ascon128a': 'H-Frodo',
            'Hybrid_ClassicMcEliece_348864_Ascon128a': 'H-McEliece',
            'Hybrid_X25519_Ascon128a': 'H-X25519',
        }
        return abbrev.get(model, model[:12])
    
    # Standard display names (more readable)
    replacements = {
        'Standalone_': '',
        'Hybrid_': 'H-',
        '_Ascon128a': '+A',
        'ClassicMcEliece_348864': 'McEliece',
        'FrodoKEM_640_AES': 'Frodo',
        '_': '-',
    }
    name = model
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name


def get_model_color(model):
    """Get color for a model, handling various naming conventions."""
    base_model = model.replace('Standalone_', '').replace('Hybrid_', 'Hybrid_')
    return MODEL_COLORS.get(base_model, COLORS['neutral'])


# =============================================================================
# BASE MACHINE VISUALIZATIONS
# =============================================================================

def plot_execution_time_comparison(df, output_dir, title_suffix='', filename='exec_time_comparison.pdf'):
    """Bar chart comparing mean execution times with error bars."""
    fig, ax = plt.subplots(figsize=(7, 4))

    stats = df.groupby('Model')['Total_ns'].agg(['mean', 'std', 'count'])
    stats['se'] = stats['std'] / np.sqrt(stats['count'])
    stats['ci_95'] = 1.96 * stats['se']
    stats = stats.sort_values('mean')

    # Convert to microseconds for better readability
    stats['mean_us'] = stats['mean'] / 1000
    stats['ci_95_us'] = stats['ci_95'] / 1000

    colors = [get_model_color(m) for m in stats.index]

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
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_operation_breakdown(df, output_dir, filename='operation_breakdown.pdf'):
    """Stacked bar chart showing breakdown of cryptographic operations."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ops = ['KeyGen_ns', 'Encaps_ns', 'Decaps_ns', 'KDF_ns', 'Encryption_ns', 'Decryption_ns']
    op_labels = ['Key Generation', 'Encapsulation', 'Decapsulation', 'KDF', 'Encryption', 'Decryption']
    op_colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51', '#9B2226']

    means = df.groupby('Model')[ops].mean()
    means = means.apply(pd.to_numeric, errors='coerce')
    means = means / 1000  # Convert to μs
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
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_boxplot_comparison(df, output_dir, filename='boxplot_comparison.pdf'):
    """Box plot for distribution comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    model_order = df.groupby('Model')['Total_ns'].median().sort_values().index.tolist()
    data_to_plot = [df[df['Model'] == m]['Total_ns'].values / 1000 for m in model_order]

    bp = ax.boxplot(data_to_plot, vert=False, patch_artist=True,
                    widths=0.6, showfliers=False)

    for i, (patch, model) in enumerate(zip(bp['boxes'], model_order)):
        color = get_model_color(model)
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')

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
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_violin_comparison(df, output_dir, filename='violin_comparison.pdf'):
    """Violin plot for detailed distribution comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    model_order = df.groupby('Model')['Total_ns'].median().sort_values().index.tolist()
    filtered_models = []
    data_to_plot = []
    for model in model_order:
        data = pd.to_numeric(df[df['Model'] == model]['Total_ns'], errors='coerce').dropna().values / 1000
        if len(data) >= 2:
            data_to_plot.append(data)
            filtered_models.append(model)

    if not data_to_plot:
        print("Skipping violin plot: insufficient data")
        plt.close()
        return

    parts = ax.violinplot(data_to_plot, positions=range(len(filtered_models)),
                          showmeans=True, showmedians=True, widths=0.7)

    for i, (pc, model) in enumerate(zip(parts['bodies'], filtered_models)):
        color = get_model_color(model)
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

    legend_elements = [Line2D([0], [0], color='red', linewidth=2, label='Mean'),
                       Line2D([0], [0], color='black', linewidth=2, label='Median')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_memory_comparison(df, output_dir, filename='memory_comparison.pdf'):
    """Compare peak memory allocation across models."""
    if 'Peak_Alloc_KB' not in df.columns:
        print("Skipping memory comparison: Peak_Alloc_KB not found")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    stats = df.groupby('Model')['Peak_Alloc_KB'].agg(['mean', 'std'])
    stats = stats.sort_values('mean')

    colors = [get_model_color(m) for m in stats.index]

    bars = ax.barh(range(len(stats)), stats['mean'],
                   color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels([get_model_display_name(m) for m in stats.index])
    ax.set_xlabel('Peak Memory Allocation (KB)')
    ax.set_title('Memory Usage Comparison')

    for i, (idx, row) in enumerate(stats.iterrows()):
        ax.text(row['mean'] + stats['mean'].max() * 0.02, i,
                f'{row["mean"]:,.0f}', va='center', fontsize=7)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_performance_heatmap(df, output_dir, filename='performance_heatmap.pdf'):
    """Heatmap of normalized performance metrics."""
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = ['KeyGen_ns', 'Encaps_ns', 'Decaps_ns', 'Encryption_ns', 'Decryption_ns']
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        print("Skipping heatmap: No metrics available")
        plt.close()
        return

    pivot = df.groupby('Model')[available_metrics].mean()
    pivot = pivot.apply(pd.to_numeric, errors='coerce')

    normalized = (pivot - pivot.min()) / (pivot.max() - pivot.min())
    normalized = normalized.fillna(0).astype(np.float64)
    normalized = normalized.loc[normalized.sum(axis=1).sort_values().index]

    im = ax.imshow(normalized.values.astype(np.float64), cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(available_metrics)))
    ax.set_xticklabels([m.replace('_ns', '').replace('_', ' ') for m in available_metrics], rotation=45, ha='right')
    ax.set_yticks(range(len(normalized)))
    ax.set_yticklabels([get_model_display_name(m) for m in normalized.index])

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Time\n(0=fastest, 1=slowest)', fontsize=8)

    for i in range(len(normalized)):
        for j in range(len(available_metrics)):
            val = normalized.iloc[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

    ax.set_title('Normalized Performance Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_standalone_vs_hybrid(base_df, output_dir, filename='standalone_vs_hybrid.pdf'):
    """Compare standalone PQC vs hybrid constructions."""
    fig, ax = plt.subplots(figsize=(7, 5))

    standalone = base_df[base_df['Model'].str.startswith('Standalone_')]
    hybrid = base_df[base_df['Model'].str.startswith('Hybrid_')]

    if standalone.empty or hybrid.empty:
        print("Skipping standalone vs hybrid: Missing data")
        plt.close()
        return

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

    stat, p_value = mannwhitneyu(all_standalone, all_hybrid)
    sig_text = f'Mann-Whitney U: p = {p_value:.2e}'
    ax.text(0.5, 0.95, sig_text, transform=ax.transAxes, ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_cv_comparison(df, output_dir, filename='cv_comparison.pdf'):
    """Bar chart of coefficient of variation (timing consistency)."""
    fig, ax = plt.subplots(figsize=(7, 4))

    cv_data = df.groupby('Model')['Total_ns'].agg(
        lambda x: (x.std() / x.mean()) * 100 if x.mean() != 0 else 0
    ).sort_values()

    colors = [get_model_color(m) for m in cv_data.index]

    bars = ax.barh(range(len(cv_data)), cv_data.values,
                   color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_yticks(range(len(cv_data)))
    ax.set_yticklabels([get_model_display_name(m) for m in cv_data.index])
    ax.set_xlabel('Coefficient of Variation (%)')
    ax.set_title('Timing Consistency (Lower = More Consistent)')

    for i, v in enumerate(cv_data.values):
        ax.text(v + cv_data.max() * 0.02, i, f'{v:.1f}%', va='center', fontsize=7)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


# =============================================================================
# ENTROPY VISUALIZATIONS
# =============================================================================

def plot_entropy_analysis(df, output_dir, filename='entropy_analysis.pdf'):
    """Visualize entropy test results."""
    if df.empty or 'Entropy_Bits_Per_Byte' not in df.columns:
        print("Skipping entropy analysis: No data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Entropy bits per byte
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

    # Plot 2: Serial correlation coefficient
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
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


# =============================================================================
# IOT VISUALIZATIONS
# =============================================================================

def plot_iot_comparison(df, output_dir, filename='iot_comparison.pdf'):
    """Compare performance across IoT configurations."""
    if df.empty or 'IoT_Config' not in df.columns:
        print("Skipping IoT comparison: No data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    df['Base_Model'] = df['Model'].str.replace('Standalone_', '').str.replace('Hybrid_', 'Hybrid_')

    ax1 = axes[0]
    configs = ['1Core512MB', '2Core1GB', '4Core2GB']
    models = df['Base_Model'].unique()
    x = np.arange(len(models))
    width = 0.25

    for i, config in enumerate(configs):
        config_data = df[df['IoT_Config'] == config]
        means = config_data.groupby('Base_Model')['Total_ns'].mean() / 1000
        means = means.reindex(models).fillna(0)
        bars = ax1.bar(x + i * width, means.values, width, 
                       label=config.replace('Core', ' Core, ').replace('MB', ' MB').replace('GB', ' GB'),
                       color=IOT_COLORS[config], edgecolor='black', linewidth=0.5, alpha=0.85)

    ax1.set_xticks(x + width)
    ax1.set_xticklabels([get_model_display_name(m) for m in models], rotation=45, ha='right')
    ax1.set_ylabel('Execution Time (μs)')
    ax1.set_title('Performance by IoT Configuration')
    ax1.legend(title='Configuration', loc='upper right')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    ax2 = axes[1]
    if 'Cpu_Pct' in df.columns:
        for i, config in enumerate(configs):
            config_data = df[df['IoT_Config'] == config]
            means = config_data.groupby('Base_Model')['Cpu_Pct'].mean()
            means = means.reindex(models).fillna(0)
            ax2.bar(x + i * width, means.values, width,
                    label=config.replace('Core', ' Core, ').replace('MB', ' MB').replace('GB', ' GB'),
                    color=IOT_COLORS[config], edgecolor='black', linewidth=0.5, alpha=0.85)

    ax2.set_xticks(x + width)
    ax2.set_xticklabels([get_model_display_name(m) for m in models], rotation=45, ha='right')
    ax2.set_ylabel('CPU Utilization (%)')
    ax2.set_title('CPU Utilization by Configuration')
    ax2.legend(title='Configuration', loc='upper right')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_iot_scaling(df, output_dir, filename='iot_scaling.pdf'):
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
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_iot_vs_base_ratio(base_df, iot_df, output_dir, filename='iot_vs_base_ratio.pdf'):
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
    colors = [get_model_color(m) for m in ratio.index]

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
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


# =============================================================================
# GEM5 VISUALIZATIONS
# =============================================================================

def plot_gem5_cycles_comparison(gem5_perf, gem5_hw, output_dir, filename='gem5_cycles.pdf'):
    """Bar chart comparing CPU cycles from gem5 simulation."""
    if gem5_perf.empty and gem5_hw.empty:
        print("Skipping gem5 cycles: No data available")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Use hardware summary if available (more accurate)
    if not gem5_hw.empty and 'Cycles' in gem5_hw.columns:
        data = gem5_hw.sort_values('Cycles')
        models = data['Model'].values
        cycles = data['Cycles'].values / 1e6  # Convert to millions
        
        colors = [get_model_color(m) for m in models]
        ax.barh(range(len(models)), cycles, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([get_model_display_name(m) for m in models])
        ax.set_xlabel('CPU Cycles (millions)')
        ax.set_title('gem5 Simulation: CPU Cycles Comparison')
        
        for i, v in enumerate(cycles):
            ax.text(v + max(cycles) * 0.02, i, f'{v:.1f}M', va='center', fontsize=7)
    else:
        # Fall back to performance CSV data
        stats = gem5_perf.groupby('Model')['Total_ns'].agg(['mean']).sort_values('mean')
        models = stats.index
        means = stats['mean'].values / 1000  # μs
        
        colors = [get_model_color(m) for m in models]
        ax.barh(range(len(models)), means, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([get_model_display_name(m) for m in models])
        ax.set_xlabel('Execution Time (μs)')
        ax.set_title('gem5 Simulation: Execution Time Comparison')

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_gem5_cpi_ipc(gem5_raw, output_dir, filename='gem5_cpi_ipc.pdf'):
    """Plot CPI and IPC from gem5 raw stats."""
    if gem5_raw.empty or 'CPI' not in gem5_raw.columns:
        print("Skipping gem5 CPI/IPC: No data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    data = gem5_raw.sort_values('CPI')
    models = data['Model'].values

    # Plot CPI
    ax1 = axes[0]
    ax1.barh(range(len(models)), data['CPI'].values, color=GEM5_COLORS['cpi'], 
             edgecolor='black', linewidth=0.5, alpha=0.85)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels([get_model_display_name(m) for m in models])
    ax1.set_xlabel('Cycles Per Instruction (CPI)')
    ax1.set_title('CPI Comparison (Lower = Better)')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, v in enumerate(data['CPI'].values):
        ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=7)

    # Plot IPC
    ax2 = axes[1]
    data_ipc = gem5_raw.sort_values('IPC', ascending=False)
    models_ipc = data_ipc['Model'].values
    ax2.barh(range(len(models_ipc)), data_ipc['IPC'].values, color=GEM5_COLORS['ipc'],
             edgecolor='black', linewidth=0.5, alpha=0.85)
    ax2.set_yticks(range(len(models_ipc)))
    ax2.set_yticklabels([get_model_display_name(m) for m in models_ipc])
    ax2.set_xlabel('Instructions Per Cycle (IPC)')
    ax2.set_title('IPC Comparison (Higher = Better)')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, v in enumerate(data_ipc['IPC'].values):
        ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_gem5_sim_seconds(gem5_hw, output_dir, filename='gem5_sim_seconds.pdf'):
    """Plot simulated seconds from gem5."""
    if gem5_hw.empty or 'Sim_Seconds' not in gem5_hw.columns:
        print("Skipping gem5 sim seconds: No data available")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    data = gem5_hw.sort_values('Sim_Seconds')
    models = data['Model'].values
    sim_seconds = data['Sim_Seconds'].values
    colors = [get_model_color(m) for m in models]

    ax.barh(range(len(models)), sim_seconds, color=GEM5_COLORS['sim_seconds'],
            edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([get_model_display_name(m) for m in models])
    ax.set_xlabel('Simulated Seconds')
    ax.set_title('gem5 Simulation: Simulated Time')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, v in enumerate(sim_seconds):
        if v < 0.1:
            label = f'{v*1000:.2f}ms'
        else:
            label = f'{v:.3f}s'
        ax.text(v + max(sim_seconds) * 0.02, i, label, va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_gem5_vs_base(base_df, gem5_df, output_dir, filename='gem5_vs_base.pdf'):
    """Compare gem5 simulation results vs base machine."""
    if base_df.empty or gem5_df.empty:
        print("Skipping gem5 vs base: Missing data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    base_means = base_df.groupby('Model')['Total_ns'].mean()
    gem5_means = gem5_df.groupby('Model')['Total_ns'].mean()
    common_models = base_means.index.intersection(gem5_means.index)

    if len(common_models) == 0:
        print("Skipping gem5 vs base: No common models")
        plt.close()
        return

    x = np.arange(len(common_models))
    width = 0.35

    base_vals = base_means.loc[common_models].values / 1000
    gem5_vals = gem5_means.loc[common_models].values / 1000

    ax.bar(x - width/2, base_vals, width, label='Base Machine', color=COLORS['standalone'], 
           edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.bar(x + width/2, gem5_vals, width, label='gem5 Simulation', color=COLORS['hybrid'],
           edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([get_model_display_name(m) for m in common_models], rotation=45, ha='right')
    ax.set_ylabel('Execution Time (μs)')
    ax.set_title('Base Machine vs gem5 Simulation')
    ax.legend(loc='upper right')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: f'{y:,.0f}'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_gem5_operation_breakdown(gem5_df, output_dir, filename='gem5_operation_breakdown.pdf'):
    """Stacked bar chart for gem5 operation breakdown."""
    if gem5_df.empty:
        print("Skipping gem5 operation breakdown: No data available")
        return

    plot_operation_breakdown(gem5_df, output_dir, filename)


def plot_all_environments_comparison(base_df, iot_df, gem5_df, output_dir, filename='all_environments.pdf'):
    """Compare execution times across all three environments."""
    # Find common models
    dfs = {'Base Machine': base_df, 'IoT Simulation': iot_df, 'gem5 Simulation': gem5_df}
    valid_dfs = {k: v for k, v in dfs.items() if not v.empty}
    
    if len(valid_dfs) < 2:
        print("Skipping all environments comparison: Need at least 2 environments")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get means for each environment
    means_dict = {}
    for env_name, df in valid_dfs.items():
        means_dict[env_name] = df.groupby('Model')['Total_ns'].mean() / 1000

    # Find common models across all valid environments
    common_models = None
    for means in means_dict.values():
        if common_models is None:
            common_models = set(means.index)
        else:
            common_models = common_models.intersection(set(means.index))
    
    if not common_models:
        print("Skipping all environments comparison: No common models")
        plt.close()
        return

    common_models = sorted(list(common_models))
    x = np.arange(len(common_models))
    width = 0.8 / len(valid_dfs)

    env_colors = {'Base Machine': '#2E86AB', 'IoT Simulation': '#E63946', 'gem5 Simulation': '#2D6A4F'}

    for i, (env_name, means) in enumerate(means_dict.items()):
        vals = [means.loc[m] if m in means.index else 0 for m in common_models]
        offset = (i - len(valid_dfs)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=env_name, color=env_colors.get(env_name, COLORS['neutral']),
               edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([get_model_display_name(m) for m in common_models], rotation=45, ha='right')
    ax.set_ylabel('Execution Time (μs)')
    ax.set_title('Performance Comparison Across Environments')
    ax.legend(loc='upper right')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: f'{y:,.0f}'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")
