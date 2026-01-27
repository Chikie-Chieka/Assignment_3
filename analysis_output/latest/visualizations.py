#!/usr/bin/env python3
"""
Visualization Module for 2Core1GB PQC Benchmarking Analysis
Simplified version for single dataset analysis
Includes native vs gem5 comparison plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    'hybrid': '#A23B72',      # Deep magenta
    'classical': '#F18F01',   # Amber
    'pqc': '#C73E1D',         # Vermillion
    'ascon': '#3B7A57',       # Forest green
    'highlight': '#E63946',   # Red highlight
    'native': '#1F77B4',      # Native measurement
    'gem5': '#FF7F0E',        # gem5 simulation
    'neutral': '#6C757D',     # Gray
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


def get_model_display_name(model, short=False):
    """Convert model name to display-friendly format."""
    if short:
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
    """Get color for a model."""
    base_model = model.replace('Standalone_', '').replace('Hybrid_', 'Hybrid_')
    return MODEL_COLORS.get(base_model, COLORS['neutral'])


def _build_gem5_comparison_frame(perf_df, gem5_df):
    """Prepare aligned mean execution times for native vs gem5 comparison."""
    if perf_df.empty or gem5_df.empty:
        return pd.DataFrame()

    native = perf_df.groupby('Model')['Total_ns'].mean()
    gem5 = gem5_df.groupby('Model')['Total_ns'].mean()

    comp = pd.concat([native, gem5], axis=1, keys=['native_ns', 'gem5_ns'])
    comp = comp.dropna()
    if comp.empty:
        return comp

    comp['native_us'] = comp['native_ns'] / 1000
    comp['gem5_us'] = comp['gem5_ns'] / 1000
    return comp


# =============================================================================
# PERFORMANCE VISUALIZATIONS
# =============================================================================

def plot_execution_time_comparison(df, output_dir, filename='exec_time_2core1gb.pdf'):
    """Bar chart comparing mean execution times with error bars."""
    fig, ax = plt.subplots(figsize=(7, 4))

    stats = df.groupby('Model')['Total_ns'].agg(['mean', 'std', 'count'])
    stats['se'] = stats['std'] / np.sqrt(stats['count'])
    stats['ci_95'] = 1.96 * stats['se']
    stats = stats.sort_values('mean')

    stats['mean_us'] = stats['mean'] / 1000
    stats['ci_95_us'] = stats['ci_95'] / 1000

    colors = [get_model_color(m) for m in stats.index]

    ax.barh(range(len(stats)), stats['mean_us'], xerr=stats['ci_95_us'],
            color=colors, edgecolor='black', linewidth=0.5, capsize=3, alpha=0.85)

    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels([get_model_display_name(m) for m in stats.index])
    ax.set_xlabel('Execution Time (μs)')
    ax.set_title('Mean Execution Time - 2Core 1GB Configuration')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    for i, (idx, row) in enumerate(stats.iterrows()):
        ax.text(row['mean_us'] + row['ci_95_us'] + stats['mean_us'].max() * 0.02, i,
                f'{row["mean_us"]:,.1f}', va='center', fontsize=7)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_operation_breakdown(df, output_dir, filename='operation_breakdown_2core1gb.pdf'):
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
    ax.set_title('Cryptographic Operation Breakdown - 2Core 1GB')
    ax.legend(loc='lower right', framealpha=0.9, ncol=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_boxplot_comparison(df, output_dir, filename='boxplot_2core1gb.pdf'):
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
    ax.set_title('Execution Time Distribution - 2Core 1GB')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_violin_comparison(df, output_dir, filename='violin_2core1gb.pdf'):
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
    ax.set_xticklabels([get_model_display_name(m, short=True) for m in filtered_models], rotation=45, ha='right')
    ax.set_ylabel('Execution Time (μs)')
    ax.set_title('Execution Time Distribution (Violin Plot) - 2Core 1GB')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    legend_elements = [Line2D([0], [0], color='red', linewidth=2, label='Mean'),
                       Line2D([0], [0], color='black', linewidth=2, label='Median')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_performance_heatmap(df, output_dir, filename='heatmap_2core1gb.pdf'):
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

    ax.set_title('Normalized Performance Comparison - 2Core 1GB')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_standalone_vs_hybrid(df, output_dir, filename='standalone_vs_hybrid_2core1gb.pdf'):
    """Compare standalone PQC vs hybrid constructions."""
    fig, ax = plt.subplots(figsize=(7, 5))

    standalone = df[df['Model'].str.startswith('Standalone_')]
    hybrid = df[df['Model'].str.startswith('Hybrid_')]

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
    ax.set_title('Standalone vs Hybrid Construction - 2Core 1GB')
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


def plot_cv_comparison(df, output_dir, filename='cv_comparison_2core1gb.pdf'):
    """Bar chart of coefficient of variation (timing consistency)."""
    fig, ax = plt.subplots(figsize=(7, 4))

    cv_data = df.groupby('Model')['Total_ns'].agg(
        lambda x: (x.std() / x.mean()) * 100 if x.mean() != 0 else 0
    ).sort_values()

    colors = [get_model_color(m) for m in cv_data.index]

    ax.barh(range(len(cv_data)), cv_data.values,
            color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_yticks(range(len(cv_data)))
    ax.set_yticklabels([get_model_display_name(m) for m in cv_data.index])
    ax.set_xlabel('Coefficient of Variation (%)')
    ax.set_title('Timing Consistency - 2Core 1GB (Lower = More Consistent)')

    for i, v in enumerate(cv_data.values):
        ax.text(v + cv_data.max() * 0.02, i, f'{v:.1f}%', va='center', fontsize=7)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_cycle_count_comparison(df, output_dir, filename='cycle_count_2core1gb.pdf'):
    """Bar chart comparing CPU cycle counts."""
    if 'Cycle_Count' not in df.columns:
        print("Skipping cycle count: Column not found")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    stats = df.groupby('Model')['Cycle_Count'].agg(['mean', 'std'])
    stats = stats.sort_values('mean')

    colors = [get_model_color(m) for m in stats.index]

    ax.barh(range(len(stats)), stats['mean'] / 1e6,
            color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels([get_model_display_name(m) for m in stats.index])
    ax.set_xlabel('CPU Cycles (millions)')
    ax.set_title('CPU Cycle Count Comparison - 2Core 1GB')

    for i, (idx, row) in enumerate(stats.iterrows()):
        ax.text(row['mean'] / 1e6 + stats['mean'].max() / 1e6 * 0.02, i,
                f'{row["mean"]/1e6:.2f}M', va='center', fontsize=7)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_cpu_utilization(df, output_dir, filename='cpu_utilization_2core1gb.pdf'):
    """Bar chart comparing CPU utilization percentage."""
    if 'Cpu_Pct' not in df.columns:
        print("Skipping CPU utilization: Column not found")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    stats = df.groupby('Model')['Cpu_Pct'].agg(['mean', 'std'])
    stats = stats.sort_values('mean')

    colors = [get_model_color(m) for m in stats.index]

    ax.barh(range(len(stats)), stats['mean'],
            color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels([get_model_display_name(m) for m in stats.index])
    ax.set_xlabel('CPU Utilization (%)')
    ax.set_title('CPU Utilization Comparison - 2Core 1GB')
    ax.set_xlim(0, 100)

    for i, (idx, row) in enumerate(stats.iterrows()):
        ax.text(row['mean'] + 2, i, f'{row["mean"]:.1f}%', va='center', fontsize=7)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_gem5_vs_native_execution_time(perf_df, gem5_df, output_dir,
                                       filename='gem5_vs_native_exec_time_2core1gb.pdf'):
    """Grouped bar chart comparing native vs gem5 mean execution time."""
    comp = _build_gem5_comparison_frame(perf_df, gem5_df)
    if comp.empty:
        print("Skipping gem5 vs native execution time: insufficient data")
        return

    comp = comp.sort_values('native_us')
    y = np.arange(len(comp))
    bar_h = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y - bar_h / 2, comp['native_us'], height=bar_h,
            color=COLORS['native'], edgecolor='black', linewidth=0.4, label='Native')
    ax.barh(y + bar_h / 2, comp['gem5_us'], height=bar_h,
            color=COLORS['gem5'], edgecolor='black', linewidth=0.4, label='gem5')

    ax.set_yticks(y)
    ax.set_yticklabels([get_model_display_name(m) for m in comp.index])
    ax.set_xlabel('Mean Execution Time (us)')
    ax.set_title('Native vs gem5 Mean Execution Time - 2Core 1GB')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_gem5_vs_native_ratio(perf_df, gem5_df, output_dir,
                              filename='gem5_vs_native_ratio_2core1gb.pdf'):
    """Bar chart of gem5/native ratio (values < 1 mean gem5 faster)."""
    comp = _build_gem5_comparison_frame(perf_df, gem5_df)
    if comp.empty:
        print("Skipping gem5 vs native ratio: insufficient data")
        return

    comp['ratio'] = comp['gem5_us'] / comp['native_us']
    comp = comp.sort_values('ratio')

    colors = [COLORS['standalone'] if r <= 1 else COLORS['highlight'] for r in comp['ratio']]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(range(len(comp)), comp['ratio'], color=colors,
            edgecolor='black', linewidth=0.4, alpha=0.85)
    ax.axvline(1.0, color='black', linestyle='--', linewidth=0.8)
    ax.set_yticks(range(len(comp)))
    ax.set_yticklabels([get_model_display_name(m) for m in comp.index])
    ax.set_xlabel('gem5 / Native Mean Time (ratio)')
    ax.set_title('gem5 vs Native Speed Ratio - 2Core 1GB')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, r in enumerate(comp['ratio'].values):
        ax.text(r + 0.02, i, f'{r:.2f}', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_gem5_scatter(perf_df, gem5_df, output_dir,
                      filename='gem5_vs_native_scatter_2core1gb.pdf'):
    """Scatter plot comparing native vs gem5 mean execution time."""
    comp = _build_gem5_comparison_frame(perf_df, gem5_df)
    if comp.empty:
        print("Skipping gem5 scatter: insufficient data")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    max_val = max(comp['native_us'].max(), comp['gem5_us'].max())
    min_val = min(comp['native_us'].min(), comp['gem5_us'].min())

    for model, row in comp.iterrows():
        ax.scatter(row['native_us'], row['gem5_us'],
                   color=get_model_color(model), s=45, edgecolor='black', linewidth=0.4, alpha=0.85)
        ax.text(row['native_us'], row['gem5_us'],
                get_model_display_name(model, short=True), fontsize=7, ha='left', va='bottom')

    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', linewidth=0.8)
    ax.set_xlabel('Native Mean Execution Time (us)')
    ax.set_ylabel('gem5 Mean Execution Time (us)')
    ax.set_title('Native vs gem5 Mean Execution Time')

    if max_val / max(min_val, 1e-6) > 50:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_gem5_memory_usage(gem5_df, output_dir, filename='gem5_memory_usage_2core1gb.pdf'):
    """Compare gem5 peak memory usage across models (or cycle count if memory columns are absent)."""
    if gem5_df.empty:
        print("Skipping gem5 memory usage: no gem5 data")
        return

    has_memory = 'Peak_RSS_KB' in gem5_df.columns and 'Peak_Alloc_KB' in gem5_df.columns
    has_cycles = 'Cycle_Count' in gem5_df.columns

    if not has_memory and not has_cycles:
        print("Skipping gem5 memory usage: columns not available")
        return

    if not has_memory and has_cycles:
        cycle_filename = filename
        if filename == 'gem5_memory_usage_2core1gb.pdf':
            cycle_filename = 'gem5_cycle_count_2core1gb.pdf'

        stats = gem5_df.groupby('Model')['Cycle_Count'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(range(len(stats)), stats.values / 1e6,
                color=COLORS['gem5'], edgecolor='black', linewidth=0.4, alpha=0.85)
        ax.set_yticks(range(len(stats)))
        ax.set_yticklabels([get_model_display_name(m) for m in stats.index])
        ax.set_xlabel('CPU Cycles (millions)')
        ax.set_title('gem5 Cycle Count - 2Core 1GB')
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        for i, val in enumerate(stats.values):
            ax.text(val / 1e6 + stats.max() / 1e6 * 0.02, i,
                    f'{val/1e6:.2f}M', va='center', fontsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, cycle_filename))
        plt.close()
        print(f"Saved: {cycle_filename}")
        return

    stats = gem5_df.groupby('Model')[['Peak_Alloc_KB', 'Peak_RSS_KB']].mean()
    stats = stats.sort_values('Peak_RSS_KB')

    y = np.arange(len(stats))
    bar_h = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y - bar_h / 2, stats['Peak_Alloc_KB'], height=bar_h,
            color=COLORS['standalone'], edgecolor='black', linewidth=0.4, label='Peak Alloc (KB)')
    ax.barh(y + bar_h / 2, stats['Peak_RSS_KB'], height=bar_h,
            color=COLORS['hybrid'], edgecolor='black', linewidth=0.4, label='Peak RSS (KB)')

    ax.set_yticks(y)
    ax.set_yticklabels([get_model_display_name(m) for m in stats.index])
    ax.set_xlabel('Memory (KB)')
    ax.set_title('gem5 Peak Memory Usage - 2Core 1GB')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


# =============================================================================
# ENTROPY VISUALIZATIONS
# =============================================================================

def plot_entropy_analysis(df, output_dir, filename='entropy_analysis_2core1gb.pdf'):
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
        color = get_model_color(model)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)

    ax1.axhline(y=8.0, color='red', linestyle='--', linewidth=1, label='Ideal (8.0)')
    ax1.set_xticks(positions)
    ax1.set_xticklabels([get_model_display_name(m, short=True) for m in models], rotation=45, ha='right')
    ax1.set_ylabel('Entropy (bits/byte)')
    ax1.set_title('Entropy Quality - 2Core 1GB')
    ax1.set_ylim(7.9999, 8.0001)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Serial correlation coefficient
    ax2 = axes[1]
    for i, model in enumerate(models):
        data = df[df['Model'] == model]['Serial_Correlation_Coefficient']
        bp = ax2.boxplot([data.values], positions=[i], widths=0.6,
                         patch_artist=True, showfliers=False)
        color = get_model_color(model)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)

    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Ideal (0.0)')
    ax2.set_xticks(positions)
    ax2.set_xticklabels([get_model_display_name(m, short=True) for m in models], rotation=45, ha='right')
    ax2.set_ylabel('Serial Correlation Coefficient')
    ax2.set_title('Randomness Quality - 2Core 1GB')
    ax2.legend(loc='lower right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_entropy_distribution(df, output_dir, filename='entropy_distribution_2core1gb.pdf'):
    """Detailed entropy distribution violin plot."""
    if df.empty or 'Entropy_Bits_Per_Byte' not in df.columns:
        print("Skipping entropy distribution: No data available")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    models = df['Model'].unique()
    data_to_plot = []
    for model in models:
        data = df[df['Model'] == model]['Entropy_Bits_Per_Byte'].values
        if len(data) >= 2:
            data_to_plot.append(data)

    if not data_to_plot:
        print("Skipping entropy distribution: insufficient data")
        plt.close()
        return

    parts = ax.violinplot(data_to_plot, positions=range(len(models)),
                          showmeans=True, showmedians=True, widths=0.7)

    for i, (pc, model) in enumerate(zip(parts['bodies'], models)):
        color = get_model_color(model)
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    parts['cmeans'].set_color('red')
    parts['cmedians'].set_color('black')

    ax.axhline(y=8.0, color='green', linestyle='--', linewidth=1, label='Ideal (8.0)')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([get_model_display_name(m, short=True) for m in models], rotation=45, ha='right')
    ax.set_ylabel('Entropy (bits/byte)')
    ax.set_title('Entropy Distribution - 2Core 1GB')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_summary_dashboard(perf_df, ent_df, output_dir, filename='summary_dashboard_2core1gb.pdf'):
    """Create a summary dashboard with key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Execution Time
    ax1 = axes[0, 0]
    stats = perf_df.groupby('Model')['Total_ns'].mean().sort_values() / 1000
    colors = [get_model_color(m) for m in stats.index]
    ax1.barh(range(len(stats)), stats.values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(stats)))
    ax1.set_yticklabels([get_model_display_name(m, short=True) for m in stats.index])
    ax1.set_xlabel('Mean Execution Time (μs)')
    ax1.set_title('Performance Ranking')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Plot 2: CV (Consistency)
    ax2 = axes[0, 1]
    cv_data = perf_df.groupby('Model')['Total_ns'].agg(
        lambda x: (x.std() / x.mean()) * 100 if x.mean() != 0 else 0
    ).sort_values()
    colors = [get_model_color(m) for m in cv_data.index]
    ax2.barh(range(len(cv_data)), cv_data.values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(cv_data)))
    ax2.set_yticklabels([get_model_display_name(m, short=True) for m in cv_data.index])
    ax2.set_xlabel('Coefficient of Variation (%)')
    ax2.set_title('Timing Consistency (Lower = Better)')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # Plot 3: Cycle Count
    ax3 = axes[1, 0]
    if 'Cycle_Count' in perf_df.columns:
        cycle_stats = perf_df.groupby('Model')['Cycle_Count'].mean().sort_values() / 1e6
        colors = [get_model_color(m) for m in cycle_stats.index]
        ax3.barh(range(len(cycle_stats)), cycle_stats.values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax3.set_yticks(range(len(cycle_stats)))
        ax3.set_yticklabels([get_model_display_name(m, short=True) for m in cycle_stats.index])
        ax3.set_xlabel('CPU Cycles (millions)')
        ax3.set_title('CPU Cycle Count')
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
    else:
        ax3.text(0.5, 0.5, 'Cycle data not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('CPU Cycle Count')

    # Plot 4: Entropy Quality
    ax4 = axes[1, 1]
    if not ent_df.empty and 'Entropy_Bits_Per_Byte' in ent_df.columns:
        ent_stats = ent_df.groupby('Model')['Entropy_Bits_Per_Byte'].mean().sort_values(ascending=False)
        colors = [get_model_color(m) for m in ent_stats.index]
        ax4.barh(range(len(ent_stats)), ent_stats.values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax4.set_yticks(range(len(ent_stats)))
        ax4.set_yticklabels([get_model_display_name(m, short=True) for m in ent_stats.index])
        ax4.set_xlabel('Entropy (bits/byte)')
        ax4.set_title('Entropy Quality (Ideal = 8.0)')
        ax4.axvline(x=8.0, color='red', linestyle='--', linewidth=1)
        ax4.set_xlim(7.9999, 8.0001)
        ax4.grid(axis='x', alpha=0.3, linestyle='--')
    else:
        ax4.text(0.5, 0.5, 'Entropy data not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Entropy Quality')

    plt.suptitle('2Core 1GB Configuration - Performance Summary Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {filename}")
