#!/usr/bin/env python3
"""
Statistical Analysis for 2Core1GB PQC Benchmarking
Simplified version for single dataset analysis

Analyzes:
- testing_process_2Core1GB.csv (performance data)
- testing_process_ent_2Core1GB.csv (entropy data)
- gem5_tests/* or saved_output/gem5_sim_results/* (gem5 performance comparisons)
"""

import os
import glob
import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import shapiro, kruskal, mannwhitneyu

warnings.filterwarnings('ignore')

# Import visualization functions from local module
from visualizations import (
    plot_execution_time_comparison,
    plot_operation_breakdown,
    plot_boxplot_comparison,
    plot_violin_comparison,
    plot_performance_heatmap,
    plot_standalone_vs_hybrid,
    plot_cv_comparison,
    plot_cycle_count_comparison,
    plot_cpu_utilization,
    plot_entropy_analysis,
    plot_entropy_distribution,
    plot_summary_dashboard,
    plot_gem5_vs_native_execution_time,
    plot_gem5_vs_native_ratio,
    plot_gem5_scatter,
    plot_gem5_memory_usage,
)

# Output directory (same folder as script)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, '..', '..'))

# Data files
DATA_DIR = os.path.join(ROOT_DIR, 'saved_output')
PERF_FILE = os.path.join(DATA_DIR, 'testing_process_2Core1GB.csv')
ENT_FILE = os.path.join(DATA_DIR, 'testing_process_ent_2Core1GB.csv')
GEM5_DIR_CANDIDATES = [
    os.path.join(ROOT_DIR, 'gem5_tests'),
    os.path.join(DATA_DIR, 'gem5_sim_results'),
]


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_performance_data():
    """Load performance CSV file."""
    if not os.path.exists(PERF_FILE):
        print(f"  Warning: Performance file not found: {PERF_FILE}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(PERF_FILE)
        print(f"  Performance data: {len(df)} records")
        return df
    except Exception as e:
        print(f"  Error loading performance data: {e}")
        return pd.DataFrame()


def load_entropy_data():
    """Load entropy test CSV file."""
    if not os.path.exists(ENT_FILE):
        print(f"  Warning: Entropy file not found: {ENT_FILE}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(ENT_FILE)
        print(f"  Entropy data: {len(df)} records")
        return df
    except Exception as e:
        print(f"  Error loading entropy data: {e}")
        return pd.DataFrame()


def clean_performance_df(df):
    """Filter failed rows and coerce numeric columns for performance data."""
    if df.empty:
        return df

    cleaned = df.copy()
    if 'Failed' in cleaned.columns:
        cleaned['Failed'] = pd.to_numeric(cleaned['Failed'], errors='coerce').fillna(0)
        cleaned = cleaned[cleaned['Failed'] == 0]

    numeric_cols = [
        'KeyGen_ns', 'Encaps_ns', 'Decaps_ns', 'KDF_ns',
        'Encryption_ns', 'Decryption_ns', 'Total_ns', 'Total_s',
        'Cpu_Pct', 'Cycle_Count', 'Peak_Alloc_KB', 'Peak_RSS_KB'
    ]
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')

    if 'Total_ns' in cleaned.columns:
        cleaned = cleaned.dropna(subset=['Total_ns'])

    return cleaned


def clean_entropy_df(df):
    """Filter failed rows and coerce numeric columns for entropy data."""
    if df.empty:
        return df

    cleaned = df.copy()
    if 'Status' in cleaned.columns:
        cleaned = cleaned[cleaned['Status'].str.lower() == 'success']

    numeric_cols = ['Entropy_Bytes', 'Entropy_Bits_Per_Byte', 'Serial_Correlation_Coefficient']
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')

    return cleaned


def select_gem5_dir():
    """Select the first available gem5 directory with data."""
    for path in GEM5_DIR_CANDIDATES:
        if os.path.isdir(path):
            return path
    return None


def load_gem5_performance_data(gem5_dir):
    """Load and merge gem5 performance CSV files."""
    if not gem5_dir or not os.path.isdir(gem5_dir):
        print("  Warning: gem5 directory not found")
        return pd.DataFrame()

    pattern = os.path.join(gem5_dir, 'testing_process_*.csv')
    files = [p for p in glob.glob(pattern) if 'testing_process_ent_' not in os.path.basename(p)]

    frames = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  Warning: Could not read gem5 file {os.path.basename(path)}: {e}")
            continue

        if df.empty:
            continue
        df['Source'] = 'gem5'
        df['Source_File'] = os.path.basename(path)
        frames.append(df)

    if not frames:
        print("  Warning: No gem5 performance data found")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    print(f"  gem5 performance data: {len(merged)} records from {len(frames)} files")
    return merged


def load_gem5_latency_report(report_path):
    """Parse FINAL_CRYPTO_HARDWARE_REPORT.txt for latency summary."""
    if not report_path or not os.path.exists(report_path):
        return pd.DataFrame()

    rows = []
    line_re = re.compile(r'\s*\|\s*')
    with open(GEM5_REPORT, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if '|' not in line or line.strip().startswith('Type') or line.strip().startswith('-'):
                continue
            parts = [p.strip() for p in line_re.split(line.strip()) if p.strip()]
            if len(parts) < 4:
                continue
            entry_type, model, cycles, sim_seconds = parts[:4]
            if entry_type.upper() != 'LATENCY':
                continue
            try:
                cycles_val = float(cycles)
                seconds_val = float(sim_seconds)
            except ValueError:
                continue
            rows.append({
                'Model': model,
                'Cpu_Cycles': cycles_val,
                'Sim_Seconds': seconds_val
            })

    if not rows:
        print("  Warning: gem5 latency report contained no usable rows")
        return pd.DataFrame()

    return pd.DataFrame(rows)


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
        ('cv', lambda x: (x.std() / x.mean()) * 100 if x.mean() != 0 else 0),
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
            stat, p_value = shapiro(data[:5000])  # Shapiro-Wilk limit
            results[model] = {'statistic': stat, 'p_value': p_value, 'normal': p_value > 0.05}
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
            data1 = np.asarray(df[df[group_col] == model1][value_col].values, dtype=np.float64)
            data2 = np.asarray(df[df[group_col] == model2][value_col].values, dtype=np.float64)
            stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            corrected_p = min(p_value * n_comparisons, 1.0)  # Bonferroni
            results.append({
                'model1': model1,
                'model2': model2,
                'statistic': stat,
                'p_value': p_value,
                'corrected_p': corrected_p,
                'significant': corrected_p < 0.05
            })

    return pd.DataFrame(results)


def compute_entropy_stats(df):
    """Compute entropy statistics."""
    if df.empty or 'Entropy_Bits_Per_Byte' not in df.columns:
        return None
    
    return df.groupby('Model').agg({
        'Entropy_Bits_Per_Byte': ['mean', 'std', 'min', 'max'],
        'Serial_Correlation_Coefficient': ['mean', 'std', 'min', 'max'],
    }).round(6)


def compute_gem5_comparison_stats(native_df, gem5_df):
    """Compare mean execution times between native and gem5 data."""
    if native_df.empty or gem5_df.empty:
        return None, None
    if 'Total_ns' not in native_df.columns or 'Total_ns' not in gem5_df.columns:
        return None, None

    native_means = native_df.groupby('Model')['Total_ns'].mean()
    gem5_means = gem5_df.groupby('Model')['Total_ns'].mean()

    comparison = pd.concat([native_means, gem5_means], axis=1, keys=['native_mean_ns', 'gem5_mean_ns'])
    comparison = comparison.dropna()
    if comparison.empty:
        return None, None

    comparison['native_mean_us'] = comparison['native_mean_ns'] / 1000
    comparison['gem5_mean_us'] = comparison['gem5_mean_ns'] / 1000
    comparison['ratio'] = comparison['gem5_mean_ns'] / comparison['native_mean_ns']
    comparison['pct_diff'] = (comparison['gem5_mean_ns'] - comparison['native_mean_ns']) / comparison['native_mean_ns'] * 100
    comparison = comparison.sort_values('ratio')

    pearson = comparison[['native_mean_ns', 'gem5_mean_ns']].corr(method='pearson').iloc[0, 1]
    spearman = comparison[['native_mean_ns', 'gem5_mean_ns']].corr(method='spearman').iloc[0, 1]

    return comparison, {'pearson': float(pearson), 'spearman': float(spearman)}


def compute_gem5_cycle_comparison(native_df, gem5_latency_df, gem5_perf_df=None):
    """Compare native cycle counts with gem5 cycles from latency report or gem5 CSV."""
    if native_df.empty or 'Cycle_Count' not in native_df.columns:
        return None, None

    native_cycles = native_df.groupby('Model')['Cycle_Count'].mean()

    source = None
    if gem5_latency_df is not None and not gem5_latency_df.empty:
        gem5_cycles = gem5_latency_df.set_index('Model')['Cpu_Cycles']
        source = 'latency_report'
    elif gem5_perf_df is not None and not gem5_perf_df.empty and 'Cycle_Count' in gem5_perf_df.columns:
        gem5_cycles = gem5_perf_df.groupby('Model')['Cycle_Count'].mean()
        source = 'gem5_csv'
    else:
        return None, None

    comparison = pd.concat([native_cycles, gem5_cycles], axis=1, keys=['native_mean_cycles', 'gem5_cycles'])
    comparison = comparison.dropna()
    if comparison.empty:
        return None, None

    comparison['ratio'] = comparison['gem5_cycles'] / comparison['native_mean_cycles']
    comparison['pct_diff'] = (comparison['gem5_cycles'] - comparison['native_mean_cycles']) / comparison['native_mean_cycles'] * 100
    comparison = comparison.sort_values('ratio')
    return comparison, source


# =============================================================================
# REPORT GENERATION
# =============================================================================

def save_statistics_report(
    perf_stats,
    ent_stats,
    kw_results,
    pairwise_results,
    perf_df,
    ent_df,
    gem5_df=None,
    gem5_comparison=None,
    gem5_corr=None,
    gem5_cycle_comparison=None,
    gem5_cycle_source=None,
    gem5_latency_df=None,
):
    """Save comprehensive statistics report."""
    report_path = os.path.join(OUTPUT_DIR, 'statistical_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL ANALYSIS REPORT: 2Core 1GB Configuration\n")
        f.write("=" * 80 + "\n\n")

        # Configuration info
        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write("System: 2 CPU Cores, 1GB RAM\n")
        f.write(f"Performance records: {len(perf_df)}\n")
        f.write(f"Entropy records: {len(ent_df)}\n")
        f.write(f"Models tested: {perf_df['Model'].nunique()}\n\n")
        if gem5_df is not None and not gem5_df.empty:
            f.write(f"gem5 performance records: {len(gem5_df)}\n")
            f.write(f"gem5 models covered: {gem5_df['Model'].nunique()}\n")
            standalone_count = gem5_df[gem5_df['Model'].str.startswith('Standalone_')]['Model'].nunique()
            hybrid_count = gem5_df[gem5_df['Model'].str.startswith('Hybrid_')]['Model'].nunique()
            f.write(f"gem5 standalone models: {standalone_count}\n")
            f.write(f"gem5 hybrid models: {hybrid_count}\n")
            if gem5_latency_df is not None and not gem5_latency_df.empty:
                f.write(f"gem5 latency report entries: {len(gem5_latency_df)}\n")
            f.write("\n")

        # Descriptive Statistics
        f.write("1. DESCRIPTIVE STATISTICS (Execution Time in nanoseconds)\n")
        f.write("-" * 80 + "\n")
        if perf_stats is not None:
            f.write(perf_stats.to_string())
        f.write("\n\n")

        # Performance Summary (in microseconds)
        f.write("2. PERFORMANCE SUMMARY (μs)\n")
        f.write("-" * 80 + "\n")
        if not perf_df.empty:
            summary = perf_df.groupby('Model')['Total_ns'].agg(['mean', 'std', 'median']).sort_values('mean')
            summary = summary / 1000  # Convert to μs
            summary['cv'] = (perf_df.groupby('Model')['Total_ns'].std() / 
                            perf_df.groupby('Model')['Total_ns'].mean() * 100)
            summary.columns = ['Mean (μs)', 'Std (μs)', 'Median (μs)', 'CV (%)']
            f.write(summary.round(2).to_string())
        f.write("\n\n")

        # CPU Cycle Statistics
        if 'Cycle_Count' in perf_df.columns:
            f.write("3. CPU CYCLE STATISTICS\n")
            f.write("-" * 80 + "\n")
            cycle_stats = perf_df.groupby('Model')['Cycle_Count'].agg(['mean', 'std', 'min', 'max'])
            cycle_stats.columns = ['Mean Cycles', 'Std', 'Min', 'Max']
            f.write(cycle_stats.round(0).to_string())
            f.write("\n\n")

        # CPU Utilization Statistics
        if 'Cpu_Pct' in perf_df.columns:
            f.write("4. CPU UTILIZATION STATISTICS (%)\n")
            f.write("-" * 80 + "\n")
            cpu_stats = perf_df.groupby('Model')['Cpu_Pct'].agg(['mean', 'std', 'min', 'max'])
            cpu_stats.columns = ['Mean %', 'Std', 'Min %', 'Max %']
            f.write(cpu_stats.round(2).to_string())
            f.write("\n\n")

        # Kruskal-Wallis test
        f.write("5. KRUSKAL-WALLIS H-TEST (Non-parametric ANOVA)\n")
        f.write("-" * 80 + "\n")
        if kw_results:
            f.write(f"H-statistic: {kw_results['statistic']:.4f}\n")
            f.write(f"p-value: {kw_results['p_value']:.2e}\n")
            f.write(f"Significant difference: {'Yes' if kw_results['significant'] else 'No'}\n")
        f.write("\n\n")

        # Pairwise comparisons (top significant)
        f.write("6. PAIRWISE MANN-WHITNEY U TESTS (Bonferroni corrected)\n")
        f.write("-" * 80 + "\n")
        if pairwise_results is not None and not pairwise_results.empty:
            sig_results = pairwise_results[pairwise_results['significant']].sort_values('corrected_p')
            if not sig_results.empty:
                f.write(f"Significant pairwise differences: {len(sig_results)}/{len(pairwise_results)}\n\n")
                f.write("Top 10 most significant comparisons:\n")
                f.write(sig_results.head(10).to_string(index=False))
            else:
                f.write("No significant pairwise differences found.\n")
        f.write("\n\n")

        # Entropy Statistics
        f.write("7. ENTROPY STATISTICS\n")
        f.write("-" * 80 + "\n")
        if ent_stats is not None:
            f.write(ent_stats.to_string())
            f.write("\n\nNote: Ideal entropy = 8.0 bits/byte, Ideal serial correlation = 0.0\n")
        else:
            f.write("No entropy data available.\n")
        f.write("\n\n")

        # Model Rankings
        f.write("8. MODEL RANKINGS\n")
        f.write("-" * 80 + "\n")
        if not perf_df.empty:
            rankings = perf_df.groupby('Model')['Total_ns'].mean().sort_values()
            f.write("\nBy Execution Time (fastest to slowest):\n")
            for i, (model, time) in enumerate(rankings.items(), 1):
                f.write(f"  {i}. {model}: {time/1000:.2f} μs\n")
        f.write("\n")

        # Standalone vs Hybrid Comparison
        f.write("9. STANDALONE vs HYBRID COMPARISON\n")
        f.write("-" * 80 + "\n")
        standalone = perf_df[perf_df['Model'].str.startswith('Standalone_')]['Total_ns']
        hybrid = perf_df[perf_df['Model'].str.startswith('Hybrid_')]['Total_ns']
        if not standalone.empty and not hybrid.empty:
            f.write(f"Standalone PQC - Mean: {standalone.mean()/1000:.2f} μs, Median: {standalone.median()/1000:.2f} μs\n")
            f.write(f"Hybrid PQC+AEAD - Mean: {hybrid.mean()/1000:.2f} μs, Median: {hybrid.median()/1000:.2f} μs\n")
            stat, p_value = mannwhitneyu(standalone.values, hybrid.values)
            f.write(f"Mann-Whitney U test: p = {p_value:.2e}\n")
        f.write("\n")

        # gem5 comparison
        f.write("10. GEM5 COMPARISON (Native vs gem5)\n")
        f.write("-" * 80 + "\n")
        if gem5_df is not None and not gem5_df.empty and gem5_comparison is not None:
            f.write(f"gem5 performance records: {len(gem5_df)}\n")
            f.write(f"Models compared: {gem5_comparison.shape[0]}\n")
            if gem5_corr:
                f.write(f"Correlation (Pearson): {gem5_corr['pearson']:.3f}\n")
                f.write(f"Correlation (Spearman): {gem5_corr['spearman']:.3f}\n")
            f.write("\nMean Execution Time Comparison (us):\n")
            comp_display = gem5_comparison[['native_mean_us', 'gem5_mean_us', 'ratio', 'pct_diff']].copy()
            comp_display.columns = ['Native Mean (us)', 'gem5 Mean (us)', 'gem5/native', 'Pct Diff (%)']
            f.write(comp_display.round(3).to_string())
            f.write("\n\nNote: gem5/native < 1 means gem5 is faster than native measurement.\n")
        else:
            f.write("No gem5 comparison data available.\n")
        f.write("\n")

        if gem5_cycle_comparison is not None and not gem5_cycle_comparison.empty:
            f.write("11. GEM5 CPU CYCLE COMPARISON (gem5 vs Native)\n")
            f.write("-" * 80 + "\n")
            if gem5_cycle_source:
                f.write(f"Source: {gem5_cycle_source}\n")
            cycle_display = gem5_cycle_comparison.copy()
            cycle_display.columns = ['Native Mean Cycles', 'gem5 Cycles', 'gem5/native', 'Pct Diff (%)']
            f.write(cycle_display.round(3).to_string())
            f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Saved: statistical_report.txt")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    print("=" * 60)
    print("2CORE 1GB PQC BENCHMARKING ANALYSIS")
    print("=" * 60)
    print()

    # Load data
    print("Loading data...")
    perf_df = load_performance_data()
    ent_df = load_entropy_data()

    perf_df = clean_performance_df(perf_df)
    ent_df = clean_entropy_df(ent_df)

    if perf_df.empty:
        print("Error: No performance data loaded. Exiting.")
        return

    # Load gem5 data
    gem5_dir = select_gem5_dir()
    gem5_perf_df = load_gem5_performance_data(gem5_dir)
    gem5_perf_df = clean_performance_df(gem5_perf_df)
    gem5_report = None
    if gem5_dir:
        candidate_report = os.path.join(gem5_dir, 'FINAL_CRYPTO_HARDWARE_REPORT.txt')
        if os.path.exists(candidate_report):
            gem5_report = candidate_report
    gem5_latency_df = load_gem5_latency_report(gem5_report)

    # Compute statistics
    print("\nComputing statistics...")
    perf_stats = compute_descriptive_stats(perf_df)
    print("  Descriptive statistics computed")

    kw_results = perform_kruskal_wallis(perf_df)
    if kw_results:
        print(f"  Kruskal-Wallis: H={kw_results['statistic']:.2f}, p={kw_results['p_value']:.2e}")

    pairwise_results = pairwise_mannwhitney(perf_df)
    print(f"  Pairwise Mann-Whitney: {len(pairwise_results)} comparisons")

    ent_stats = compute_entropy_stats(ent_df)
    if ent_stats is not None:
        print("  Entropy statistics computed")

    gem5_comparison, gem5_corr = compute_gem5_comparison_stats(perf_df, gem5_perf_df)
    gem5_cycle_comparison, gem5_cycle_source = compute_gem5_cycle_comparison(
        perf_df, gem5_latency_df, gem5_perf_df
    )

    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Performance visualizations
    plot_execution_time_comparison(perf_df, OUTPUT_DIR)
    plot_operation_breakdown(perf_df, OUTPUT_DIR)
    plot_boxplot_comparison(perf_df, OUTPUT_DIR)
    plot_violin_comparison(perf_df, OUTPUT_DIR)
    plot_performance_heatmap(perf_df, OUTPUT_DIR)
    plot_standalone_vs_hybrid(perf_df, OUTPUT_DIR)
    plot_cv_comparison(perf_df, OUTPUT_DIR)
    plot_cycle_count_comparison(perf_df, OUTPUT_DIR)
    plot_cpu_utilization(perf_df, OUTPUT_DIR)

    # Entropy visualizations
    if not ent_df.empty:
        plot_entropy_analysis(ent_df, OUTPUT_DIR)
        plot_entropy_distribution(ent_df, OUTPUT_DIR)

    # gem5 comparison visualizations
    if not gem5_perf_df.empty:
        plot_gem5_vs_native_execution_time(perf_df, gem5_perf_df, OUTPUT_DIR)
        plot_gem5_vs_native_ratio(perf_df, gem5_perf_df, OUTPUT_DIR)
        plot_gem5_scatter(perf_df, gem5_perf_df, OUTPUT_DIR)
        plot_gem5_memory_usage(gem5_perf_df, OUTPUT_DIR)

    # Summary dashboard
    plot_summary_dashboard(perf_df, ent_df, OUTPUT_DIR)

    # Save report
    print("\nSaving statistical report...")
    save_statistics_report(
        perf_stats,
        ent_stats,
        kw_results,
        pairwise_results,
        perf_df,
        ent_df,
        gem5_df=gem5_perf_df,
        gem5_comparison=gem5_comparison,
        gem5_corr=gem5_corr,
        gem5_cycle_comparison=gem5_cycle_comparison,
        gem5_cycle_source=gem5_cycle_source,
        gem5_latency_df=gem5_latency_df,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: 2CORE 1GB PERFORMANCE (μs)")
    print("=" * 60)
    summary = perf_df.groupby('Model')['Total_ns'].agg(['mean', 'std', 'median'])
    summary = summary / 1000
    summary['cv'] = (perf_df.groupby('Model')['Total_ns'].std() / 
                    perf_df.groupby('Model')['Total_ns'].mean() * 100)
    summary = summary.sort_values('mean')
    summary.columns = ['Mean (μs)', 'Std (μs)', 'Median (μs)', 'CV (%)']
    print(summary.round(2).to_string())

    print("\n" + "=" * 60)
    print(f"Analysis complete. Output saved to: {OUTPUT_DIR}/")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
