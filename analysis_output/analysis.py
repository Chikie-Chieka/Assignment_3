#!/usr/bin/env python3
"""
Statistical Analysis for Post-Quantum Cryptography Benchmarking
Generates publication-quality figures for Q1 conference papers

Supports:
- Base machine benchmarks
- IoT simulation results
- gem5 architectural simulation results
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, kruskal, mannwhitneyu, f_oneway
import re

warnings.filterwarnings('ignore')

# Import visualization functions
from analysis_output.visualizations import (
    plot_execution_time_comparison,
    plot_operation_breakdown,
    plot_boxplot_comparison,
    plot_violin_comparison,
    plot_memory_comparison,
    plot_performance_heatmap,
    plot_standalone_vs_hybrid,
    plot_cv_comparison,
    plot_entropy_analysis,
    plot_iot_comparison,
    plot_iot_scaling,
    plot_iot_vs_base_ratio,
    plot_gem5_cycles_comparison,
    plot_gem5_cpi_ipc,
    plot_gem5_sim_seconds,
    plot_gem5_vs_base,
    plot_gem5_operation_breakdown,
    plot_all_environments_comparison,
)

OUTPUT_DIR = 'analysis_output'

# Model ID to name mapping for gem5 stats directories
GEM5_MODEL_MAP = {
    'm1': 'Standalone_Ascon_80pq',
    'm2': 'Standalone_BIKE_L1',
    'm3': 'Standalone_Kyber512',
    'm4': 'Standalone_FrodoKEM_640_AES',
    'm5': 'Standalone_HQC_128',
    'm6': 'Standalone_ClassicMcEliece_348864',
    'm7': 'Standalone_X25519',
    'm8': 'Hybrid_BIKE_L1_Ascon128a',
    'm9': 'Hybrid_Kyber512_Ascon128a',
    'm10': 'Hybrid_FrodoKEM_640_AES_Ascon128a',
    'm11': 'Hybrid_HQC_128_Ascon128a',
    'm12': 'Hybrid_ClassicMcEliece_348864_Ascon128a',
    'm13': 'Hybrid_X25519_Ascon128a',
}

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
            all_data.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {filename}: {e}")

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
            all_data.append(df)
        except Exception as e:
            print(f"  Warning: Could not load entropy file: {e}")

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
            # Extract IoT config from filename (e.g., testing_process_1Core512MB_...)
            config_match = re.search(r'(\d+Core\d+[MG]B)', filename)
            if config_match:
                df['IoT_Config'] = config_match.group(1)
            all_data.append(df)
        except Exception as e:
            print(f"  Warning: Could not load IoT file {filename}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_gem5_performance_data(gem5_path):
    """Load gem5 performance CSV files (same format as base machine)."""
    all_data = []

    for csv_file in glob.glob(os.path.join(gem5_path, 'testing_process_*.csv')):
        filename = os.path.basename(csv_file)
        # Skip entropy files
        if '_ent_' in filename:
            continue

        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"  Warning: Could not load gem5 file {filename}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_gem5_entropy_data(gem5_path):
    """Load gem5 entropy test CSV files."""
    all_data = []

    for csv_file in glob.glob(os.path.join(gem5_path, '*_ent_*.csv')):
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"  Warning: Could not load gem5 entropy file: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_gem5_hardware_summary(gem5_path):
    """Load gem5 hardware summary from FINAL_CRYPTO_HARDWARE_REPORT.txt."""
    summary_file = os.path.join(gem5_path, 'FINAL_CRYPTO_HARDWARE_REPORT.txt')
    
    if not os.path.exists(summary_file):
        return pd.DataFrame()

    try:
        data = []
        with open(summary_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                # Skip header and separator lines
                if not line or line.startswith('-') or line.startswith('GEM5') or line.startswith('Type') or line.startswith('Generated'):
                    continue
                
                # Parse pipe-separated format: Type | Algorithm Name | CPU Cycles | Sim Seconds
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 4:
                        test_type = parts[0]
                        model_name = parts[1]
                        try:
                            cycles = int(parts[2])
                            sim_seconds = float(parts[3])
                        except ValueError:
                            continue
                        
                        # Only include LATENCY tests (not ENTROPY)
                        if test_type == 'LATENCY':
                            data.append({
                                'Model': model_name,
                                'Cycles': cycles,
                                'Sim_Seconds': sim_seconds,
                            })
        
        if data:
            df = pd.DataFrame(data)
            # Calculate CPI and IPC estimates (assuming ~2GHz clock)
            # These are approximations since we don't have instruction counts in summary
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"  Warning: Could not load gem5 hardware summary: {e}")
        return pd.DataFrame()


def parse_gem5_raw_stats(gem5_path):
    """Parse raw gem5 stats files from final_stats_m* directories."""
    all_stats = []
    
    # Look for final_stats_m* directories
    for stats_dir in glob.glob(os.path.join(gem5_path, 'final_stats_m*')):
        dirname = os.path.basename(stats_dir)
        # Extract model ID (e.g., m1, m2, etc.)
        match = re.search(r'final_stats_(m\d+)', dirname)
        if not match:
            continue
        
        model_id = match.group(1)
        model_name = GEM5_MODEL_MAP.get(model_id, model_id)
        stats_file = os.path.join(stats_dir, 'stats.txt')
        
        if not os.path.exists(stats_file):
            continue
        
        try:
            stats = {'Model_ID': model_id, 'Model': model_name}
            with open(stats_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('-'):
                        continue
                    
                    # Parse key-value pairs
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0]
                        try:
                            value = float(parts[1])
                        except ValueError:
                            continue
                        
                        # Extract key metrics
                        if key == 'simSeconds':
                            stats['Sim_Seconds'] = value
                        elif key == 'simInsts':
                            stats['Instructions'] = int(value)
                        elif key == 'simOps':
                            stats['Ops'] = int(value)
                        elif key == 'system.cpu.numCycles':
                            stats['Cycles'] = int(value)
                        elif key == 'system.cpu.cpi':
                            stats['CPI'] = value
                        elif key == 'system.cpu.ipc':
                            stats['IPC'] = value
                        elif key == 'hostSeconds':
                            stats['Host_Seconds'] = value
                        elif key == 'hostMemory':
                            stats['Host_Memory_Bytes'] = int(value)
                        # Instruction breakdown
                        elif 'issuedInstType_0::IntAlu' in key:
                            stats['IntAlu_Insts'] = int(value)
                        elif 'issuedInstType_0::IntMult' in key:
                            stats['IntMult_Insts'] = int(value)
                        elif 'issuedInstType_0::IntDiv' in key:
                            stats['IntDiv_Insts'] = int(value)
                        elif 'issuedInstType_0::MemRead' in key:
                            stats['MemRead_Insts'] = int(value)
                        elif 'issuedInstType_0::MemWrite' in key:
                            stats['MemWrite_Insts'] = int(value)
            
            all_stats.append(stats)
        except Exception as e:
            print(f"  Warning: Could not parse {stats_file}: {e}")
    
    if all_stats:
        return pd.DataFrame(all_stats)
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


# =============================================================================
# REPORT GENERATION
# =============================================================================

def create_output_dir():
    """Create output directory for figures."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


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
                           base_perf=None, iot_data=None, entropy_data=None,
                           gem5_perf=None, gem5_hw=None, gem5_raw=None, gem5_entropy=None):
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
        f.write("gem5 metrics: Cycles, CPI (Cycles Per Instruction), IPC (Instructions Per Cycle), Sim_Seconds.\n")
        f.write("Comparisons in tables: base machine, IoT simulation, and gem5 compare models within each environment.\n\n")

        section_num = 1

        # =====================================================================
        # SECTION: BASE MACHINE STATISTICS
        # =====================================================================
        f.write("=" * 100 + "\n")
        f.write(f"{section_num}. BASE MACHINE STATISTICS\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"  {section_num}.1 TOTAL EXECUTION TIME (Total_ns)\n")
        f.write("  " + "-" * 50 + "\n")
        if base_stats is not None and not base_stats.empty:
            f.write(base_stats.to_string())
            f.write("\n\n")
        else:
            f.write("  No data available.\n\n")

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

        f.write(f"  {section_num}.1 TOTAL EXECUTION TIME (Total_ns)\n")
        f.write("  " + "-" * 50 + "\n")
        if iot_stats is not None and not iot_stats.empty:
            f.write(iot_stats.to_string())
            f.write("\n\n")
        else:
            f.write("  No IoT data available.\n\n")

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
        # SECTION: GEM5 SIMULATION STATISTICS
        # =====================================================================
        f.write("=" * 100 + "\n")
        f.write(f"{section_num}. GEM5 ARCHITECTURAL SIMULATION STATISTICS\n")
        f.write("=" * 100 + "\n\n")

        # gem5 Hardware Summary
        f.write(f"  {section_num}.1 HARDWARE PERFORMANCE SUMMARY\n")
        f.write("  " + "-" * 50 + "\n")
        if gem5_hw is not None and not gem5_hw.empty:
            f.write("  Model mapping: m1=Kyber512, m2=ClassicMcEliece, m3=HQC_128, m4=FrodoKEM,\n")
            f.write("                 m5=X25519, m6=Hybrid_McEliece, m7=Hybrid_HQC, m8=Hybrid_Frodo,\n")
            f.write("                 m9=Ascon_80pq, m10=BIKE_L1\n\n")
            f.write(gem5_hw.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("  No gem5 hardware summary available.\n\n")

        # gem5 Performance Data
        sub_section = 2
        if gem5_perf is not None and not gem5_perf.empty:
            gem5_stats = compute_descriptive_stats(gem5_perf, 'Model', 'Total_ns')
            f.write(f"  {section_num}.{sub_section} TOTAL EXECUTION TIME (Total_ns)\n")
            f.write("  " + "-" * 50 + "\n")
            f.write(gem5_stats.to_string())
            f.write("\n\n")
            sub_section += 1

            for metric in time_metrics:
                if metric in gem5_perf.columns:
                    metric_stats = compute_metric_stats(gem5_perf, 'Model', metric)
                    if metric_stats is not None and not metric_stats.empty:
                        metric_label = metric.replace('_ns', '').replace('_', ' ')
                        f.write(f"  {section_num}.{sub_section} {metric_label.upper()} ({metric})\n")
                        f.write("  " + "-" * 50 + "\n")
                        f.write(metric_stats.to_string())
                        f.write("\n\n")
                        sub_section += 1

        # gem5 Raw Stats (detailed instruction breakdown)
        if gem5_raw is not None and not gem5_raw.empty:
            f.write(f"  {section_num}.{sub_section} DETAILED INSTRUCTION BREAKDOWN (from raw stats)\n")
            f.write("  " + "-" * 50 + "\n")
            display_cols = ['Model', 'Cycles', 'Instructions', 'CPI', 'IPC', 'Host_Seconds']
            available_cols = [c for c in display_cols if c in gem5_raw.columns]
            if available_cols:
                f.write(gem5_raw[available_cols].to_string(index=False))
                f.write("\n\n")
            
            # Instruction type breakdown
            inst_cols = ['Model', 'IntAlu_Insts', 'IntMult_Insts', 'IntDiv_Insts', 'MemRead_Insts', 'MemWrite_Insts']
            available_inst_cols = [c for c in inst_cols if c in gem5_raw.columns]
            if len(available_inst_cols) > 1:
                f.write("  Instruction Type Breakdown:\n")
                f.write(gem5_raw[available_inst_cols].to_string(index=False))
                f.write("\n\n")
            sub_section += 1

        # gem5 Entropy
        if gem5_entropy is not None and not gem5_entropy.empty:
            gem5_entropy_stats = compute_descriptive_stats(gem5_entropy, 'Model', 'Entropy_Bits_Per_Byte')
            f.write(f"  {section_num}.{sub_section} GEM5 ENTROPY TEST RESULTS\n")
            f.write("  " + "-" * 50 + "\n")
            f.write(gem5_entropy_stats.to_string())
            f.write("\n\n")

        section_num += 1

        # =====================================================================
        # SECTION: ENTROPY STATISTICS
        # =====================================================================
        f.write("=" * 100 + "\n")
        f.write(f"{section_num}. ENTROPY TEST STATISTICS (Base Machine)\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"  {section_num}.1 ENTROPY BITS PER BYTE (Entropy_Bits_Per_Byte)\n")
        f.write("  " + "-" * 50 + "\n")
        f.write("  Ideal value: 8.0 bits/byte (maximum entropy)\n")
        if entropy_stats is not None and not entropy_stats.empty:
            f.write(entropy_stats.to_string())
            f.write("\n\n")
        else:
            f.write("  No entropy data available.\n\n")

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

        f.write(f"  {section_num}.1 KRUSKAL-WALLIS H-TEST (Non-parametric ANOVA)\n")
        f.write("  " + "-" * 50 + "\n")
        if kw_results:
            f.write(f"  H-statistic: {kw_results['statistic']:.4f}\n")
            f.write(f"  p-value: {kw_results['p_value']:.2e}\n")
            f.write(f"  Significant difference: {'Yes' if kw_results['significant'] else 'No'}\n")
            f.write("  Interpretation: Tests whether all models have the same median execution time.\n\n")
        else:
            f.write("  Test not performed.\n\n")

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
        f.write(f"{section_num}. COMPARATIVE ANALYSIS\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"  {section_num}.1 BASE MACHINE VS IoT SIMULATION\n")
        f.write("  " + "-" * 50 + "\n")
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

                    ratio_df = pd.DataFrame({
                        'Base_Mean_us': base_stats.loc[common_models, 'mean'] / 1000,
                        'IoT_Mean_us': iot_stats.loc[common_models, 'mean'] / 1000,
                        'Ratio': ratios
                    }).round(2)
                    ratio_df = ratio_df.sort_values('Ratio')
                    f.write("Per-Model IoT/Base Ratio:\n")
                    f.write(ratio_df.to_string())
                    f.write("\n\n")

        # gem5 vs Base comparison
        f.write(f"  {section_num}.2 BASE MACHINE VS GEM5 SIMULATION\n")
        f.write("  " + "-" * 50 + "\n")
        if gem5_perf is not None and not gem5_perf.empty and base_stats is not None and not base_stats.empty:
            gem5_stats = compute_descriptive_stats(gem5_perf, 'Model', 'Total_ns')
            common_models = base_stats.index.intersection(gem5_stats.index)
            if len(common_models) > 0:
                ratios = (gem5_stats.loc[common_models, 'mean'] / base_stats.loc[common_models, 'mean']).dropna()
                if len(ratios) > 0:
                    f.write("gem5 vs Base Machine Performance Ratio (common models):\n")
                    f.write(f"  Median ratio: {ratios.median():.2f}x\n")
                    f.write(f"  Min ratio: {ratios.min():.2f}x\n")
                    f.write(f"  Max ratio: {ratios.max():.2f}x\n\n")

                    ratio_df = pd.DataFrame({
                        'Base_Mean_us': base_stats.loc[common_models, 'mean'] / 1000,
                        'gem5_Mean_us': gem5_stats.loc[common_models, 'mean'] / 1000,
                        'Ratio': ratios
                    }).round(2)
                    ratio_df = ratio_df.sort_values('Ratio')
                    f.write("Per-Model gem5/Base Ratio:\n")
                    f.write(ratio_df.to_string())
                    f.write("\n\n")
        else:
            f.write("  No common models between base machine and gem5 for comparison.\n\n")

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
    gem5_path = os.path.join(base_path, 'gem5_sim_results')

    # Create output directory
    create_output_dir()

    # Load data
    print("Loading data...")
    
    # Base machine data
    base_perf = load_performance_data(base_path, 'base_machine')
    entropy_data = load_entropy_data(base_path, 'base_machine')
    iot_data = load_iot_data(base_path)

    # gem5 data (from saved_output/gem5_sim_results)
    gem5_perf = load_gem5_performance_data(gem5_path)
    gem5_entropy = load_gem5_entropy_data(gem5_path)
    gem5_hw = load_gem5_hardware_summary(gem5_path)
    gem5_raw = parse_gem5_raw_stats(gem5_path)

    print(f"  Base machine: {len(base_perf)} records")
    print(f"  Base entropy tests: {len(entropy_data)} records")
    print(f"  IoT simulation: {len(iot_data)} records")
    print(f"  gem5 performance: {len(gem5_perf)} records")
    print(f"  gem5 entropy tests: {len(gem5_entropy)} records")
    print(f"  gem5 hardware summary: {len(gem5_hw)} models")
    print(f"  gem5 raw stats: {len(gem5_raw)} models")

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

    if not gem5_perf.empty:
        print("  gem5 performance statistics computed")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Base machine visualizations
    if not base_perf.empty:
        plot_execution_time_comparison(base_perf, OUTPUT_DIR, ' (Base Machine)', 'exec_time_base.pdf')
        plot_operation_breakdown(base_perf, OUTPUT_DIR, 'operation_breakdown.pdf')
        plot_boxplot_comparison(base_perf, OUTPUT_DIR, 'boxplot_base.pdf')
        plot_violin_comparison(base_perf, OUTPUT_DIR, 'violin_base.pdf')
        plot_memory_comparison(base_perf, OUTPUT_DIR, 'memory_comparison.pdf')
        plot_performance_heatmap(base_perf, OUTPUT_DIR, 'performance_heatmap.pdf')
        plot_standalone_vs_hybrid(base_perf, OUTPUT_DIR, 'standalone_vs_hybrid.pdf')
        plot_cv_comparison(base_perf, OUTPUT_DIR, 'cv_comparison.pdf')

    # Entropy visualizations
    if not entropy_data.empty:
        plot_entropy_analysis(entropy_data, OUTPUT_DIR, 'entropy_analysis.pdf')

    # IoT visualizations
    if not iot_data.empty:
        plot_iot_comparison(iot_data, OUTPUT_DIR, 'iot_comparison.pdf')
        plot_iot_scaling(iot_data, OUTPUT_DIR, 'iot_scaling.pdf')
        if not base_perf.empty:
            plot_iot_vs_base_ratio(base_perf, iot_data, OUTPUT_DIR, 'iot_vs_base_ratio.pdf')

    # gem5 visualizations
    if not gem5_perf.empty or not gem5_hw.empty or not gem5_raw.empty:
        plot_gem5_cycles_comparison(gem5_perf, gem5_hw, OUTPUT_DIR, 'gem5_cycles.pdf')
        plot_gem5_cpi_ipc(gem5_raw, OUTPUT_DIR, 'gem5_cpi_ipc.pdf')
        plot_gem5_sim_seconds(gem5_hw, OUTPUT_DIR, 'gem5_sim_seconds.pdf')
        if not base_perf.empty:
            plot_gem5_vs_base(base_perf, gem5_perf, OUTPUT_DIR, 'gem5_vs_base.pdf')
        plot_gem5_operation_breakdown(gem5_perf, OUTPUT_DIR, 'gem5_operation_breakdown.pdf')

    # Cross-environment comparison
    if not base_perf.empty or not iot_data.empty or not gem5_perf.empty:
        plot_all_environments_comparison(base_perf, iot_data, gem5_perf, OUTPUT_DIR, 'all_environments.pdf')

    # Save statistics report
    print("\nSaving statistical report...")
    save_statistics_report(
        base_stats, iot_stats, entropy_stats, kw_results, pairwise_results,
        base_perf=base_perf, iot_data=iot_data, entropy_data=entropy_data,
        gem5_perf=gem5_perf, gem5_hw=gem5_hw, gem5_raw=gem5_raw, gem5_entropy=gem5_entropy
    )

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

    if not gem5_hw.empty:
        print("\n" + "=" * 60)
        print("SUMMARY: GEM5 HARDWARE METRICS")
        print("=" * 60)
        print(gem5_hw[['Model', 'Cycles', 'Sim_Seconds']].to_string(index=False))
    
    if not gem5_raw.empty:
        print("\n" + "=" * 60)
        print("SUMMARY: GEM5 CPI/IPC METRICS")
        print("=" * 60)
        print(gem5_raw[['Model', 'CPI', 'IPC']].to_string(index=False))

    print("\n" + "=" * 60)
    print(f"Analysis complete. Output saved to: {OUTPUT_DIR}/")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
