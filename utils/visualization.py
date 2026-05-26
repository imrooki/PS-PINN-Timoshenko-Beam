
import os
from typing import Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .analysis_config import PARAM_LABELS

def create_summary_plots(analysis_data: Dict[str, Any], output_dir: Path):
    print(f"[DATA] Generating summary comparison charts...")

    output_dir.mkdir(parents=True, exist_ok=True)

    for param_name, data in analysis_data.items():
        if 'successful_param_values' not in data or not data['successful_param_values']:
            print(f"[WARN] Skipping {param_name} (no successful data)")
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        param_vals = data['successful_param_values']
        linear_deflections = data['max_linear_deflections']
        nonlinear_deflections = data['max_nonlinear_deflections']

        if param_name == 'q':
            display_vals = [abs(v) for v in param_vals]
        else:
            display_vals = param_vals

        ax1.plot(display_vals, linear_deflections, 'b-o', linewidth=2.5, markersize=8,
                markerfacecolor='white', markeredgecolor='blue', markeredgewidth=2,
                label='Linear Theory')
        ax1.plot(display_vals, nonlinear_deflections, 'r-s', linewidth=2.5, markersize=8,
                markerfacecolor='white', markeredgecolor='red', markeredgewidth=2,
                label='Nonlinear Theory')

        xlabel, title_param = PARAM_LABELS.get(param_name, (param_name, param_name))
        ax1.set_xlabel(xlabel, fontsize=13)
        ax1.set_ylabel('Maximum Deflection |w|_max', fontsize=13)
        ax1.set_title(f'Linear vs Nonlinear: Effect of {title_param}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', fontsize=11)

        nonlinear_effect = [(nl - l) / l * 100 if l > 1e-12 else 0
                          for l, nl in zip(linear_deflections, nonlinear_deflections)]

        ax2.plot(display_vals, nonlinear_effect, 'g-^', linewidth=2.5, markersize=8,
                markerfacecolor='white', markeredgecolor='green', markeredgewidth=2,
                label='Nonlinear Effect')
        ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)

        ax2.set_xlabel(xlabel, fontsize=13)
        ax2.set_ylabel('Nonlinear Effect (%)', fontsize=13)
        ax2.set_title(f'Nonlinear Effect: (w_nl - w_l)/w_l × 100%', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=11)

        plt.tight_layout()

        plot_file = output_dir / f"{param_name}_sensitivity_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"   [Done] {param_name} comparison plot saved: {plot_file.name}")

    _create_comprehensive_plot(analysis_data, output_dir)

    create_summary_table(analysis_data, output_dir)

    create_max_deflection_plots(analysis_data, output_dir)

def _create_comprehensive_plot(analysis_data: Dict[str, Any], output_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    param_labels = {k: v[1] if isinstance(v, tuple) else v for k, v in PARAM_LABELS.items()}

    plot_idx = 0
    for param_name, data in analysis_data.items():
        if 'successful_param_values' not in data or not data['successful_param_values']:
            continue

        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]
        param_vals = data['successful_param_values']
        nonlinear_deflections = data['max_nonlinear_deflections']

        if param_name == 'q':
            display_vals = [abs(v) for v in param_vals]
        else:
            display_vals = param_vals

        ax.plot(display_vals, nonlinear_deflections, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel(param_labels.get(param_name, param_name), fontsize=12)
        ax.set_ylabel('Max Deflection', fontsize=12)
        ax.set_title(f'Sensitivity to {param_labels.get(param_name, param_name)}', fontsize=13)
        ax.grid(True, alpha=0.3)

        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    comprehensive_plot = output_dir / "comprehensive_sensitivity_analysis.png"
    plt.savefig(comprehensive_plot, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"   [Done] Comprehensive comparison plot saved: {comprehensive_plot.name}")

def create_summary_table(analysis_data: Dict[str, Any], output_dir: Path,
                        filename: str = "sensitivity_summary.csv"):
    summary_data = []

    for param_name, data in analysis_data.items():
        if 'successful_param_values' not in data:
            continue

        param_vals = data['successful_param_values']
        linear_vals = data['max_linear_deflections']
        nonlinear_vals = data['max_nonlinear_deflections']
        elapsed_times = data.get('elapsed_times', [0] * len(param_vals))

        for i, pv in enumerate(param_vals):
            summary_data.append({
                'Parameter': param_name,
                'Parameter_Value': pv,
                'Max_Linear_Deflection': linear_vals[i],
                'Max_Nonlinear_Deflection': nonlinear_vals[i],
                'Nonlinear_Effect_Percent': (nonlinear_vals[i] - linear_vals[i]) / linear_vals[i] * 100
                if linear_vals[i] > 1e-12 else 0,
                'Elapsed_Time_Seconds': elapsed_times[i] if i < len(elapsed_times) else 0
            })

    if summary_data:
        df = pd.DataFrame(summary_data)
        summary_file = output_dir / filename
        df.to_csv(summary_file, index=False, float_format='%.8f')
        print(f"   [Done] Summary data table saved: {summary_file.name}")

def create_max_deflection_plots(analysis_data: Dict[str, Any], output_dir: Path):
    print(f"   [DATA] Generating single-parameter max deflection summary plots...")

    for param_name, data in analysis_data.items():
        if 'successful_param_values' not in data or not data['successful_param_values']:
            print(f"      [WARN] Skipping {param_name} (no successful data)")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        param_vals = data['successful_param_values']
        nonlinear_deflections = data['max_nonlinear_deflections']

        if param_name == 'q':
            display_vals = [abs(v) for v in param_vals]
            xlabel = 'Load Magnitude |q|'
        else:
            display_vals = param_vals
            xlabel = PARAM_LABELS.get(param_name, (param_name, param_name))
            if isinstance(xlabel, tuple):
                xlabel = xlabel[0]

        ax.plot(display_vals, nonlinear_deflections, 'o-', linewidth=2.5, markersize=8,
               markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2,
               color='red', label='Maximum Deflection')

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel('Maximum Deflection |w|_max', fontsize=13)

        title_param = PARAM_LABELS.get(param_name, (param_name, param_name))
        if isinstance(title_param, tuple):
            title_param = title_param[0]
        ax.set_title(f'Parameter Sensitivity: {title_param}', fontsize=14, fontweight='bold')

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11)

        for i, (x, y) in enumerate(zip(display_vals, nonlinear_deflections)):
            if i % max(1, len(display_vals)//6) == 0:
                ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=9, alpha=0.7)

        plt.tight_layout()

        plot_file = output_dir / f"max_deflection_vs_{param_name}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"      [Done] {param_name} max deflection plot saved: {plot_file.name}")

    print(f"   [Done] Single-parameter max deflection summary plots completed")

def create_comparison_plot(param_name: str, param_values: list,
                          linear_values: list, nonlinear_values: list,
                          output_path: Path, title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(10, 6))

    if param_name == 'q':
        display_vals = [abs(v) for v in param_values]
        xlabel = 'Load Magnitude |q|'
    else:
        display_vals = param_values
        xlabel = PARAM_LABELS.get(param_name, (param_name, param_name))
        if isinstance(xlabel, tuple):
            xlabel = xlabel[0]

    ax.plot(display_vals, linear_values, 'b-o', linewidth=2.5, markersize=8,
            markerfacecolor='white', markeredgecolor='blue', markeredgewidth=2,
            label='Linear Theory')
    ax.plot(display_vals, nonlinear_values, 'r-s', linewidth=2.5, markersize=8,
            markerfacecolor='white', markeredgecolor='red', markeredgewidth=2,
            label='Nonlinear Theory')

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel('Maximum Deflection |w|_max', fontsize=13)

    title_param = PARAM_LABELS.get(param_name, (param_name, param_name))
    if isinstance(title_param, tuple):
        title_param = title_param[1]

    title = f'Effect of {title_param}'
    if title_suffix:
        title += f' - {title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"   [Done] Comparison plot saved: {output_path.name}")
