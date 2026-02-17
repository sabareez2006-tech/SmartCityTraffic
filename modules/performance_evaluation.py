"""
Module 9: Performance Evaluation Module

Evaluates traffic evolution trends and robotaxi system behavior
under dynamic conditions compared to static baselines. Generates
comprehensive visualizations and comparison reports.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.helpers import ensure_dir


class PerformanceEvaluationModule:
    """
    Comprehensive evaluation and visualization for the federated
    learning traffic modeling and robotaxi simulation framework.
    """

    def __init__(self, results_dir: str = None):
        """
        Args:
            results_dir: Directory to save plots and reports.
        """
        self.results_dir = results_dir or config.RESULTS_DIR
        ensure_dir(self.results_dir)

        # Set plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    # ═══════════════════════════════════════════════════
    # Federated Learning Evaluation Plots
    # ═══════════════════════════════════════════════════

    def plot_training_loss(self, round_losses: List[float],
                           zone_losses: Dict[int, List[float]] = None):
        """
        Plot federated learning training loss over rounds.

        Args:
            round_losses: Average loss per FL round.
            zone_losses:  Optional per-zone losses over rounds.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Global loss curve
        ax = axes[0]
        ax.plot(range(1, len(round_losses) + 1), round_losses,
                'b-o', linewidth=2, markersize=6, label='Global Avg Loss')
        ax.set_xlabel('Federated Learning Round', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.set_title('Global Model Training Loss', fontsize=14,
                     fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Per-zone loss curves
        ax = axes[1]
        if zone_losses:
            for zid in sorted(zone_losses.keys()):
                losses = zone_losses[zid]
                ax.plot(range(1, len(losses) + 1), losses,
                        alpha=0.5, linewidth=1, label=f'Zone {zid}')
            ax.set_xlabel('FL Round', fontsize=12)
            ax.set_ylabel('MSE Loss', fontsize=12)
            ax.set_title('Per-Zone Training Loss', fontsize=14,
                         fontweight='bold')
            if len(zone_losses) <= 10:
                ax.legend(fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.results_dir, 'training_loss.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {path}")

    def plot_prediction_accuracy(self, actuals: Dict[int, np.ndarray],
                                  predictions: Dict[int, np.ndarray]):
        """
        Plot actual vs predicted traffic for sample zones.

        Args:
            actuals:     zone_id → actual values array (N, 3)
            predictions: zone_id → predicted values array (N, 3)
        """
        feature_names = ['Ride Requests', 'Congestion Level', 'Traffic Flow']
        sample_zones = sorted(actuals.keys())[:4]  # Show up to 4 zones

        fig, axes = plt.subplots(len(sample_zones), 3,
                                 figsize=(20, 4 * len(sample_zones)))
        if len(sample_zones) == 1:
            axes = axes.reshape(1, -1)

        for i, zid in enumerate(sample_zones):
            actual = actuals[zid]
            pred = predictions[zid]
            num_points = min(len(actual), len(pred))
            t = range(num_points)

            for j, fname in enumerate(feature_names):
                ax = axes[i, j]
                ax.plot(t, actual[:num_points, j], 'b-', alpha=0.7,
                        linewidth=1, label='Actual')
                ax.plot(t, pred[:num_points, j], 'r--', alpha=0.7,
                        linewidth=1, label='Predicted')
                ax.set_title(f'Zone {zid} – {fname}', fontsize=11,
                             fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

        plt.suptitle('Traffic Prediction: Actual vs Predicted',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(self.results_dir, 'prediction_accuracy.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {path}")

    def plot_compression_analysis(self, compression_ratios: List[float],
                                   communication_savings: List[float],
                                   accuracy_with_compression: List[float],
                                   accuracy_without: List[float]):
        """Plot gradient compression effects on accuracy and savings."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Communication savings
        ax = axes[0]
        ax.bar(range(len(communication_savings)), communication_savings,
               color='steelblue', alpha=0.8)
        ax.set_xlabel('FL Round', fontsize=12)
        ax.set_ylabel('Communication Savings (%)', fontsize=12)
        ax.set_title('Communication Savings via Gradient Compression',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Accuracy comparison
        ax = axes[1]
        rounds = range(1, len(accuracy_with_compression) + 1)
        ax.plot(rounds, accuracy_without, 'g-s', linewidth=2,
                markersize=5, label='Without Compression')
        ax.plot(rounds, accuracy_with_compression, 'r-^', linewidth=2,
                markersize=5, label='With Compression')
        ax.set_xlabel('FL Round', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('Prediction RMSE: Compressed vs Uncompressed',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.results_dir, 'compression_analysis.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {path}")

    def plot_correlation_selection(self, selection_history: List[Dict]):
        """Plot correlation-based client selection statistics."""
        if not selection_history:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        rounds = range(1, len(selection_history) + 1)
        selected = [h["selected_clients"] for h in selection_history]
        total = [h["total_clients"] for h in selection_history]
        rates = [s / t for s, t in zip(selected, total)]

        ax = axes[0]
        ax.bar(rounds, total, alpha=0.4, color='blue', label='Total Clients')
        ax.bar(rounds, selected, alpha=0.8, color='green',
               label='Selected Clients')
        ax.set_xlabel('FL Round', fontsize=12)
        ax.set_ylabel('Number of Clients', fontsize=12)
        ax.set_title('Client Selection per Round', fontsize=13,
                     fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[1]
        ax.plot(rounds, rates, 'o-', color='purple', linewidth=2)
        ax.axhline(y=np.mean(rates), color='red', linestyle='--',
                   label=f'Average: {np.mean(rates):.1%}')
        ax.set_xlabel('FL Round', fontsize=12)
        ax.set_ylabel('Selection Rate', fontsize=12)
        ax.set_title('Client Selection Rate', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.results_dir, 'correlation_selection.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {path}")

    # ═══════════════════════════════════════════════════
    # Robotaxi Simulation Evaluation Plots
    # ═══════════════════════════════════════════════════

    def plot_simulation_comparison(self, dynamic_metrics: Dict,
                                    static_metrics: Dict):
        """
        Compare robotaxi performance under dynamic vs static traffic.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        labels = ['Dynamic Traffic\n(FL-Based)', 'Static Traffic\n(Baseline)']
        colors = ['#2196F3', '#FF9800']

        # Service Rate
        ax = axes[0, 0]
        vals = [dynamic_metrics['service_rate'] * 100,
                static_metrics['service_rate'] * 100]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
        ax.set_ylabel('Service Rate (%)', fontsize=12)
        ax.set_title('Service Rate Comparison', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')

        # Average Wait Time
        ax = axes[0, 1]
        vals = [dynamic_metrics['avg_wait_time'],
                static_metrics['avg_wait_time']]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f'{val:.1f} min', ha='center', fontsize=12,
                    fontweight='bold')
        ax.set_ylabel('Wait Time (minutes)', fontsize=12)
        ax.set_title('Average Wait Time Comparison', fontsize=13,
                     fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Fleet Utilization
        ax = axes[1, 0]
        vals = [dynamic_metrics['avg_fleet_utilization'] * 100,
                static_metrics['avg_fleet_utilization'] * 100]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
        ax.set_ylabel('Utilization (%)', fontsize=12)
        ax.set_title('Fleet Utilization Comparison', fontsize=13,
                     fontweight='bold')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')

        # Completed Rides
        ax = axes[1, 1]
        vals = [dynamic_metrics['total_completed'],
                static_metrics['total_completed']]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    f'{int(val):,}', ha='center', fontsize=12,
                    fontweight='bold')
        ax.set_ylabel('Completed Rides', fontsize=12)
        ax.set_title('Total Completed Rides', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Robotaxi Performance: Dynamic vs Static Traffic',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(self.results_dir, 'simulation_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {path}")

    def plot_traffic_evolution(self, zone_data: Dict[int, np.ndarray],
                                predictions: Dict[int, List[np.ndarray]],
                                sample_zones: List[int] = None):
        """
        Plot how traffic evolves over time for sample zones,
        showing actual data and federated predictions.
        """
        if sample_zones is None:
            sample_zones = sorted(zone_data.keys())[:4]

        fig, axes = plt.subplots(len(sample_zones), 1,
                                 figsize=(16, 4 * len(sample_zones)))
        if len(sample_zones) == 1:
            axes = [axes]

        for i, zid in enumerate(sample_zones):
            ax = axes[i]
            data = zone_data[zid]

            # Convert time steps to hours
            hours = np.arange(len(data)) * config.TIME_STEP_MINUTES / 60

            ax.plot(hours, data[:, 0], 'b-', alpha=0.6, linewidth=1,
                    label='Ride Requests')
            ax2 = ax.twinx()
            ax2.plot(hours, data[:, 1], 'r-', alpha=0.6, linewidth=1,
                     label='Congestion')
            ax2.set_ylabel('Congestion Level', color='red', fontsize=11)

            ax.set_xlabel('Time (hours)', fontsize=11)
            ax.set_ylabel('Ride Requests', color='blue', fontsize=11)
            ax.set_title(f'Zone {zid} – Traffic Evolution Over Time',
                         fontsize=12, fontweight='bold')

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      loc='upper right', fontsize=9)

        plt.suptitle('Dynamic Traffic Evolution Across City Zones',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(self.results_dir, 'traffic_evolution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {path}")

    def plot_city_heatmap(self, zone_values: np.ndarray,
                           grid_size: Tuple[int, int],
                           title: str = 'Zone Demand Heatmap',
                           filename: str = 'city_heatmap.png'):
        """Plot a heatmap of values across the city grid."""
        grid = zone_values.reshape(grid_size)

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        im = ax.imshow(grid, cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Annotate cells
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                ax.text(j, i, f'{grid[i, j]:.1f}', ha='center',
                        va='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(grid_size[1]))
        ax.set_yticks(range(grid_size[0]))

        plt.tight_layout()
        path = os.path.join(self.results_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {path}")

    def plot_fleet_utilization_over_time(self, step_metrics: List[Dict]):
        """Plot fleet utilization and wait time over simulation time."""
        if not step_metrics:
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        times = [m['time'] / 60 for m in step_metrics]  # Convert to hours

        # Fleet utilization
        ax = axes[0]
        util = [m['fleet_utilization'] * 100 for m in step_metrics]
        ax.plot(times, util, 'b-', linewidth=1.5, alpha=0.8)
        ax.fill_between(times, util, alpha=0.2, color='blue')
        ax.set_ylabel('Fleet Utilization (%)', fontsize=12)
        ax.set_title('Fleet Utilization Over Time', fontsize=13,
                     fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Wait time
        ax = axes[1]
        waits = [m['avg_wait_time'] for m in step_metrics]
        ax.plot(times, waits, 'r-', linewidth=1.5, alpha=0.8)
        ax.fill_between(times, waits, alpha=0.2, color='red')
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Avg Wait Time (min)', fontsize=12)
        ax.set_title('Average Wait Time Over Time', fontsize=13,
                     fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.results_dir,
                            'fleet_utilization_timeline.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {path}")

    # ═══════════════════════════════════════════════════
    # Report Generation
    # ═══════════════════════════════════════════════════

    def generate_report(self, fl_metrics: Dict, dynamic_sim: Dict,
                        static_sim: Dict, compression_stats: Dict,
                        correlation_stats: Dict) -> str:
        """
        Generate a comprehensive text report.

        Returns:
            Report string.
        """
        report = []
        report.append("=" * 70)
        report.append("PERFORMANCE EVALUATION REPORT")
        report.append("Federated Learning-Based Dynamic Traffic Modeling")
        report.append("for Robotaxi Simulation")
        report.append("=" * 70)

        # FL Training Results
        report.append("\n1. FEDERATED LEARNING TRAINING")
        report.append("-" * 40)
        report.append(f"   Total FL Rounds:    {fl_metrics.get('total_rounds', 'N/A')}")
        report.append(f"   Final Global Loss:  {fl_metrics.get('final_loss', 'N/A'):.6f}")
        report.append(f"   Best Global Loss:   {fl_metrics.get('best_loss', 'N/A'):.6f}")
        report.append(f"   Prediction MAE:     {fl_metrics.get('mae', 'N/A'):.4f}")
        report.append(f"   Prediction RMSE:    {fl_metrics.get('rmse', 'N/A'):.4f}")
        report.append(f"   Prediction MAPE:    {fl_metrics.get('mape', 'N/A'):.2f}%")

        # Compression Results
        report.append("\n2. GRADIENT COMPRESSION")
        report.append("-" * 40)
        report.append(f"   Target Ratio:       {compression_stats.get('target_ratio', 'N/A')}")
        report.append(f"   Actual Ratio:       {compression_stats.get('actual_ratio', 'N/A')}")
        report.append(f"   Comm. Savings:      {compression_stats.get('savings', 'N/A')}")

        # Correlation Selection
        report.append("\n3. CORRELATION-BASED AGGREGATION")
        report.append("-" * 40)
        report.append(f"   Avg Selection Rate: {correlation_stats.get('selection_rate', 'N/A')}")
        report.append(f"   Redundant Removed:  {correlation_stats.get('total_removed', 'N/A')}")

        # Simulation Comparison
        report.append("\n4. ROBOTAXI SIMULATION COMPARISON")
        report.append("-" * 40)
        report.append(f"   {'Metric':<25s} {'Dynamic':>12s} {'Static':>12s}")
        report.append(f"   {'─' * 49}")

        metrics_to_compare = [
            ('Service Rate', 'service_rate', lambda x: f"{x:.1%}"),
            ('Avg Wait Time (min)', 'avg_wait_time', lambda x: f"{x:.1f}"),
            ('Fleet Utilization', 'avg_fleet_utilization', lambda x: f"{x:.1%}"),
            ('Completed Rides', 'total_completed', lambda x: f"{int(x):,}"),
            ('Expired Requests', 'total_expired', lambda x: f"{int(x):,}"),
        ]

        for label, key, fmt in metrics_to_compare:
            dyn_val = dynamic_sim.get(key, 0)
            sta_val = static_sim.get(key, 0)
            report.append(f"   {label:<25s} {fmt(dyn_val):>12s} {fmt(sta_val):>12s}")

        report.append("\n" + "=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)

        report_text = "\n".join(report)

        # Save report
        path = os.path.join(self.results_dir, 'evaluation_report.txt')
        with open(path, 'w') as f:
            f.write(report_text)
        print(f"  ✓ Saved: {path}")

        return report_text
