"""
PTK-HUIM Performance Visualization Suite
Creates publication-quality graphs and analysis visualizations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List
from scipy import stats
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Color schemes for different metrics
COLORS = {
    'time': '#2E86AB',
    'memory': '#A23B72',
    'pruning': '#F18F01',
    'throughput': '#C73E1D',
    'efficiency': '#6A994E'
}


class ResearchVisualizer:
    def __init__(self, results_file: str, output_dir: str = 'exp_results/figures'):
        """Initialize visualizer with results file"""
        self.results_file = results_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.load_data()

    def load_data(self):
        """Load and preprocess results data"""
        if self.results_file.endswith('.json'):
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
        else:
            self.df = pd.read_csv(self.results_file)

        # Filter successful runs
        self.df_success = self.df[self.df['status'] == 'SUCCESS'].copy()

        # Calculate additional metrics if not present
        if 'speedup' not in self.df_success.columns:
            # Calculate speedup (assuming first configuration as baseline)
            baseline_times = {}
            for dataset in self.df_success['dataset'].unique():
                dataset_df = self.df_success[self.df_success['dataset'] == dataset]
                if not dataset_df.empty:
                    baseline_times[dataset] = dataset_df['execution_time_ms'].min()

            self.df_success['speedup'] = self.df_success.apply(
                lambda row: baseline_times.get(row['dataset'], row['execution_time_ms']) / row['execution_time_ms'],
                axis=1
            )

    def create_performance_overview(self):
        """Create comprehensive performance overview figure"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Execution Time by Dataset and K
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_time_by_k(ax1)

        # 2. Memory Usage Analysis
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_memory_usage(ax2)

        # 3. Pruning Effectiveness
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_pruning_effectiveness(ax3)

        # 4. Scalability Analysis
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_scalability(ax4)

        # 5. Parameter Sensitivity (K)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_k_sensitivity(ax5)

        # 6. Parameter Sensitivity (MinProb)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_minprob_sensitivity(ax6)

        # 7. Throughput Comparison
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_throughput(ax7)

        # 8. CAS Efficiency
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_cas_efficiency(ax8)

        # 9. Overall Statistics
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_statistics_table(ax9)

        plt.suptitle('PTK-HUIM-U± Performance Analysis', fontsize=16, y=1.02)

        # Save figure
        output_path = self.output_dir / 'performance_overview.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        print(f"Saved performance overview to {output_path}")

    def _plot_time_by_k(self, ax):
        """Plot execution time by K values"""
        data_pivot = self.df_success.pivot_table(
            values='execution_time_ms',
            index='k',
            columns='dataset',
            aggfunc='mean'
        )

        data_pivot.plot(kind='bar', ax=ax, color=sns.color_palette("husl", len(data_pivot.columns)))
        ax.set_xlabel('K (Number of Patterns)')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Execution Time by K Value')
        ax.legend(title='Dataset', loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    def _plot_memory_usage(self, ax):
        """Plot memory usage patterns"""
        datasets = self.df_success['dataset'].unique()
        memory_data = []

        for dataset in datasets:
            dataset_df = self.df_success[self.df_success['dataset'] == dataset]
            memory_data.append(dataset_df['peak_memory_mb'].values)

        bp = ax.boxplot(memory_data, labels=datasets, patch_artist=True)

        for patch, color in zip(bp['boxes'], sns.color_palette("muted")):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel('Dataset')
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('Memory Usage Distribution')
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    def _plot_pruning_effectiveness(self, ax):
        """Plot pruning effectiveness"""
        pruning_by_dataset = self.df_success.groupby('dataset')['pruning_percentage'].mean()

        bars = ax.bar(range(len(pruning_by_dataset)), pruning_by_dataset.values,
                      color=COLORS['pruning'], alpha=0.7)

        # Add value labels on bars
        for bar, value in zip(bars, pruning_by_dataset.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(range(len(pruning_by_dataset)))
        ax.set_xticklabels(pruning_by_dataset.index, rotation=45, ha='right')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Pruning Percentage (%)')
        ax.set_title('Average Pruning Effectiveness')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_scalability(self, ax):
        """Plot scalability analysis"""
        # Group by database size if available
        if 'database_size' in self.df_success.columns:
            scalability_data = self.df_success.groupby('database_size').agg({
                'execution_time_ms': 'mean',
                'peak_memory_mb': 'mean'
            }).reset_index()

            ax2 = ax.twinx()

            line1 = ax.plot(scalability_data['database_size'],
                            scalability_data['execution_time_ms'],
                            'o-', color=COLORS['time'], label='Execution Time', linewidth=2)
            line2 = ax2.plot(scalability_data['database_size'],
                             scalability_data['peak_memory_mb'],
                             's-', color=COLORS['memory'], label='Memory Usage', linewidth=2)

            ax.set_xlabel('Database Size (transactions)')
            ax.set_ylabel('Execution Time (ms)', color=COLORS['time'])
            ax2.set_ylabel('Memory Usage (MB)', color=COLORS['memory'])
            ax.tick_params(axis='y', labelcolor=COLORS['time'])
            ax2.tick_params(axis='y', labelcolor=COLORS['memory'])

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')

            ax.set_title('Scalability Analysis')
            ax.grid(True, alpha=0.3)
        else:
            # Alternative plot if database_size not available
            ax.text(0.5, 0.5, 'Database size data not available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Scalability Analysis')

    def _plot_k_sensitivity(self, ax):
        """Plot sensitivity to K parameter"""
        k_sensitivity = self.df_success.groupby('k').agg({
            'execution_time_ms': ['mean', 'std'],
            'pruning_percentage': ['mean', 'std']
        }).reset_index()

        k_values = k_sensitivity['k'].values
        time_means = k_sensitivity['execution_time_ms']['mean'].values
        time_stds = k_sensitivity['execution_time_ms']['std'].values

        ax.errorbar(k_values, time_means, yerr=time_stds,
                    marker='o', capsize=5, capthick=2, color=COLORS['time'],
                    linewidth=2, markersize=8)

        ax.set_xlabel('K (Top-K Value)')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Sensitivity to K Parameter')
        ax.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(k_values, time_means, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(k_values.min(), k_values.max(), 100)
        ax.plot(x_smooth, p(x_smooth), '--', alpha=0.5, color='red', label='Trend')
        ax.legend()

    def _plot_minprob_sensitivity(self, ax):
        """Plot sensitivity to MinProb parameter"""
        prob_sensitivity = self.df_success.groupby('min_prob').agg({
            'execution_time_ms': ['mean', 'std'],
            'topk_found': ['mean', 'std']
        }).reset_index()

        prob_values = prob_sensitivity['min_prob'].values
        time_means = prob_sensitivity['execution_time_ms']['mean'].values

        ax.semilogx(prob_values, time_means, 'o-', color=COLORS['efficiency'],
                    linewidth=2, markersize=8)

        ax.set_xlabel('Minimum Probability Threshold')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Sensitivity to MinProb Parameter')
        ax.grid(True, alpha=0.3, which='both')

        # Add annotations for specific points
        for i, (x, y) in enumerate(zip(prob_values, time_means)):
            if i % 2 == 0:  # Annotate every other point to avoid clutter
                ax.annotate(f'{y:.0f}', (x, y), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=8)

    def _plot_throughput(self, ax):
        """Plot throughput comparison"""
        if 'transactions_per_second' in self.df_success.columns:
            throughput_data = self.df_success.groupby(['dataset', 'k']).agg({
                'transactions_per_second': 'mean'
            }).reset_index()

            pivot_data = throughput_data.pivot(index='k', columns='dataset',
                                               values='transactions_per_second')

            pivot_data.plot(kind='line', ax=ax, marker='o', linewidth=2)
            ax.set_xlabel('K Value')
            ax.set_ylabel('Transactions per Second')
            ax.set_title('Throughput Performance')
            ax.legend(title='Dataset')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Throughput data not available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Throughput Performance')

    def _plot_cas_efficiency(self, ax):
        """Plot CAS efficiency metrics"""
        if 'cas_efficiency' in self.df_success.columns:
            cas_data = self.df_success['cas_efficiency'].dropna()

            ax.hist(cas_data, bins=20, color=COLORS['efficiency'],
                    alpha=0.7, edgecolor='black')
            ax.axvline(cas_data.mean(), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {cas_data.mean():.1f}%')
            ax.axvline(cas_data.median(), color='blue', linestyle='--',
                       linewidth=2, label=f'Median: {cas_data.median():.1f}%')

            ax.set_xlabel('CAS Efficiency (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('CAS Operation Efficiency Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'CAS efficiency data not available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('CAS Operation Efficiency')

    def _plot_statistics_table(self, ax):
        """Display summary statistics table"""
        ax.axis('tight')
        ax.axis('off')

        # Calculate summary statistics
        stats = []
        stats.append(['Metric', 'Mean', 'Std', 'Min', 'Max'])

        metrics = [
            ('Execution Time (ms)', 'execution_time_ms'),
            ('Memory (MB)', 'peak_memory_mb'),
            ('Pruning (%)', 'pruning_percentage'),
            ('Candidates', 'candidates_generated')
        ]

        for label, col in metrics:
            if col in self.df_success.columns:
                data = self.df_success[col].dropna()
                stats.append([
                    label,
                    f'{data.mean():.1f}',
                    f'{data.std():.1f}',
                    f'{data.min():.1f}',
                    f'{data.max():.1f}'
                ])

        table = ax.table(cellText=stats, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Style the header row
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Summary Statistics', pad=20)

    def create_detailed_comparison(self):
        """Create detailed comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Time complexity analysis
        ax1 = axes[0, 0]
        self._plot_time_complexity(ax1)

        # 2. Memory complexity analysis
        ax2 = axes[0, 1]
        self._plot_memory_complexity(ax2)

        # 3. Pruning breakdown
        ax3 = axes[1, 0]
        self._plot_pruning_breakdown(ax3)

        # 4. Parallel efficiency
        ax4 = axes[1, 1]
        self._plot_parallel_efficiency(ax4)

        plt.suptitle('Detailed Performance Analysis', fontsize=14)
        plt.tight_layout()

        output_path = self.output_dir / 'detailed_comparison.png'
        plt.savefig(output_path)
        plt.close()
        print(f"Saved detailed comparison to {output_path}")

    def _plot_time_complexity(self, ax):
        """Analyze time complexity"""
        if 'database_size' in self.df_success.columns and 'num_items' in self.df_success.columns:
            # Create complexity measure: database_size * 2^(num_items)
            self.df_success['complexity'] = self.df_success['database_size'] * \
                                            (2 ** (self.df_success['num_items'] / 100))  # Scale down

            ax.scatter(self.df_success['complexity'], self.df_success['execution_time_ms'],
                       alpha=0.6, color=COLORS['time'])

            # Fit log-log relationship
            valid_data = self.df_success[['complexity', 'execution_time_ms']].dropna()
            if len(valid_data) > 2:
                x_log = np.log(valid_data['complexity'])
                y_log = np.log(valid_data['execution_time_ms'])
                z = np.polyfit(x_log, y_log, 1)

                x_fit = np.linspace(valid_data['complexity'].min(),
                                    valid_data['complexity'].max(), 100)
                y_fit = np.exp(z[1]) * (x_fit ** z[0])

                ax.plot(x_fit, y_fit, 'r--', alpha=0.7,
                        label=f'O(n^{z[0]:.2f})')

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Problem Complexity (|DB| × 2^|I|)')
            ax.set_ylabel('Execution Time (ms)')
            ax.set_title('Time Complexity Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3, which='both')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for complexity analysis',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Time Complexity Analysis')

    def _plot_memory_complexity(self, ax):
        """Analyze memory complexity"""
        if 'database_size' in self.df_success.columns:
            ax.scatter(self.df_success['database_size'],
                       self.df_success['peak_memory_mb'],
                       alpha=0.6, color=COLORS['memory'])

            # Fit linear relationship
            valid_data = self.df_success[['database_size', 'peak_memory_mb']].dropna()
            if len(valid_data) > 2:
                z = np.polyfit(valid_data['database_size'],
                               valid_data['peak_memory_mb'], 1)
                p = np.poly1d(z)

                x_fit = np.linspace(valid_data['database_size'].min(),
                                    valid_data['database_size'].max(), 100)
                ax.plot(x_fit, p(x_fit), 'r--', alpha=0.7,
                        label=f'Slope: {z[0]:.2e} MB/transaction')

            ax.set_xlabel('Database Size (transactions)')
            ax.set_ylabel('Peak Memory (MB)')
            ax.set_title('Memory Complexity Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for memory analysis',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Memory Complexity Analysis')

    def _plot_pruning_breakdown(self, ax):
        """Plot breakdown of pruning strategies"""
        pruning_cols = ['rtwu_pruned', 'branches_pruned', 'bulk_branches_pruned',
                        'eu_remaining_pruned', 'ep_pruned']

        available_cols = [col for col in pruning_cols if col in self.df_success.columns]

        if available_cols:
            pruning_data = self.df_success[available_cols].sum()

            colors_list = sns.color_palette("husl", len(pruning_data))
            wedges, texts, autotexts = ax.pie(pruning_data.values,
                                              labels=pruning_data.index,
                                              autopct='%1.1f%%',
                                              colors=colors_list,
                                              startangle=90)

            # Improve text readability
            for text in texts:
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_color('white')
                autotext.set_weight('bold')

            ax.set_title('Pruning Strategy Breakdown')
        else:
            ax.text(0.5, 0.5, 'Pruning breakdown data not available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Pruning Strategy Breakdown')

    def _plot_parallel_efficiency(self, ax):
        """Plot parallel processing efficiency"""
        if 'thread_pool_size' in self.df_success.columns:
            efficiency_data = self.df_success.groupby('thread_pool_size').agg({
                'execution_time_ms': 'mean',
                'speedup': 'mean' if 'speedup' in self.df_success.columns else lambda x: 1
            }).reset_index()

            ax2 = ax.twinx()

            ax.bar(efficiency_data['thread_pool_size'],
                   efficiency_data['execution_time_ms'],
                   alpha=0.7, color=COLORS['time'], label='Execution Time')

            if 'speedup' in efficiency_data.columns:
                ax2.plot(efficiency_data['thread_pool_size'],
                         efficiency_data['speedup'],
                         'ro-', markersize=8, linewidth=2, label='Speedup')

                # Add ideal speedup line
                ideal_speedup = efficiency_data['thread_pool_size'].values
                ax2.plot(efficiency_data['thread_pool_size'],
                         ideal_speedup / ideal_speedup[0],
                         'g--', alpha=0.5, label='Ideal Speedup')

            ax.set_xlabel('Number of Threads')
            ax.set_ylabel('Execution Time (ms)', color=COLORS['time'])
            ax2.set_ylabel('Speedup Factor', color='red')
            ax.set_title('Parallel Processing Efficiency')

            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Thread pool data not available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Parallel Processing Efficiency')

    def create_parameter_heatmaps(self):
        """Create parameter interaction heatmaps"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Heatmap 1: K vs MinProb - Execution Time
        ax1 = axes[0]
        pivot1 = self.df_success.pivot_table(
            values='execution_time_ms',
            index='min_prob',
            columns='k',
            aggfunc='mean'
        )
        sns.heatmap(pivot1, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax1)
        ax1.set_title('Execution Time (ms)')
        ax1.set_xlabel('K Value')
        ax1.set_ylabel('Min Probability')

        # Heatmap 2: K vs MinProb - Memory Usage
        ax2 = axes[1]
        pivot2 = self.df_success.pivot_table(
            values='peak_memory_mb',
            index='min_prob',
            columns='k',
            aggfunc='mean'
        )
        sns.heatmap(pivot2, annot=True, fmt='.0f', cmap='YlGnBu', ax=ax2)
        ax2.set_title('Peak Memory (MB)')
        ax2.set_xlabel('K Value')
        ax2.set_ylabel('Min Probability')

        # Heatmap 3: K vs MinProb - Pruning Effectiveness
        ax3 = axes[2]
        if 'pruning_percentage' in self.df_success.columns:
            pivot3 = self.df_success.pivot_table(
                values='pruning_percentage',
                index='min_prob',
                columns='k',
                aggfunc='mean'
            )
            sns.heatmap(pivot3, annot=True, fmt='.1f', cmap='Greens', ax=ax3,
                        vmin=0, vmax=100)
            ax3.set_title('Pruning Effectiveness (%)')
        else:
            ax3.text(0.5, 0.5, 'Pruning data not available',
                     ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Pruning Effectiveness')
        ax3.set_xlabel('K Value')
        ax3.set_ylabel('Min Probability')

        plt.suptitle('Parameter Interaction Analysis', fontsize=14, y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / 'parameter_heatmaps.png'
        plt.savefig(output_path)
        plt.close()
        print(f"Saved parameter heatmaps to {output_path}")

    def generate_latex_tables(self):
        """Generate LaTeX tables for paper inclusion"""
        output_file = self.output_dir / 'latex_tables.tex'

        with open(output_file, 'w') as f:
            # Performance comparison table
            f.write("% Performance Comparison Table\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Performance Comparison Across Datasets}\n")
            f.write("\\label{tab:performance}\n")
            f.write("\\begin{tabular}{lrrrrr}\n")
            f.write("\\hline\n")
            f.write("Dataset & Time (ms) & Memory (MB) & Pruning (\\%) & Throughput (t/s) & Speedup \\\\\n")
            f.write("\\hline\n")

            for dataset in self.df_success['dataset'].unique():
                dataset_df = self.df_success[self.df_success['dataset'] == dataset]

                time_mean = dataset_df['execution_time_ms'].mean()
                memory_mean = dataset_df['peak_memory_mb'].mean()
                pruning_mean = dataset_df.get('pruning_percentage', pd.Series([0])).mean()
                throughput = dataset_df.get('transactions_per_second', pd.Series([0])).mean()
                speedup = dataset_df.get('speedup', pd.Series([1])).mean()

                f.write(f"{dataset} & {time_mean:.1f} & {memory_mean:.1f} & "
                        f"{pruning_mean:.1f} & {throughput:.0f} & {speedup:.2f} \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")

            # Parameter sensitivity table
            f.write("% Parameter Sensitivity Table\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Parameter Sensitivity Analysis}\n")
            f.write("\\label{tab:sensitivity}\n")
            f.write("\\begin{tabular}{lrr}\n")
            f.write("\\hline\n")
            f.write("Parameter & Value & Execution Time (ms) \\\\\n")
            f.write("\\hline\n")
            f.write("\\multicolumn{3}{c}{\\textit{K Values}} \\\\\n")

            for k in sorted(self.df_success['k'].unique()):
                k_df = self.df_success[self.df_success['k'] == k]
                time_mean = k_df['execution_time_ms'].mean()
                f.write(f"K & {k} & {time_mean:.1f} \\\\\n")

            f.write("\\hline\n")
            f.write("\\multicolumn{3}{c}{\\textit{MinProb Values}} \\\\\n")

            for prob in sorted(self.df_success['min_prob'].unique()):
                prob_df = self.df_success[self.df_success['min_prob'] == prob]
                time_mean = prob_df['execution_time_ms'].mean()
                f.write(f"MinProb & {prob:.3f} & {time_mean:.1f} \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"Saved LaTeX tables to {output_file}")

    def generate_all_visualizations(self):
        """Generate all visualization outputs"""
        print("Generating research visualizations...")

        # Check if we have data
        if self.df_success.empty:
            print("Warning: No successful experiments found in data!")
            return

        print(f"Processing {len(self.df_success)} successful experiments...")

        # Generate all plots
        self.create_performance_overview()
        self.create_detailed_comparison()
        self.create_parameter_heatmaps()
        self.generate_latex_tables()

        # Generate summary report
        self.generate_summary_report()

        print(f"\nAll visualizations saved to {self.output_dir}/")

    def generate_summary_report(self):
        """Generate markdown summary report"""
        report_file = self.output_dir / 'REPORT.md'

        with open(report_file, 'w') as f:
            f.write("# PTK-HUIM-U± Performance Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")

            f.write("## Executive Summary\n\n")

            # Key metrics
            avg_time = self.df_success['execution_time_ms'].mean()
            avg_memory = self.df_success['peak_memory_mb'].mean()
            avg_pruning = self.df_success.get('pruning_percentage', pd.Series([0])).mean()

            f.write(f"- **Average Execution Time**: {avg_time:.2f} ms\n")
            f.write(f"- **Average Peak Memory**: {avg_memory:.2f} MB\n")
            f.write(f"- **Average Pruning Rate**: {avg_pruning:.1f}%\n\n")

            f.write("## Best Configurations\n\n")

            # Find best configurations
            best_time = self.df_success.nsmallest(1, 'execution_time_ms').iloc[0]
            f.write(f"### Fastest Execution\n")
            f.write(f"- Dataset: {best_time['dataset']}\n")
            f.write(f"- K: {best_time['k']}, MinProb: {best_time['min_prob']}\n")
            f.write(f"- Time: {best_time['execution_time_ms']:.0f} ms\n\n")

            if 'pruning_percentage' in self.df_success.columns:
                best_pruning = self.df_success.nlargest(1, 'pruning_percentage').iloc[0]
                f.write(f"### Best Pruning\n")
                f.write(f"- Dataset: {best_pruning['dataset']}\n")
                f.write(f"- K: {best_pruning['k']}, MinProb: {best_pruning['min_prob']}\n")
                f.write(f"- Pruning: {best_pruning['pruning_percentage']:.1f}%\n\n")

            f.write("## Figures Generated\n\n")
            f.write("1. `performance_overview.png` - Comprehensive performance analysis\n")
            f.write("2. `detailed_comparison.png` - Detailed algorithm comparisons\n")
            f.write("3. `parameter_heatmaps.png` - Parameter interaction analysis\n")
            f.write("4. `latex_tables.tex` - Tables for paper inclusion\n\n")

            f.write("## Recommendations\n\n")
            f.write("Based on the experimental results:\n\n")

            # Generate recommendations based on data
            if avg_pruning > 90:
                f.write("- ✓ Excellent pruning effectiveness (>90%)\n")
            elif avg_pruning > 80:
                f.write("- ✓ Good pruning effectiveness (>80%)\n")
            else:
                f.write("- ⚠ Consider optimizing pruning strategies\n")

            if 'cas_efficiency' in self.df_success.columns:
                cas_eff = self.df_success['cas_efficiency'].mean()
                if cas_eff > 95:
                    f.write("- ✓ Excellent CAS efficiency (>95%)\n")
                elif cas_eff > 90:
                    f.write("- ✓ Good CAS efficiency (>90%)\n")
                else:
                    f.write("- ⚠ Consider optimizing lock-free operations\n")

        print(f"Saved summary report to {report_file}")


def main():
    """Main entry point"""
    import sys

    if len(sys.argv) < 2:
        # Try to find the most recent results file
        results_dir = Path('exp_results')
        if results_dir.exists():
            json_files = list(results_dir.glob('results_*.json'))
            if json_files:
                results_file = max(json_files, key=lambda p: p.stat().st_mtime)
                print(f"Using most recent results file: {results_file}")
            else:
                print("No results files found. Run experiments.py first.")
                sys.exit(1)
        else:
            print("No exp_results directory found. Run experiments.py first.")
            sys.exit(1)
    else:
        results_file = sys.argv[1]

    # Create visualizer and generate all outputs
    visualizer = ResearchVisualizer(str(results_file))
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()