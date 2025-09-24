"""
PTK-HUIM Algorithm Performance Testing Suite
Tracks execution time, memory usage, and pruning effectiveness
"""

import os
import subprocess
import time
import re
import csv
import json
import psutil
import gc
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import threading

# Configuration
CONFIG = {
    'java_dir': 'testing',
    'data_dir': 'data',
    'output_dir': 'exp_results',
    'k_values': [50, 100, 200],
    'min_prob_values': [0.001, 0.005, 0.01],
    'timeout': 600,  # seconds
    'java_heap': '4G',
    'repetitions': 3  # Run each experiment 3 times
}

# Dataset definitions
DATASETS = [
    ('database.txt', 'profits.txt', 'small_test'),
    # Add your actual datasets here
    # ('chess_database.txt', 'chess_profits.txt', 'chess'),
    # ('mushroom_database.txt', 'mushroom_profits.txt', 'mushroom'),
    # ('retail_database.txt', 'retail_profits.txt', 'retail'),
]


class PerformanceTester:
    def __init__(self, config: Dict):
        self.config = config
        self.results = []
        self.create_directories()

    def create_directories(self):
        """Create necessary output directories"""
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)

    def monitor_memory(self, pid: int, interval: float = 0.1) -> Dict:
        """Monitor memory usage of a process"""
        memory_samples = []
        cpu_samples = []

        try:
            process = psutil.Process(pid)
            while process.is_running():
                try:
                    mem_info = process.memory_info()
                    memory_samples.append(mem_info.rss / (1024 * 1024))  # MB
                    cpu_samples.append(process.cpu_percent())
                    time.sleep(interval)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        except psutil.NoSuchProcess:
            pass

        if memory_samples:
            return {
                'peak_memory_mb': max(memory_samples),
                'avg_memory_mb': sum(memory_samples) / len(memory_samples),
                'min_memory_mb': min(memory_samples),
                'avg_cpu_percent': sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
            }
        return {'peak_memory_mb': 0, 'avg_memory_mb': 0, 'min_memory_mb': 0, 'avg_cpu_percent': 0}

    def run_single_experiment(self, dataset: Tuple, k: int, min_prob: float, rep: int) -> Dict:
        """Run a single experiment and collect metrics"""
        db_file, profit_file, dataset_name = dataset

        db_path = f"{self.config['data_dir']}/{db_file}"
        profit_path = f"{self.config['data_dir']}/{profit_file}"

        # Check if files exist
        if not os.path.exists(db_path) or not os.path.exists(profit_path):
            return {'status': 'SKIPPED', 'reason': 'Data files not found'}

        # Prepare command
        cmd = [
            'java',
            f'-Xmx{self.config["java_heap"]}',
            f'-Xms{self.config["java_heap"]}',
            '-XX:+UseG1GC',
            'testing.PTK_HUIM_U',
            db_path,
            profit_path,
            str(k),
            str(min_prob)
        ]

        print(f"Running: {dataset_name}, k={k}, minPro={min_prob}, rep={rep+1}")

        # Start process
        start_time = time.perf_counter()
        gc.collect()  # Clean up before starting

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Monitor memory in background thread
            memory_monitor = {}
            monitor_thread = threading.Thread(
                target=lambda: memory_monitor.update(
                    self.monitor_memory(process.pid)
                )
            )
            monitor_thread.start()

            # Wait for completion
            stdout, stderr = process.communicate(timeout=self.config['timeout'])
            execution_time = (time.perf_counter() - start_time) * 1000  # ms

            monitor_thread.join()

            if process.returncode == 0:
                # Extract metrics from output
                metrics = self.extract_comprehensive_metrics(stdout)
                metrics.update({
                    'status': 'SUCCESS',
                    'dataset': dataset_name,
                    'k': k,
                    'min_prob': min_prob,
                    'repetition': rep + 1,
                    'execution_time_ms': execution_time,
                    'return_code': process.returncode
                })
                metrics.update(memory_monitor)

                return metrics
            else:
                return {
                    'status': 'FAILED',
                    'dataset': dataset_name,
                    'k': k,
                    'min_prob': min_prob,
                    'repetition': rep + 1,
                    'error': stderr,
                    'return_code': process.returncode
                }

        except subprocess.TimeoutExpired:
            process.kill()
            return {
                'status': 'TIMEOUT',
                'dataset': dataset_name,
                'k': k,
                'min_prob': min_prob,
                'repetition': rep + 1,
                'timeout_seconds': self.config['timeout']
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'dataset': dataset_name,
                'k': k,
                'min_prob': min_prob,
                'repetition': rep + 1,
                'error': str(e)
            }

    def extract_comprehensive_metrics(self, output: str) -> Dict:
        """Extract all metrics from program output"""
        metrics = {}

        # Patterns for various metrics
        patterns = {
            # Time metrics
            'execution_time': r'Execution time:\s*(\d+)\s*ms',

            # Item/database metrics
            'database_size': r'Database size:\s*(\d+)',
            'num_items': r'Number of items:\s*(\d+)',
            'items_after_filter': r'Items after filtering:\s*(\d+)',

            # Generation metrics
            'candidates_generated': r'Candidates generated:\s*(\d+)',
            'utility_lists_created': r'Utility lists created:\s*(\d+)',

            # Pruning metrics
            'rtwu_pruned': r'RTWU pruned:\s*(\d+)',
            'branches_pruned': r'Branches pruned:\s*(\d+)',
            'bulk_branches_pruned': r'Bulk branches pruned:\s*(\d+)',
            'eu_remaining_pruned': r'EU\+remaining pruned:\s*(\d+)',
            'ep_pruned': r'Existential probability pruned:\s*(\d+)',
            'total_pruned': r'Total pruned:\s*(\d+)',

            # Memory metrics
            'peak_memory_reported': r'Peak memory usage:\s*(\d+)\s*MB',

            # Result metrics
            'topk_found': r'Top-K found:\s*(\d+)',
            'final_threshold': r'Final threshold:\s*([\d.]+)',

            # CAS metrics
            'successful_updates': r'Successful updates:\s*(\d+)',
            'cas_retries': r'CAS retries:\s*(\d+)',
            'cas_efficiency': r'CAS efficiency:\s*([\d.]+)%',

            # Thread pool
            'thread_pool_size': r'Thread pool size:\s*(\d+)'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                value = match.group(1)
                try:
                    # Try to convert to appropriate type
                    if '.' in value:
                        metrics[key] = float(value)
                    else:
                        metrics[key] = int(value)
                except ValueError:
                    metrics[key] = value

        # Calculate pruning effectiveness
        if 'candidates_generated' in metrics and 'total_pruned' in metrics:
            total = metrics['candidates_generated']
            pruned = metrics['total_pruned']
            if total > 0:
                metrics['pruning_ratio'] = pruned / total
                metrics['pruning_percentage'] = (pruned / total) * 100

        # Calculate throughput
        if 'execution_time' in metrics and 'database_size' in metrics:
            time_sec = metrics['execution_time'] / 1000
            if time_sec > 0:
                metrics['transactions_per_second'] = metrics['database_size'] / time_sec

        return metrics

    def run_experiments(self):
        """Run all experiments"""
        print("=" * 80)
        print("PTK-HUIM Performance Testing Suite")
        print("=" * 80)
        print(f"Datasets: {len(DATASETS)}")
        print(f"K values: {self.config['k_values']}")
        print(f"MinProb values: {self.config['min_prob_values']}")
        print(f"Repetitions: {self.config['repetitions']}")
        print(f"Total experiments: {len(DATASETS) * len(self.config['k_values']) * len(self.config['min_prob_values']) * self.config['repetitions']}")
        print("=" * 80)

        # Compile Java code first
        print("\nCompiling Java code...")
        compile_cmd = ['javac', '-d', '.', 'testing/**/*.java']
        compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if compile_result.returncode != 0:
            print(f"Compilation failed: {compile_result.stderr}")
            return
        print("Compilation successful!")

        # Run experiments
        experiment_count = 0
        total_experiments = len(DATASETS) * len(self.config['k_values']) * \
                          len(self.config['min_prob_values']) * self.config['repetitions']

        for dataset in DATASETS:
            for k in self.config['k_values']:
                for min_prob in self.config['min_prob_values']:
                    rep_results = []
                    for rep in range(self.config['repetitions']):
                        experiment_count += 1
                        print(f"\n[{experiment_count}/{total_experiments}] ", end="")

                        result = self.run_single_experiment(dataset, k, min_prob, rep)
                        rep_results.append(result)
                        self.results.append(result)

                        # Save intermediate results
                        if experiment_count % 10 == 0:
                            self.save_results()

                    # Calculate and store average metrics for this configuration
                    self.calculate_average_metrics(rep_results)

        # Save final results
        self.save_results()
        self.generate_summary_report()

    def calculate_average_metrics(self, rep_results: List[Dict]):
        """Calculate average metrics across repetitions"""
        successful = [r for r in rep_results if r.get('status') == 'SUCCESS']

        if successful:
            avg_metrics = {
                'dataset': successful[0]['dataset'],
                'k': successful[0]['k'],
                'min_prob': successful[0]['min_prob'],
                'repetitions_successful': len(successful),
                'type': 'AVERAGE'
            }

            # Metrics to average
            metrics_to_avg = [
                'execution_time_ms', 'peak_memory_mb', 'avg_memory_mb',
                'candidates_generated', 'total_pruned', 'pruning_percentage',
                'transactions_per_second', 'cas_efficiency'
            ]

            for metric in metrics_to_avg:
                values = [r.get(metric, 0) for r in successful if metric in r]
                if values:
                    avg_metrics[f'avg_{metric}'] = sum(values) / len(values)
                    avg_metrics[f'std_{metric}'] = self.calculate_std(values)

            self.results.append(avg_metrics)

    def calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def save_results(self):
        """Save results to CSV and JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results to CSV
        csv_file = f"{self.config['output_dir']}/results_{timestamp}.csv"
        if self.results:
            fieldnames = set()
            for result in self.results:
                fieldnames.update(result.keys())

            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                writer.writeheader()
                writer.writerows(self.results)

        # Save to JSON for easier processing
        json_file = f"{self.config['output_dir']}/results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to:")
        print(f"  - {csv_file}")
        print(f"  - {json_file}")

    def generate_summary_report(self):
        """Generate a summary report"""
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)

        successful = [r for r in self.results if r.get('status') == 'SUCCESS']
        failed = [r for r in self.results if r.get('status') == 'FAILED']
        timeout = [r for r in self.results if r.get('status') == 'TIMEOUT']

        print(f"Total experiments: {len(self.results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Timeout: {len(timeout)}")

        if successful:
            # Performance summary
            print("\nPerformance Summary (averages):")
            avg_time = sum(r.get('execution_time_ms', 0) for r in successful) / len(successful)
            avg_memory = sum(r.get('peak_memory_mb', 0) for r in successful) / len(successful)
            avg_pruning = sum(r.get('pruning_percentage', 0) for r in successful) / len(successful)

            print(f"  Execution time: {avg_time:.2f} ms")
            print(f"  Peak memory: {avg_memory:.2f} MB")
            print(f"  Pruning effectiveness: {avg_pruning:.1f}%")

            # Best configurations
            print("\nBest Configurations:")
            best_time = min(successful, key=lambda x: x.get('execution_time_ms', float('inf')))
            print(f"  Fastest: {best_time['dataset']} with k={best_time['k']}, "
                  f"minPro={best_time['min_prob']} ({best_time['execution_time_ms']:.0f} ms)")

            best_pruning = max(successful, key=lambda x: x.get('pruning_percentage', 0))
            print(f"  Best pruning: {best_pruning['dataset']} with k={best_pruning['k']}, "
                  f"minPro={best_pruning['min_prob']} ({best_pruning['pruning_percentage']:.1f}%)")


def main():
    """Main entry point"""
    tester = PerformanceTester(CONFIG)
    tester.run_experiments()


if __name__ == "__main__":
    main()