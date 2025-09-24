"""
PTK-HUIM Algorithm Testing Suite
Tests multiple versions of the algorithm against various datasets
"""

import os
import subprocess
import time
import re
import csv
from pathlib import Path
import json
from datetime import datetime
import sys
import concurrent.futures
from typing import Dict, List, Tuple

# Configuration
CONFIG = {
    'java_files_dir': '.',
    'data_dir': 'data',
    'output_dir': 'exp_results',
    'k': 100,
    'min_prob': 0.001,
    'timeout': 300000,  # seconds
    'max_workers': 4
}

# Dataset definitions (database_file, profit_file, name)
DATASETS = [
    ('test_database.txt', 'test_profits.txt', 'test'),
    ('accident_database.txt', 'accident_profits.txt', 'accident'),
    ('chess_datasets.txt', 'chess_profits.txt', 'chess'),
    ('connect_database.txt', 'connect_profits.txt', 'connect'),
    ('kosarak_database.txt', 'kosarak_profits.txt', 'kosarak'),
    ('mushroom_database.txt', 'mushroom_profits.txt', 'mushroom'),
    ('retail_database.txt', 'retail_profits.txt', 'retail'),
    # ('large_db.txt', 'large_profit.txt', 'large'),
]


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


class AlgorithmTester:
    def __init__(self, config: Dict):
        self.config = config
        self.results = []
        self.create_directories()

    def create_directories(self):
        """Create necessary output directories"""
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)

    def log(self, message: str, color: str = ''):
        """Print colored log message"""
        print(f"{color}{message}{Colors.NC}")

    def find_java_files(self) -> List[Tuple[str, str]]:
        """Find all version files and extract version numbers"""
        java_files = []
        for file in Path(self.config['java_files_dir']).glob('ver*.java'):
            version_match = re.search(r'ver(\d+(?:\.\d+)?)', file.name)
            if version_match:
                version = version_match.group(1)
                java_files.append((str(file), version))
        return sorted(java_files, key=lambda x: float(x[1]) if '.' in x[1] else int(x[1]))

    def detect_class_name(self, java_file: str) -> str:
        """Detect the main class name from Java file"""
        with open(java_file, 'r') as f:
            content = f.read()
            if 'class PTKHUIMCorrect' in content:
                return 'PTKHUIMCorrect'
            elif 'class PTKHUIM_old' in content:
                return 'PTKHUIM_old'
            elif 'class PTKHUIM' in content:
                return 'PTKHUIM'
        return 'PTKHUIM'

    def compile_java(self, java_file: str, class_name: str) -> bool:
        """Compile Java file"""
        self.log(f"Compiling {java_file}...", Colors.YELLOW)
        try:
            result = subprocess.run(
                ['javac', java_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                self.log("✓ Compilation successful", Colors.GREEN)
                return True
            else:
                self.log(f"✗ Compilation failed: {result.stderr}", Colors.RED)
                return False
        except subprocess.TimeoutExpired:
            self.log("✗ Compilation timeout", Colors.RED)
            return False
        except Exception as e:
            self.log(f"✗ Compilation error: {str(e)}", Colors.RED)
            return False

    def run_test(self, class_name: str, db_file: str, profit_file: str,
                 dataset_name: str, version: str) -> Dict:
        """Run a single test"""
        output_file = f"{self.config['output_dir']}/ver{version}_{dataset_name}.txt"

        self.log(f"  Testing {dataset_name} dataset...", Colors.CYAN)

        db_path = f"{self.config['data_dir']}/{db_file}"
        profit_path = f"{self.config['data_dir']}/{profit_file}"

        # Check if files exist
        if not os.path.exists(db_path) or not os.path.exists(profit_path):
            self.log(f"  ⚠ Data files not found for {dataset_name}", Colors.YELLOW)
            return {'status': 'SKIPPED', 'reason': 'Data files not found'}

        start_time = time.time()

        try:
            result = subprocess.run(
                ['java', class_name, db_path, profit_path,
                 str(self.config['k']), str(self.config['min_prob'])],
                capture_output=True,
                text=True,
                timeout=self.config['timeout']
            )

            execution_time = (time.time() - start_time) * 1000  # Convert to ms

            # Save output
            with open(output_file, 'w') as f:
                f.write(result.stdout)
                f.write(result.stderr)

            if result.returncode == 0:
                self.log(f"  ✓ Test completed in {execution_time:.0f}ms", Colors.GREEN)
                metrics = self.extract_metrics(result.stdout)
                metrics['status'] = 'SUCCESS'
                metrics['execution_time_actual'] = execution_time
                return metrics
            else:
                self.log(f"  ✗ Test failed with error code {result.returncode}", Colors.RED)
                return {'status': 'FAILED', 'error_code': result.returncode}

        except subprocess.TimeoutExpired:
            self.log(f"  ✗ Test timeout (>{self.config['timeout']}s)", Colors.RED)
            with open(output_file, 'w') as f:
                f.write(f"TIMEOUT after {self.config['timeout']} seconds\n")
            return {'status': 'TIMEOUT'}

        except Exception as e:
            self.log(f"  ✗ Test error: {str(e)}", Colors.RED)
            return {'status': 'ERROR', 'error': str(e)}

    def extract_metrics(self, output: str) -> Dict:
        """Extract metrics from program output"""
        metrics = {
            'execution_time': None,
            'candidates_generated': None,
            'candidates_pruned': None,
            'memory_usage': None,
            'topk_found': None,
            'utility_lists': None,
            'rtwu_pruned': None,
            'eucp_pruned': None,
            'ep_pruned': None,
            'eu_pruned': None
        }

        patterns = {
            'execution_time': r'(?:execution time|Total execution time):\s*(\d+)\s*ms',
            'candidates_generated': r'candidates generated:\s*(\d+)',
            'candidates_pruned': r'candidates pruned:\s*(\d+)',
            'memory_usage': r'peak memory.*?(\d+)\s*MB',
            'topk_found': r'top-k found:\s*(\d+)',
            'utility_lists': r'utility lists created:\s*(\d+)',
            'rtwu_pruned': r'RTWU pruned:\s*(\d+)',
            'eucp_pruned': r'EUCP pruned:\s*(\d+)',
            'ep_pruned': r'Existential probability pruned:\s*(\d+)',
            'eu_pruned': r'EU\+remaining pruned:\s*(\d+)'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                metrics[key] = int(match.group(1))

        return metrics

    def test_version(self, java_file: str, version: str) -> List[Dict]:
        """Test a single version against all datasets"""
        self.log(f"\n{'=' * 60}", Colors.BLUE)
        self.log(f"Processing Version {version}", Colors.BLUE)
        self.log(f"{'=' * 60}", Colors.BLUE)

        class_name = self.detect_class_name(java_file)

        # Compile
        if not self.compile_java(java_file, class_name):
            self.log(f"Skipping version {version} due to compilation error", Colors.YELLOW)
            return []

        version_results = []

        # Test with each dataset
        for db_file, profit_file, dataset_name in DATASETS:
            self.log(f"\nTesting Version {version} on {dataset_name}:", Colors.MAGENTA)

            result = self.run_test(class_name, db_file, profit_file, dataset_name, version)
            result['version'] = version
            result['dataset'] = dataset_name
            version_results.append(result)

        # Clean up compiled files
        for file in Path('.').glob('*.class'):
            file.unlink()

        return version_results

    def run_all_tests(self):
        """Run all tests"""
        self.log("=" * 60, Colors.BLUE)
        self.log("PTK-HUIM Algorithm Testing Suite", Colors.BLUE)
        self.log("=" * 60, Colors.BLUE)
        self.log(f"Configuration:")
        self.log(f"  K = {self.config['k']}")
        self.log(f"  MinPro = {self.config['min_prob']}")
        self.log(f"  Timeout = {self.config['timeout']}s")
        self.log(f"  Output directory: {self.config['output_dir']}")
        self.log("=" * 60 + "\n", Colors.BLUE)

        java_files = self.find_java_files()

        if not java_files:
            self.log("No version files found!", Colors.RED)
            return

        self.log(f"Found {len(java_files)} versions to test\n")

        # Test each version
        all_results = []
        for java_file, version in java_files:
            version_results = self.test_version(java_file, version)
            all_results.extend(version_results)

        self.results = all_results

        # Save results
        self.save_results()
        self.print_summary()

    def save_results(self):
        """Save results to CSV and JSON"""
        # Save to CSV
        csv_file = f"{self.config['output_dir']}/summary.csv"
        with open(csv_file, 'w', newline='') as f:
            if self.results:
                fieldnames = list(self.results[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)

        # Save to JSON
        json_file = f"{self.config['output_dir']}/summary.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        self.log(f"\nResults saved:", Colors.GREEN)
        self.log(f"  CSV: {csv_file}")
        self.log(f"  JSON: {json_file}")

    def print_summary(self):
        """Print test summary"""
        self.log("\n" + "=" * 60, Colors.BLUE)
        self.log("Testing Complete!", Colors.BLUE)
        self.log("=" * 60, Colors.BLUE)

        total = len(self.results)
        successful = sum(1 for r in self.results if r.get('status') == 'SUCCESS')
        failed = sum(1 for r in self.results if r.get('status') == 'FAILED')
        timeout = sum(1 for r in self.results if r.get('status') == 'TIMEOUT')
        skipped = sum(1 for r in self.results if r.get('status') == 'SKIPPED')

        self.log(f"Total tests: {total}")
        self.log(f"  Successful: {successful}", Colors.GREEN)
        self.log(f"  Failed: {failed}", Colors.RED if failed > 0 else '')
        self.log(f"  Timeout: {timeout}", Colors.YELLOW if timeout > 0 else '')
        self.log(f"  Skipped: {skipped}", Colors.YELLOW if skipped > 0 else '')

        # Find best performing versions
        if successful > 0:
            self.log("\n" + "=" * 60, Colors.BLUE)
            self.log("Best Performing Versions by Dataset:", Colors.BLUE)
            self.log("=" * 60, Colors.BLUE)

            datasets = set(r['dataset'] for r in self.results if r.get('status') == 'SUCCESS')
            for dataset in sorted(datasets):
                dataset_results = [r for r in self.results
                                   if r['dataset'] == dataset and r.get('status') == 'SUCCESS']
                if dataset_results:
                    best = min(dataset_results,
                               key=lambda x: x.get('execution_time', float('inf')))
                    self.log(f"  {dataset}: Version {best['version']} "
                             f"({best.get('execution_time', 'N/A')}ms)")

    def create_comparison_script(self):
        """Create a comparison analysis script"""
        script_path = f"{self.config['output_dir']}/analyze_results.py"

        script_content = '''#!/usr/bin/env python3
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Read results
with open('summary.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df_success = df[df['status'] == 'SUCCESS'].copy()

if not df_success.empty:
    # Convert version to numeric for sorting
    df_success['version_num'] = pd.to_numeric(df_success['version'], errors='coerce')

    print("\\n=== Performance Summary by Version ===")
    version_summary = df_success.groupby('version').agg({
        'execution_time': ['mean', 'std'],
        'memory_usage': 'mean',
        'candidates_generated': 'mean',
        'candidates_pruned': 'mean'
    }).round(2)
    print(version_summary)

    print("\\n=== Performance Summary by Dataset ===")
    dataset_summary = df_success.groupby('dataset').agg({
        'execution_time': ['mean', 'min', 'max'],
        'topk_found': 'mean'
    }).round(2)
    print(dataset_summary)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Execution time by version
    pivot_time = df_success.pivot(index='dataset', columns='version', values='execution_time')
    pivot_time.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Execution Time by Version and Dataset')
    axes[0, 0].set_ylabel('Time (ms)')

    # Memory usage by version
    if 'memory_usage' in df_success.columns:
        version_memory = df_success.groupby('version')['memory_usage'].mean()
        version_memory.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Average Memory Usage by Version')
        axes[0, 1].set_ylabel('Memory (MB)')

    # Pruning effectiveness
    if 'candidates_pruned' in df_success.columns and 'candidates_generated' in df_success.columns:
        df_success['prune_ratio'] = df_success['candidates_pruned'] / df_success['candidates_generated'].replace(0, 1)
        version_prune = df_success.groupby('version')['prune_ratio'].mean()
        version_prune.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Pruning Effectiveness by Version')
        axes[1, 0].set_ylabel('Prune Ratio')

    # Dataset complexity (avg execution time)
    dataset_time = df_success.groupby('dataset')['execution_time'].mean().sort_values()
    dataset_time.plot(kind='barh', ax=axes[1, 1])
    axes[1, 1].set_title('Dataset Complexity (Avg Execution Time)')
    axes[1, 1].set_xlabel('Time (ms)')

    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\\nVisualization saved as 'performance_analysis.png'")
'''

        with open(script_path, 'w') as f:
            f.write(script_content)

        os.chmod(script_path, 0o755)
        self.log(f"\nAnalysis script created: {script_path}")


def main():
    """Main entry point"""
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("Usage: python test_algorithms.py ")

            return

    tester = AlgorithmTester(CONFIG)
    tester.run_all_tests()
    tester.create_comparison_script()

    print(f"\nTo analyze results, run:")
    print(f"  cd {CONFIG['output_dir']} && python3 analyze_results.py")


if __name__ == "__main__":
    main()