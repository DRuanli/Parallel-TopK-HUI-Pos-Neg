# PTK-HUIM-U±: Parallel Top-K High-Utility Itemset Mining from Uncertain Databases

[![Java](https://img.shields.io/badge/Java-11%2B-orange)](https://openjdk.java.net/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

PTK-HUIM-U± is an advanced parallel algorithm for mining top-K high-utility itemsets from uncertain databases with both positive and negative utilities. This implementation features multiple optimizations for maximum performance in large-scale data mining tasks.

## Key Features

- **Parallel Processing**: ForkJoinPool-based parallelization for multi-core systems
- **Lock-Free Operations**: CAS-based TopKManager for thread-safe operations
- **Optimized Data Structures**: Pre-computed utility lists with O(1) access
- **Advanced Pruning**: Multiple pruning strategies (RTWU, EU, EP, bulk pruning)
- **Memory Efficient**: Suffix sum preprocessing eliminates O(T²) complexity
- **Numerical Stability**: Log-space probability computations prevent underflow

## Algorithm Highlights

### Core Optimizations

1. **Suffix Sum Preprocessing**: Reduces remaining utility calculation from O(T²) to O(T)
2. **Pre-computed Utility Lists**: Eliminates lazy evaluation overhead
3. **CAS-based TopK Management**: Lock-free concurrent updates
4. **Single-pass RTWU Calculation**: Reduces initialization complexity
5. **Bulk Branch Pruning**: Prunes entire subtrees in parallel tasks

### Performance Characteristics

- **Time Complexity**: O(|DB| × 2^|I|) worst case, significantly reduced by pruning
- **Space Complexity**: O(|DB| × |I|) for utility lists
- **Parallel Efficiency**: Near-linear speedup with available cores

## Installation

### Prerequisites

- Java 11 or higher
- Python 3.8+ (for experiments and visualization)
- At least 4GB RAM (8GB+ recommended for large datasets)

### Setup

```bash
# Clone the repository
git clone https://github.com/DRuanli/Parallel-TopK-HUI-Pos-Neg.git
cd PTK-HUIM-U

# Compile Java sources
javac -d testing/build -cp . testing/**/*.java
```

## Usage

### Basic Usage

```bash
# Run the main algorithm
java -cp testing/build testing.PTK_HUIM_U <database_file> <profit_file> <k> <min_probability>

# Example
java -cp testing/build testing.PTK_HUIM_U data/database.txt data/profits.txt 100 0.001
```

### Input Format

**Database File** (database.txt):
```
item:quantity:probability item:quantity:probability ...
```
Example:
```
2:3:0.85 3:1:1.0 4:2:0.70
1:1:1.0 2:1:0.60 3:3:0.75
```

**Profit File** (profits.txt):
```
item profit
```
Example:
```
1 6.0
2 7.0
3 1.0
4 -5.0
```

### Running Experiments

```bash
# Run performance experiments on all datasets
python experiments.py

# Generate visualization reports
python exp_visualize.py
```

## Project Structure

```
PTK-HUIM-U/
├── testing/
│   ├── core/           # Core data structures
│   │   ├── Itemset.java
│   │   ├── Transaction.java
│   │   └── UtilityList.java
│   ├── mining/         # Mining algorithms
│   │   ├── MiningEngine.java
│   │   ├── JoinOperator.java
│   │   └── PruningStrategy.java
│   ├── parallel/       # Parallel execution
│   │   ├── TopKManager.java
│   │   └── TaskScheduler.java
│   └── io/            # Input/Output utilities
├── data/              # Dataset files
├── exp_results/       # Experiment results
└── src/main/         # Version implementations

```

## Algorithm Parameters

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| k | Number of top itemsets | 50-500            |
| minPro | Minimum existential probability | 0.0001-0.1        |
| PARALLEL_THRESHOLD | Min items for parallel processing | 20-50             |
| TASK_GRANULARITY | Items per task | 5-10              |

## Performance Tuning

### Memory Settings

```bash
# Increase heap size for large datasets
java -Xmx8G -Xms4G testing.PTK_HUIM_U database.txt profits.txt 100 0.001
```

### Thread Pool Configuration

The algorithm automatically uses all available CPU cores. To limit:

```java
// In MiningEngine constructor
int numThreads = Math.min(Runtime.getRuntime().availableProcessors(), 8);
```

## Experimental Results

### Datasets Used

| Dataset | Transactions | Items | Avg Length | Type |
|---------|-------------|-------|------------|------|
| Chess | 3,196 | 75 | 37 | Dense |
| Mushroom | 8,124 | 119 | 23 | Dense |
| Retail | 88,162 | 16,470 | 10.3 | Sparse |
| Kosarak | 990,002 | 41,270 | 8.1 | Sparse |



## Contact

For questions or support, please open an issue on GitHub or contact [elio.ruanli@gmail.com]