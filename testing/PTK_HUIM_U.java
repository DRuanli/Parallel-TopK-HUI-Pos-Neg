package testing;

import testing.core.*;
import testing.io.*;
import testing.mining.*;
import testing.parallel.*;
import java.io.IOException;
import java.time.*;
import java.util.*;

/**
 * PTK-HUIM-U±: Enhanced Parallel Top-K High-Utility Itemset Mining
 * from Uncertain Databases with Positive and Negative Utilities
 *
 * VERSION 5.2 OPTIMIZED - ULTIMATE HYBRID
 * - CAS-based TopKManager from ver5_2 (maximum parallel performance)
 * - Pre-computed EnhancedUtilityList from ver5_2 (O(1) utility access)
 * - Optimized initialization from ver4_9 (suffix sum preprocessing)
 * - Best algorithmic AND runtime optimizations combined!
 *
 * @author Enhanced Merge
 * @version 5.2-optimized
 */
public class PTK_HUIM_U {

    public static void main(String[] args) throws IOException {
        if (args.length != 4) {
            System.err.println("Usage: PTK_HUIM_U <database_file> <profit_file> <k> <min_probability>");
            System.exit(1);
        }

        String dbFile = args[0];
        String profitFile = args[1];
        int k = Integer.parseInt(args[2]);
        double minPro = Double.parseDouble(args[3]);

        // Load data
        Map<Integer, Double> profits = DataLoader.readProfitTable(profitFile);
        List<Transaction> database = DataLoader.readDatabase(dbFile);

        System.out.println("=== PTK-HUIM-U± Version 5.2 OPTIMIZED ===");
        System.out.println("Ultimate hybrid: CAS TopKManager + Pre-computed utilities + Suffix sum optimization");
        System.out.println();

        // Create and configure mining engine
        MiningEngine miningEngine = new MiningEngine(profits, k, minPro);

        // Print configuration
        ResultWriter.printConfiguration(
            database.size(),
            profits.size(),
            k,
            minPro,
            Runtime.getRuntime().maxMemory(),
            miningEngine.getTaskScheduler().getParallelism()
        );

        // Start mining
        Instant start = Instant.now();
        List<Itemset> topK = miningEngine.mine(database);
        Instant end = Instant.now();

        // Print results
        Duration executionTime = Duration.between(start, end);
        ResultWriter.printResults(
            topK,
            executionTime,
            miningEngine.getStatistics(),
            miningEngine.getPeakMemoryUsage(),
            miningEngine.getTopKManager(),
            k
        );

        // Cleanup
        miningEngine.getTaskScheduler().shutdown();
    }
}