package testing.io;

import testing.core.Itemset;
import testing.mining.MiningStatistics;
import testing.parallel.TopKManager;
import java.time.Duration;
import java.util.List;

/**
 * Result writer for formatting and displaying mining results
 */
public class ResultWriter {

    /**
     * Print mining configuration
     */
    public static void printConfiguration(int databaseSize, int itemCount, int k, double minPro,
                                         long maxMemory, int threadPoolSize) {
        System.out.println("=== PTK-HUIM-U± v5.2 OPTIMIZED (Ultimate Hybrid) ===");
        System.out.println("CAS TopKManager + Pre-computed UtilityList + Suffix Sum Optimization!");
        System.out.println("Database size: " + databaseSize);
        System.out.println("Number of items: " + itemCount);
        System.out.println("K: " + k + ", MinPro: " + minPro);
        System.out.println("Available memory: " + (maxMemory / 1024 / 1024) + " MB");
        System.out.println("Thread pool size: " + threadPoolSize);
    }

    /**
     * Print mining results and statistics
     */
    public static void printResults(List<Itemset> topK, Duration executionTime,
                                   MiningStatistics statistics, long peakMemoryUsage,
                                   TopKManager topKManager, int k) {
        System.out.println("\n=== Mining Complete ===");
        System.out.println("Execution time: " + executionTime.toMillis() + " ms");

        statistics.printStatistics();

        System.out.println("Peak memory usage: " + (peakMemoryUsage / 1024 / 1024) + " MB");
        System.out.println("Final threshold: " + String.format("%.4f", topKManager.getThreshold()));
        System.out.println("Top-K found: " + topK.size());

        // CAS Performance statistics - restored from ver5_2!
        System.out.println("Lock-free TopK statistics:");
        System.out.println("  - Successful updates: " + topKManager.getSuccessfulUpdates());
        System.out.println("  - CAS retries: " + topKManager.getCASRetries());
        if (topKManager.getSuccessfulUpdates() + topKManager.getCASRetries() > 0) {
            double casEfficiency = (double) topKManager.getSuccessfulUpdates() /
                (topKManager.getSuccessfulUpdates() + topKManager.getCASRetries());
            System.out.println("  - CAS efficiency: " + String.format("%.2f%%", casEfficiency * 100));
        }

        System.out.println("\n=== Top-" + k + " PHUIs ===");
        int rank = 1;
        for (Itemset itemset : topK) {
            System.out.printf("%d. %s\n", rank++, itemset);
        }
    }

    /**
     * Print progress update during mining
     */
    public static void printProgress(int current, int total, long memoryUsed) {
        System.out.println("Progress: " + (current + 1) + "/" + total +
                         " items processed. Memory used: " + (memoryUsed / 1024 / 1024) + " MB");
    }
}