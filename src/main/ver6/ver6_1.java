package main.ver6;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.stream.*;
import java.time.*;
import java.io.*;

/**
 * PTK-HUIM-U±: Enhanced Parallel Top-K High-Utility Itemset Mining
 * from Uncertain Databases with Positive and Negative Utilities
 *
 * VERSION 6.1 IMPROVEMENTS:
 * 1. Try to Apply LA UB
 *
 * @author Elio
 * @version 6.1
 */
public class ver6_1 {
    private final Map<Integer, Double> itemProfits;
    private final int k;
    private final double minPro;
    private static final double EPSILON = 1e-10;
    private static final double LOG_EPSILON = -700; // exp(-700) ≈ 0
    private static final double LOG_ONE_MINUS_EPSILON = Math.log(1.0 - 1e-10); // For probability ≈ 1 checks

    // Thread-safe top-K management
    private final TopKManager topKManager;

    // Optimized item ordering - removed redundant structures
    private Map<Integer, Integer> itemToRank;
    private Map<Integer, Double> itemRTWU;  // Store RTWU values for dynamic pruning

    // Enhanced statistics - thread-safe
    private final AtomicLong candidatesGenerated = new AtomicLong(0);
    private final AtomicLong candidatesPruned = new AtomicLong(0);
    private final AtomicLong utilityListsCreated = new AtomicLong(0);
    private final AtomicLong euPruned = new AtomicLong(0);
    private final AtomicLong epPruned = new AtomicLong(0);
    private final AtomicLong rtwuPruned = new AtomicLong(0);
    private final AtomicLong branchPruned = new AtomicLong(0);
    private final AtomicLong bulkBranchPruned = new AtomicLong(0);

    private final AtomicLong lookAheadAttempts = new AtomicLong(0);
    private final AtomicLong lookAheadPruned = new AtomicLong(0);
    private final AtomicLong lookAheadJoinsPerformed = new AtomicLong(0);
    private final AtomicLong lookAheadTimeMs = new AtomicLong(0);

    private static final class HybridConfig {
        // Thresholds for triggering Look-Ahead
        static final int MAX_PREFIX_SIZE = 5000000;           // Only apply at top levels
        static final int MIN_EXTENSIONS_FOR_LOOKAHEAD = 0;  // Need many branches
        static final double THRESHOLD_RATIO = 0.7;      // When threshold is high relative to max
        static final int MAX_LOOKAHEAD_SAMPLES = 20;    // Limit samples to control cost
        static final boolean ENABLE_LOOKAHEAD = true;   // Master switch

        // Adaptive thresholds based on performance
        static final long MAX_LOOKAHEAD_TIME_MS = 100;  // Max time per look-ahead
    }

    // Control parallel execution - optimized parameters
    private final ForkJoinPool customThreadPool;
    private static final int PARALLEL_THRESHOLD = 30;
    private static final int TASK_GRANULARITY = 7;

    // Memory monitoring - thread-safe
    private final long maxMemory;
    private final AtomicLong peakMemoryUsage = new AtomicLong(0);

    /**
     * Calculates a tighter upper bound by looking ahead one level
     */
    private static class LookAheadCalculator {
        private final ver6_1 algorithm;
        private final boolean useSampling;

        LookAheadCalculator(ver6_1 algorithm, boolean useSampling) {
            this.algorithm = algorithm;
            this.useSampling = useSampling;
        }

        /**
         * Calculate Look-Ahead Upper Bound for a given prefix and extensions
         * Returns the maximum potential utility from any single extension path
         */
        double calculateLookAheadBound(EnhancedUtilityList prefix,
                                      List<EnhancedUtilityList> extensions,
                                      double currentThreshold) {

            if (extensions.isEmpty()) {
                return prefix.getSumEU();
            }

            long startTime = System.currentTimeMillis();
            double maxBound = 0.0;
            int samplesToCheck = extensions.size();

            // Sampling strategy for large extension sets
            if (useSampling && extensions.size() > HybridConfig.MAX_LOOKAHEAD_SAMPLES) {
                samplesToCheck = HybridConfig.MAX_LOOKAHEAD_SAMPLES;
            }

            // Sort extensions by RTWU to check most promising first
            List<EnhancedUtilityList> sortedExtensions = new ArrayList<>(extensions);
            sortedExtensions.sort((a, b) -> Double.compare(b.rtwu, a.rtwu));

            for (int i = 0; i < Math.min(samplesToCheck, sortedExtensions.size()); i++) {
                EnhancedUtilityList extension = sortedExtensions.get(i);

                // Quick pre-check using RTWU
                if (extension.rtwu < currentThreshold - EPSILON) {
                    continue;
                }

                // Perform temporary join to get actual EU
                EnhancedUtilityList joined = algorithm.join(prefix, extension);
                algorithm.lookAheadJoinsPerformed.incrementAndGet();

                if (joined != null && !joined.elements.isEmpty()) {
                    // Calculate bound for this extension
                    double extensionBound = joined.getSumEU() + joined.getSumRemaining();

                    if (extensionBound > maxBound) {
                        maxBound = extensionBound;
                    }

                    // Early termination if we find a very promising branch
                    if (maxBound > currentThreshold * 1.5) {
                        break;
                    }
                }

                // Time limit to prevent excessive computation
                if (System.currentTimeMillis() - startTime > HybridConfig.MAX_LOOKAHEAD_TIME_MS) {
                    break;
                }
            }

            long elapsedTime = System.currentTimeMillis() - startTime;
            algorithm.lookAheadTimeMs.addAndGet(elapsedTime);

            return maxBound;
        }
    }

    private final LookAheadCalculator lookAheadCalculator;

    /**
     * Enhanced Transaction class with efficient storage and RTWU ordering
     */
    static class EnhancedTransaction {
        final int tid;
        final int[] items;
        final int[] quantities;
        final double[] logProbabilities;
        final double rtu;

        // Constructor with RTWU ordering
        EnhancedTransaction(int tid, Map<Integer, Integer> itemMap,
                        Map<Integer, Double> probMap, Map<Integer, Double> profits,
                        Map<Integer, Integer> itemToRank) {
            this.tid = tid;

            // Sort items by RTWU rank
            List<Integer> sortedItems = new ArrayList<>(itemMap.keySet());
            sortedItems.sort((a, b) -> {
                Integer rankA = itemToRank.get(a);
                Integer rankB = itemToRank.get(b);
                if (rankA == null && rankB == null) return 0;
                if (rankA == null) return 1;
                if (rankB == null) return -1;
                return rankA.compareTo(rankB);
            });

            // Convert to arrays for efficiency
            int size = sortedItems.size();
            this.items = new int[size];
            this.quantities = new int[size];
            this.logProbabilities = new double[size];

            int idx = 0;
            double rtu = 0;

            for (Integer item : sortedItems) {
                items[idx] = item;
                quantities[idx] = itemMap.get(item);

                // OPTIMIZED: Better handling of probability conversion
                double prob = probMap.getOrDefault(item, 0.0);

                // More careful threshold for "zero" probability
                if (prob < EPSILON) {
                    logProbabilities[idx] = LOG_EPSILON;
                } else if (prob > 1.0 - EPSILON) {
                    // Handle probability very close to 1
                    logProbabilities[idx] = 0.0; // log(1) = 0
                } else {
                    logProbabilities[idx] = Math.log(prob);
                }

                Double profit = profits.get(item);
                if (profit != null) {
                    double utility = profit * quantities[idx];
                    if (profit > 0) {
                        rtu += utility;
                    }
                }
                idx++;
            }

            this.rtu = rtu;
        }

        int getItemIndex(int item) {
            return Arrays.binarySearch(items, item);
        }

        double getItemLogProbability(int item) {
            int idx = getItemIndex(item);
            return idx >= 0 ? logProbabilities[idx] : LOG_EPSILON;
        }

        int getItemQuantity(int item) {
            int idx = getItemIndex(item);
            return idx >= 0 ? quantities[idx] : 0;
        }
    }

    /**
     * Optimized Utility-List with on-demand calculation of sumEU and sumRemaining
     */
    static class EnhancedUtilityList {
        static class Element {
            final int tid;
            final double utility;
            final double remaining;
            final double logProbability;

            Element(int tid, double utility, double remaining, double logProbability) {
                this.tid = tid;
                this.utility = utility;
                this.remaining = remaining;
                this.logProbability = logProbability;
            }
        }

        final Set<Integer> itemset;
        final List<Element> elements;
        final double existentialProbability;
        final double rtwu;

        // Lazy-computed values with caching
        private Double cachedSumEU = null;
        private Double cachedSumRemaining = null;

        EnhancedUtilityList(Set<Integer> itemset, List<Element> elements, double rtwu) {
            this.itemset = Collections.unmodifiableSet(new HashSet<>(itemset));
            this.elements = Collections.unmodifiableList(elements);
            this.rtwu = rtwu;

            // CHANGED: Use optimized calculation
            this.existentialProbability = calculateLogSpaceExistentialProbability();
        }

        // On-demand calculation with caching
        double getSumEU() {
            if (cachedSumEU == null) {
                double eu = 0;
                for (Element e : elements) {
                    double prob = Math.exp(e.logProbability);
                    eu += e.utility * prob;
                }
                cachedSumEU = eu;
            }
            return cachedSumEU;
        }

        // On-demand calculation with caching
        double getSumRemaining() {
            if (cachedSumRemaining == null) {
                double rem = 0;
                for (Element e : elements) {
                    double prob = Math.exp(e.logProbability);
                    rem += e.remaining * prob;
                }
                cachedSumRemaining = rem;
            }
            return cachedSumRemaining;
        }

        /**
         * OPTIMIZED: Calculate existential probability using true log-space computation
         * Avoids unnecessary exp/log conversions for significant performance improvement
         */
        private double calculateLogSpaceExistentialProbability() {
            if (elements.isEmpty()) return 0.0;

            double logComplement = 0.0;

            for (Element e : elements) {
                double logP = e.logProbability;

                // Early termination if probability extremely high (p ≈ 1)
                if (logP > -1e-15) {
                    return 1.0;
                }

                // Optimized log(1 - exp(logP)) computation
                double log1MinusP;

                if (logP < -10) {
                    // For very small p (p < e^-10 ≈ 0.000045):
                    // log(1 - p) ≈ -p (Taylor series first term)
                    // This avoids exp computation for tiny probabilities
                    log1MinusP = -Math.exp(logP);

                } else if (logP < -0.693) {  // -0.693 ≈ log(0.5)
                    // For small to medium p (0.000045 < p < 0.5):
                    // Use log1p for best numerical stability
                    // log1p(-exp(logP)) = log(1 + (-exp(logP))) = log(1 - exp(logP))
                    log1MinusP = Math.log1p(-Math.exp(logP));

                } else {
                    // For large p (p >= 0.5):
                    // Direct computation is more stable
                    // Note: We only do ONE exp operation here, not exp then log
                    double p = Math.exp(logP);

                    // Special handling for p very close to 1
                    if (p > 0.9999999999) {
                        return 1.0;
                    }

                    log1MinusP = Math.log(1.0 - p);
                }

                logComplement += log1MinusP;

                // Early termination if product becomes too small
                // exp(-30) ≈ 9.36e-14, effectively 0 for practical purposes
                if (logComplement < -30) {
                    return 1.0;
                }
            }

            // Handle extreme underflow
            if (logComplement < LOG_EPSILON) {  // LOG_EPSILON = -700
                return 1.0;
            }

            double complement = Math.exp(logComplement);
            return 1.0 - complement;
        }
    }

    /**
     * ForkJoinTask for parallel prefix mining with dynamic pruning
     */
    private class PrefixMiningTask extends RecursiveAction {
        private final List<Integer> sortedItems;
        private final Map<Integer, EnhancedUtilityList> singleItemLists;
        private final int start;
        private final int end;

        PrefixMiningTask(List<Integer> sortedItems,
                         Map<Integer, EnhancedUtilityList> singleItemLists,
                         int start, int end) {
            this.sortedItems = sortedItems;
            this.singleItemLists = singleItemLists;
            this.start = start;
            this.end = end;
        }

        @Override
        protected void compute() {
            int size = end - start;

            if (size <= TASK_GRANULARITY) {
                for (int i = start; i < end; i++) {
                    processPrefix(i);
                }
            } else {
                int mid = start + (size / 2);
                PrefixMiningTask leftTask = new PrefixMiningTask(sortedItems, singleItemLists, start, mid);
                PrefixMiningTask rightTask = new PrefixMiningTask(sortedItems, singleItemLists, mid, end);

                leftTask.fork();
                rightTask.compute();
                leftTask.join();
            }
        }

        private void processPrefix(int index) {
            Integer item = sortedItems.get(index);
            EnhancedUtilityList ul = singleItemLists.get(item);

            if (ul == null) return;

            // Dynamic branch pruning based on RTWU
            double currentThreshold = topKManager.getThreshold();
            if (itemRTWU.get(item) < currentThreshold - EPSILON) {
                branchPruned.incrementAndGet();
                return;
            }

            // Get extensions with RTWU-based filtering
            List<EnhancedUtilityList> extensions = new ArrayList<>();
            for (int j = index + 1; j < sortedItems.size(); j++) {
                Integer extItem = sortedItems.get(j);
                EnhancedUtilityList extUL = singleItemLists.get(extItem);

                if (extUL == null) continue;

                // Skip if extension's RTWU is too low
                if (itemRTWU.get(extItem) < currentThreshold - EPSILON) {
                    rtwuPruned.incrementAndGet();
                    continue;
                }

                extensions.add(extUL);
            }

            // Mine with this prefix
            if (!extensions.isEmpty()) {
                searchEnhanced(ul, extensions, singleItemLists);
            }

            // Monitor memory usage periodically
            if (index % 10 == 0) {
                long usedMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                peakMemoryUsage.updateAndGet(peak -> Math.max(peak, usedMemory));
            }
        }
    }

    /**
     * ForkJoinTask for parallel extension search with enhanced pruning
     */
    private class ExtensionSearchTask extends RecursiveAction {
        private final EnhancedUtilityList prefix;
        private final List<EnhancedUtilityList> extensions;
        private final Map<Integer, EnhancedUtilityList> singleItemLists;
        private final int start;
        private final int end;

        ExtensionSearchTask(EnhancedUtilityList prefix,
                        List<EnhancedUtilityList> extensions,
                        Map<Integer, EnhancedUtilityList> singleItemLists,
                        int start, int end) {
            this.prefix = prefix;
            this.extensions = extensions;
            this.singleItemLists = singleItemLists;
            this.start = start;
            this.end = end;
        }

        @Override
        protected void compute() {
            int size = end - start;

            // MINIMUM RTWU SAFETY BRANCH PRUNING for parallel tasks
            if (size > 1) {
                double currentThreshold = topKManager.getThreshold();

                double minRTWU = Double.MAX_VALUE;
                for (int i = start; i < end; i++) {
                    EnhancedUtilityList ext = extensions.get(i);
                    if (ext.rtwu < minRTWU) {
                        minRTWU = ext.rtwu;
                    }
                }

                if (minRTWU < currentThreshold - EPSILON) {
                    bulkBranchPruned.incrementAndGet();
                    candidatesPruned.addAndGet(size);
                    return;
                }
            }

            if (size <= TASK_GRANULARITY) {
                for (int i = start; i < end; i++) {
                    processExtension(i);
                }
            } else {
                int mid = start + (size / 2);
                ExtensionSearchTask leftTask = new ExtensionSearchTask(
                    prefix, extensions, singleItemLists, start, mid
                );
                ExtensionSearchTask rightTask = new ExtensionSearchTask(
                    prefix, extensions, singleItemLists, mid, end
                );

                invokeAll(leftTask, rightTask);
            }
        }

        private void processExtension(int index) {
            EnhancedUtilityList extension = extensions.get(index);

            double currentThreshold = topKManager.getThreshold();
            if (extension.rtwu < currentThreshold - EPSILON) {
                rtwuPruned.incrementAndGet();
                candidatesPruned.incrementAndGet();
                return;
            }

            EnhancedUtilityList joined = ver6_1.this.join(prefix, extension);

            if (joined == null || joined.elements.isEmpty()) {
                return;
            }

            utilityListsCreated.incrementAndGet();
            candidatesGenerated.incrementAndGet();

            double threshold = topKManager.getThreshold();

            // Pruning Strategy 1: Existential probability
            if (joined.existentialProbability < minPro - EPSILON) {
                epPruned.incrementAndGet();
                candidatesPruned.incrementAndGet();
                return;
            }

            // Pruning Strategy 2: EU + remaining (calculated on-demand)
            double sumEU = joined.getSumEU();
            double sumRemaining = joined.getSumRemaining();

            if (sumEU + sumRemaining < threshold - EPSILON) {
                euPruned.incrementAndGet();
                candidatesPruned.incrementAndGet();
                return;
            }

            // Update top-k if qualified
            if (sumEU >= threshold - EPSILON &&
                joined.existentialProbability >= minPro - EPSILON) {
                topKManager.tryAdd(joined.itemset, sumEU, joined.existentialProbability);
            }

            // Recursive search with RTWU-filtered extensions
            if (index < extensions.size() - 1) {
                List<EnhancedUtilityList> newExtensions = new ArrayList<>();
                double currentThresholdForFilter = topKManager.getThreshold();

                for (int j = index + 1; j < extensions.size(); j++) {
                    EnhancedUtilityList ext = extensions.get(j);

                    if (ext.rtwu >= currentThresholdForFilter - EPSILON) {
                        newExtensions.add(ext);
                    } else {
                        rtwuPruned.incrementAndGet();
                    }
                }

                if (!newExtensions.isEmpty()) {
                    searchEnhanced(joined, newExtensions, singleItemLists);
                }
            }
        }
    }

    // ==================== MAIN ALGORITHM ====================

    public ver6_1(Map<Integer, Double> itemProfits, int k, double minPro) {
        this.itemProfits = Collections.unmodifiableMap(new HashMap<>(itemProfits));
        this.k = k;
        this.minPro = minPro;
        this.topKManager = new TopKManager(k);

        int numThreads = Runtime.getRuntime().availableProcessors();
        this.customThreadPool = new ForkJoinPool(numThreads);

        this.maxMemory = Runtime.getRuntime().maxMemory();

        // Initialize Look-Ahead calculator
        this.lookAheadCalculator = new LookAheadCalculator(this, true);
    }

    /**
     * Main mining method with optimized storage
     */
    public List<Itemset> mine(List<Transaction> rawDatabase) {
        Instant start = Instant.now();

        System.out.println("=== PTK-HUIM-U± v6.1 with Hybrid Look-Ahead ===");
        System.out.println("NEW: Strategic Look-Ahead Upper Bound at critical points");
        System.out.println("Look-Ahead Config:");
        System.out.println("  - Max prefix size: " + HybridConfig.MAX_PREFIX_SIZE);
        System.out.println("  - Min extensions: " + HybridConfig.MIN_EXTENSIONS_FOR_LOOKAHEAD);
        System.out.println("  - Max samples: " + HybridConfig.MAX_LOOKAHEAD_SAMPLES);
        System.out.println("Database size: " + rawDatabase.size());
        System.out.println("Number of items: " + itemProfits.size());
        System.out.println("K: " + k + ", MinPro: " + minPro);
        System.out.println("Available memory: " + (maxMemory / 1024 / 1024) + " MB");
        System.out.println("Thread pool size: " + customThreadPool.getParallelism());

        // Phase 1: Initialization (unchanged)
        System.out.println("\nPhase 1: Single-pass initialization...");
        Map<Integer, EnhancedUtilityList> singleItemLists = optimizedInitialization(rawDatabase);

        List<Integer> sortedItems = getSortedItemsByRank(singleItemLists.keySet());
        System.out.println("Items after filtering: " + sortedItems.size());

        // Process single items (unchanged)
        for (Integer item : sortedItems) {
            EnhancedUtilityList ul = singleItemLists.get(item);
            if (ul != null) {
                double sumEU = ul.getSumEU();
                if (sumEU >= topKManager.getThreshold() - EPSILON &&
                    ul.existentialProbability >= minPro - EPSILON) {
                    topKManager.tryAdd(ul.itemset, sumEU, ul.existentialProbability);
                }
            }
        }

        System.out.println("\nPhase 2: Hybrid parallel mining...");

        // Phase 2: Mining (with hybrid approach)
        if (sortedItems.size() >= PARALLEL_THRESHOLD) {
            System.out.println("Using parallel processing with hybrid pruning");
            try {
                PrefixMiningTask rootTask = new PrefixMiningTask(
                    sortedItems, singleItemLists, 0, sortedItems.size()
                );
                customThreadPool.invoke(rootTask);
            } catch (Exception e) {
                System.err.println("Error in parallel processing: " + e.getMessage());
                e.printStackTrace();
                sequentialMining(sortedItems, singleItemLists);
            }
        } else {
            System.out.println("Using sequential processing with hybrid pruning");
            sequentialMining(sortedItems, singleItemLists);
        }

        List<Itemset> results = topKManager.getTopK();

        Instant end = Instant.now();

        // Enhanced statistics report
        System.out.println("\n=== Mining Complete ===");
        System.out.println("Execution time: " + Duration.between(start, end).toMillis() + " ms");
        System.out.println("Candidates generated: " + candidatesGenerated.get());
        System.out.println("Utility lists created: " + utilityListsCreated.get());

        System.out.println("\n=== Standard Pruning Statistics ===");
        System.out.println("  - RTWU pruned: " + rtwuPruned.get());
        System.out.println("  - Branches pruned: " + branchPruned.get());
        System.out.println("  - Bulk branches pruned: " + bulkBranchPruned.get());
        System.out.println("  - EU+remaining pruned: " + euPruned.get());
        System.out.println("  - Existential probability pruned: " + epPruned.get());

        System.out.println("\n=== Look-Ahead Statistics ===");
        System.out.println("  - Look-Ahead attempts: " + lookAheadAttempts.get());
        System.out.println("  - Look-Ahead successful prunes: " + lookAheadPruned.get());

        if (lookAheadAttempts.get() > 0) {
            double successRate = (lookAheadPruned.get() * 100.0) / lookAheadAttempts.get();
            System.out.println("  - Success rate: " + String.format("%.1f%%", successRate));
        }

        System.out.println("  - Temporary joins performed: " + lookAheadJoinsPerformed.get());
        System.out.println("  - Total Look-Ahead time: " + lookAheadTimeMs.get() + " ms");

        if (lookAheadAttempts.get() > 0) {
            long avgTime = lookAheadTimeMs.get() / lookAheadAttempts.get();
            System.out.println("  - Average time per Look-Ahead: " + avgTime + " ms");
        }

        System.out.println("\n=== Overall Statistics ===");
        System.out.println("Total pruned: " + candidatesPruned.get());
        System.out.println("Peak memory usage: " + (peakMemoryUsage.get() / 1024 / 1024) + " MB");
        System.out.println("Final threshold: " + String.format("%.4f", topKManager.getThreshold()));
        System.out.println("Top-K found: " + results.size());

        // Shutdown thread pool (unchanged)
        customThreadPool.shutdown();
        try {
            if (!customThreadPool.awaitTermination(60, TimeUnit.SECONDS)) {
                System.err.println("Thread pool didn't terminate in 60 seconds, forcing shutdown");
                customThreadPool.shutdownNow();

                if (!customThreadPool.awaitTermination(60, TimeUnit.SECONDS)) {
                    System.err.println("Thread pool didn't terminate after forced shutdown");
                }
            }
        } catch (InterruptedException e) {
            System.err.println("Interrupted during shutdown");
            customThreadPool.shutdownNow();
            Thread.currentThread().interrupt();
        }

        return results;
    }

    /**
     * Sequential mining with optimized storage access
     */
    private void sequentialMining(List<Integer> sortedItems,
                              Map<Integer, EnhancedUtilityList> singleItemLists) {
        for (int i = 0; i < sortedItems.size(); i++) {
            Integer item = sortedItems.get(i);
            EnhancedUtilityList ul = singleItemLists.get(item);

            if (ul == null) continue;

            double currentThreshold = topKManager.getThreshold();
            if (itemRTWU.get(item) < currentThreshold - EPSILON) {
                branchPruned.incrementAndGet();
                continue;
            }

            List<EnhancedUtilityList> extensions = new ArrayList<>();
            for (int j = i + 1; j < sortedItems.size(); j++) {
                Integer extItem = sortedItems.get(j);
                EnhancedUtilityList extUL = singleItemLists.get(extItem);

                if (extUL == null) continue;

                if (itemRTWU.get(extItem) < currentThreshold - EPSILON) {
                    rtwuPruned.incrementAndGet();
                    continue;
                }

                extensions.add(extUL);
            }

            if (!extensions.isEmpty()) {
                searchEnhanced(ul, extensions, singleItemLists);
            }

            if (i % 10 == 0) {
                long usedMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                peakMemoryUsage.updateAndGet(peak -> Math.max(peak, usedMemory));
                System.out.println("Progress: " + (i + 1) + "/" + sortedItems.size() +
                                " items processed. Memory used: " + (usedMemory / 1024 / 1024) + " MB");
            }
        }
    }

    /**
     * Optimized single-pass initialization
     */
    private Map<Integer, EnhancedUtilityList> optimizedInitialization(List<Transaction> rawDatabase) {
        // PASS 1: Calculate RTWU for each item
        System.out.println("Pass 1: Computing RTWU values...");
        this.itemRTWU = new HashMap<>();

        for (Transaction rawTrans : rawDatabase) {
            double rtu = 0;
            for (Map.Entry<Integer, Integer> entry : rawTrans.items.entrySet()) {
                Integer item = entry.getKey();
                Integer quantity = entry.getValue();
                Double profit = itemProfits.get(item);
                if (profit != null && profit > 0) {
                    rtu += profit * quantity;
                }
            }

            for (Map.Entry<Integer, Integer> entry : rawTrans.items.entrySet()) {
                Integer item = entry.getKey();
                Double prob = rawTrans.probabilities.get(item);
                if (prob != null && prob > 0) {
                    itemRTWU.merge(item, rtu, Double::sum);
                }
            }
        }

        // Build global ordering based on RTWU
        System.out.println("Building global RTWU ordering...");
        this.itemToRank = new HashMap<>();

        List<Integer> rankedItems = itemRTWU.entrySet().stream()
            .sorted((a, b) -> {
                int cmp = Double.compare(a.getValue(), b.getValue());
                if (cmp != 0) return cmp;
                return a.getKey().compareTo(b.getKey());
            })
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());

        for (int i = 0; i < rankedItems.size(); i++) {
            itemToRank.put(rankedItems.get(i), i);
        }

        System.out.println("RTWU ordering established for " + itemToRank.size() + " items");

        // PASS 2: Build utility-lists directly (no intermediate storage)
        System.out.println("Pass 2: Building utility-lists directly...");

        // Temporary storage for building utility lists
        Map<Integer, List<EnhancedUtilityList.Element>> tempElements = new HashMap<>();

        for (Transaction rawTrans : rawDatabase) {
            EnhancedTransaction trans = new EnhancedTransaction(
                rawTrans.tid, rawTrans.items, rawTrans.probabilities, itemProfits, itemToRank
            );

            for (int i = 0; i < trans.items.length; i++) {
                int item = trans.items[i];

                if (!itemToRank.containsKey(item)) continue;

                double utility = itemProfits.getOrDefault(item, 0.0) * trans.quantities[i];
                double logProb = trans.logProbabilities[i];

                if (logProb > LOG_EPSILON) {
                    double remaining = 0;
                    for (int j = i + 1; j < trans.items.length; j++) {
                        Double profit = itemProfits.get(trans.items[j]);
                        if (profit != null && profit > 0) {
                            remaining += profit * trans.quantities[j];
                        }
                    }

                    tempElements.computeIfAbsent(item, k -> new ArrayList<>())
                        .add(new EnhancedUtilityList.Element(
                            trans.tid, utility, remaining, logProb
                        ));
                }
            }
        }

        // Create final utility lists with RTWU values
        Map<Integer, EnhancedUtilityList> singleItemLists = new HashMap<>();

        for (Map.Entry<Integer, List<EnhancedUtilityList.Element>> entry : tempElements.entrySet()) {
            Integer item = entry.getKey();
            List<EnhancedUtilityList.Element> elements = entry.getValue();

            if (!elements.isEmpty()) {
                Set<Integer> itemset = Collections.singleton(item);
                Double rtwu = itemRTWU.get(item);

                EnhancedUtilityList ul = new EnhancedUtilityList(itemset, elements, rtwu);

                if (ul.existentialProbability >= minPro - EPSILON) {
                    singleItemLists.put(item, ul);
                    utilityListsCreated.incrementAndGet();
                }
            }
        }

        return singleItemLists;
    }

    /**
     * Get items sorted by their rank
     */
    private List<Integer> getSortedItemsByRank(Set<Integer> items) {
        return items.stream()
            .sorted((a, b) -> {
                Integer rankA = itemToRank.get(a);
                Integer rankB = itemToRank.get(b);
                if (rankA == null && rankB == null) return 0;
                if (rankA == null) return 1;
                if (rankB == null) return -1;
                return rankA.compareTo(rankB);
            })
            .collect(Collectors.toList());
    }

    /**
     * Search with enhanced pruning strategies
     */
    private void searchEnhanced(EnhancedUtilityList prefix,
                               List<EnhancedUtilityList> extensions,
                               Map<Integer, EnhancedUtilityList> singleItemLists) {

        double currentThreshold = topKManager.getThreshold();

        // ========== HYBRID DECISION POINT ==========
        boolean shouldUseLookAhead = shouldApplyLookAhead(prefix, extensions, currentThreshold);

        if (shouldUseLookAhead) {
            lookAheadAttempts.incrementAndGet();

            // Calculate Look-Ahead upper bound
            double lookAheadBound = lookAheadCalculator.calculateLookAheadBound(
                prefix, extensions, currentThreshold
            );

            // Check if we can prune the entire branch
            if (lookAheadBound < currentThreshold - EPSILON) {
                lookAheadPruned.incrementAndGet();
                bulkBranchPruned.incrementAndGet();
                candidatesPruned.addAndGet(extensions.size());

                // Log significant pruning events
                if (extensions.size() > 100) {
                    System.out.println("Look-Ahead pruned " + extensions.size() +
                                     " extensions at prefix size " + prefix.itemset.size());
                }
                return;
            }
        }

        // ========== STANDARD PROCESSING (unchanged) ==========
        // Pre-check for bulk pruning using RTWU (existing optimization)
        if (extensions.size() > 1) {
            double minRTWU = Double.MAX_VALUE;
            for (EnhancedUtilityList ext : extensions) {
                if (ext.rtwu < minRTWU) {
                    minRTWU = ext.rtwu;
                }
            }

            if (minRTWU < currentThreshold - EPSILON) {
                bulkBranchPruned.incrementAndGet();
                candidatesPruned.addAndGet(extensions.size());
                return;
            }
        }

        // Determine parallel vs sequential
        if (extensions.size() >= PARALLEL_THRESHOLD && ForkJoinTask.inForkJoinPool()) {
            ExtensionSearchTask task = new ExtensionSearchTask(
                prefix, extensions, singleItemLists, 0, extensions.size()
            );
            task.invoke();
        } else {
            processExtensionsSequentially(prefix, extensions, singleItemLists);
        }
    }

    /**
     * Determines whether to apply Look-Ahead based on current conditions
     */
    private boolean shouldApplyLookAhead(EnhancedUtilityList prefix,
                                        List<EnhancedUtilityList> extensions,
                                        double currentThreshold) {

        if (!HybridConfig.ENABLE_LOOKAHEAD) {
            return false;
        }

        // Only apply at top levels of search tree
        if (prefix.itemset.size() > HybridConfig.MAX_PREFIX_SIZE) {
            return false;
        }

        return true;
    }

    /**
     * Estimate maximum possible utility from extensions
     */
    private double estimateMaxPossibleUtility(List<EnhancedUtilityList> extensions) {
        double maxRTWU = 0;
        for (EnhancedUtilityList ext : extensions) {
            if (ext.rtwu > maxRTWU) {
                maxRTWU = ext.rtwu;
            }
        }
        return maxRTWU;
    }

    /**
     * Process extensions sequentially (extracted for clarity)
     */
    private void processExtensionsSequentially(EnhancedUtilityList prefix,
                                              List<EnhancedUtilityList> extensions,
                                              Map<Integer, EnhancedUtilityList> singleItemLists) {

        List<EnhancedUtilityList> newExtensions = new ArrayList<>(extensions.size());

        for (int i = 0; i < extensions.size(); i++) {
            EnhancedUtilityList extension = extensions.get(i);

            double currentThreshold = topKManager.getThreshold();

            // Quick RTWU check
            if (extension.rtwu < currentThreshold - EPSILON) {
                rtwuPruned.incrementAndGet();
                candidatesPruned.incrementAndGet();
                continue;
            }

            // Join operation
            EnhancedUtilityList joined = join(prefix, extension);

            if (joined == null || joined.elements.isEmpty()) {
                continue;
            }

            utilityListsCreated.incrementAndGet();
            candidatesGenerated.incrementAndGet();

            double threshold = topKManager.getThreshold();

            // Existential probability check
            if (joined.existentialProbability < minPro - EPSILON) {
                epPruned.incrementAndGet();
                candidatesPruned.incrementAndGet();
                continue;
            }

            // Standard upper bound check
            double sumEU = joined.getSumEU();
            double sumRemaining = joined.getSumRemaining();

            if (sumEU + sumRemaining < threshold - EPSILON) {
                euPruned.incrementAndGet();
                candidatesPruned.incrementAndGet();
                continue;
            }

            // Update top-k if qualified
            if (sumEU >= threshold - EPSILON &&
                joined.existentialProbability >= minPro - EPSILON) {
                topKManager.tryAdd(joined.itemset, sumEU, joined.existentialProbability);
            }

            // Prepare extensions for recursive call
            if (i < extensions.size() - 1) {
                newExtensions.clear();
                double currentThresholdForFilter = topKManager.getThreshold();

                for (int j = i + 1; j < extensions.size(); j++) {
                    EnhancedUtilityList ext = extensions.get(j);
                    if (ext.rtwu >= currentThresholdForFilter - EPSILON) {
                        newExtensions.add(ext);
                    } else {
                        rtwuPruned.incrementAndGet();
                    }
                }

                if (!newExtensions.isEmpty()) {
                    searchEnhanced(joined, newExtensions, singleItemLists);
                }
            }
        }
    }

    /**
     * Join two utility-lists
     */
    private EnhancedUtilityList join(EnhancedUtilityList ul1, EnhancedUtilityList ul2) {
        // Pre-compute itemset union
        Set<Integer> newItemset = new HashSet<>(ul1.itemset);
        newItemset.addAll(ul2.itemset);

        // Calculate RTWU for the joined itemset
        double joinedRTWU = Math.min(ul1.rtwu, ul2.rtwu);

        // Pre-sized collection to prevent resizing
        int maxPossibleSize = Math.min(ul1.elements.size(), ul2.elements.size());
        List<EnhancedUtilityList.Element> joinedElements = new ArrayList<>(maxPossibleSize);

        // Standard merge
        int i = 0, j = 0;
        while (i < ul1.elements.size() && j < ul2.elements.size()) {
            EnhancedUtilityList.Element e1 = ul1.elements.get(i);
            EnhancedUtilityList.Element e2 = ul2.elements.get(j);

            if (e1.tid == e2.tid) {
                double newUtility = e1.utility + e2.utility;
                double newRemaining = Math.min(e1.remaining, e2.remaining);

                // OPTIMIZED: Better handling of joint probability in log space
                double newLogProbability = e1.logProbability + e2.logProbability;

                // More careful threshold checking
                // Only add if joint probability is meaningful
                if (newLogProbability > LOG_EPSILON + 1) { // Slightly above minimum threshold
                    joinedElements.add(new EnhancedUtilityList.Element(
                        e1.tid, newUtility, newRemaining, newLogProbability
                    ));
                }
                i++;
                j++;
            } else if (e1.tid < e2.tid) {
                i++;
            } else {
                j++;
            }
        }

        if (joinedElements.isEmpty()) {
            return null;
        }

        // Trim if significantly overallocated
        if (joinedElements.size() < maxPossibleSize / 2) {
            ((ArrayList<EnhancedUtilityList.Element>) joinedElements).trimToSize();
        }

        return new EnhancedUtilityList(newItemset, joinedElements, joinedRTWU);
    }

    /**
     * Optimized thread-safe top-K manager
     */
    private class TopKManager {
        private final int k;
        // Sử dụng ConcurrentSkipListSet để tự động sắp xếp và an toàn luồng.
        // Nó giữ cho các itemset luôn được sắp xếp theo EU giảm dần.
        private final ConcurrentSkipListSet<Itemset> topKSet;

        // Sử dụng ConcurrentHashMap để tra cứu và cập nhật O(1), loại bỏ vòng lặp O(k) tốn kém.
        // Key là tập hợp items, Value là đối tượng Itemset tương ứng.
        private final ConcurrentHashMap<Set<Integer>, Itemset> topKMap;

        // Sử dụng AtomicReference để đọc ngưỡng không cần khóa, giảm đáng kể tranh chấp.
        private final AtomicReference<Double> threshold;

        // Lock vẫn cần thiết, nhưng chỉ cho các hành động phức hợp siêu ngắn
        // (thêm/xóa/cập nhật ngưỡng cùng lúc).
        private final Object lock = new Object();

        TopKManager(int k) {
            this.k = k;
            // Comparator sắp xếp theo EU giảm dần. Nếu EU bằng nhau, sắp xếp theo hashCode
            // để đảm bảo thứ tự ổn định và tránh loại bỏ các itemset có cùng EU.
            this.topKSet = new ConcurrentSkipListSet<>((a, b) -> {
                int cmp = Double.compare(b.expectedUtility, a.expectedUtility);
                if (cmp != 0) return cmp;
                return Integer.compare(a.hashCode(), b.hashCode());
            });
            this.topKMap = new ConcurrentHashMap<>();
            this.threshold = new AtomicReference<>(0.0);
        }

        private volatile double cachedThreshold = 0.0;

        /**
         * Cố gắng thêm một itemset ứng viên vào top-K.
         * Đây là phương thức đã được tối ưu hóa cao để giảm thiểu lock contention.
         */
        boolean tryAdd(Set<Integer> items, double eu, double ep) {

            if (eu < cachedThreshold - EPSILON) {
                return false;
            }

            // === GIAI ĐOẠN 1: KIỂM TRA SƠ BỘ KHÔNG CẦN KHÓA ===
            // Nhanh chóng loại bỏ >99% các ứng viên yếu mà không gây ra bất kỳ sự tranh chấp nào.
            if (eu < threshold.get() - EPSILON) {
                return false;
            }

            // === GIAI ĐOẠN 2: KIỂM TRA TRÙNG LẶP KHÔNG CẦN KHÓA ===
            // Sử dụng O(1) get() từ HashMap thay vì O(k) duyệt vòng lặp.
            Itemset existingItemset = topKMap.get(items);
            if (existingItemset != null && existingItemset.expectedUtility >= eu - EPSILON) {
                // Đã có phiên bản tốt hơn (hoặc bằng) trong top-K, không cần làm gì cả.
                return false;
            }

            // === GIAI ĐOẠN 3: KHÓA VÙNG CRITICAL SECTION SIÊU NHỎ ===
            // Chỉ những ứng viên "nặng ký" nhất (đã vượt qua 2 vòng lọc) mới đi tới được đây.
            synchronized (lock) {
                // Mẫu "Double-checked locking": Kiểm tra lại điều kiện sau khi đã có khóa.
                // Điều này cực kỳ quan trọng vì threshold có thể đã bị một luồng khác thay đổi
                // trong khoảng thời gian luồng này chờ để lấy được khóa.
                if (topKSet.size() >= k && eu < threshold.get() - EPSILON) {
                    return false;
                }

                // Lấy lại phiên bản mới nhất từ map sau khi có khóa để đảm bảo tính nhất quán dữ liệu.
                Itemset currentVersionInMap = topKMap.get(items);
                if (currentVersionInMap != null) {
                    // Nếu một luồng khác đã thêm một phiên bản tốt hơn trong lúc ta chờ...
                    if (currentVersionInMap.expectedUtility >= eu - EPSILON) {
                        return false;
                    }
                    // ...nếu không, ta sẽ thay thế phiên bản cũ đó.
                    topKSet.remove(currentVersionInMap);
                }

                Itemset newItemset = new Itemset(items, eu, ep);
                topKSet.add(newItemset);
                topKMap.put(items, newItemset);

                // Nếu top-K bị đầy (vượt quá k), loại bỏ phần tử yếu nhất.
                if (topKSet.size() > k) {
                    Itemset removed = topKSet.pollLast(); // Lấy và xóa phần tử cuối cùng (yếu nhất).
                    if (removed != null) {
                        // Cập nhật map để phản ánh việc loại bỏ này.
                        topKMap.remove(removed.items);
                    }
                }

                // Cập nhật lại ngưỡng một cách an toàn bên trong khóa.
                if (topKSet.size() >= k) {
                    double newThreshold = topKSet.last().expectedUtility;
                    threshold.set(newThreshold);
                    cachedThreshold = newThreshold; // Update cache
                }
            }
            return true;
        }

        double getThreshold() {
            return cachedThreshold;
        }

        List<Itemset> getTopK() {
            // Không cần khóa vì chỉ đọc dữ liệu từ cấu trúc thread-safe.
            // Tạo một bản sao để đảm bảo an toàn nếu danh sách gốc bị thay đổi bởi luồng khác.
            return new ArrayList<>(topKSet);
        }
    }

    /**
     * Itemset result class
     */
    private static class Itemset {
        final Set<Integer> items;
        final double expectedUtility;
        final double probability;

        Itemset(Set<Integer> items, double eu, double p) {
            this.items = items;
            this.expectedUtility = eu;
            this.probability = p;
        }

        // === BẮT ĐẦU PHẦN CẬP NHẬT ===

        @Override
        public int hashCode() {
            // Sử dụng hashCode của tập hợp items, vì nó là định danh duy nhất của itemset.
            // Điều này rất quan trọng để HashMap hoạt động hiệu quả.
            return items.hashCode();
        }

        @Override
        public boolean equals(Object obj) {
            // Phương thức này xác định khi nào hai đối tượng Itemset được coi là "bằng nhau".
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            Itemset other = (Itemset) obj;
            // Hai Itemset được coi là bằng nhau nếu và chỉ nếu tập hợp items của chúng giống hệt nhau.
            return items.equals(other.items);
        }

        // === KẾT THÚC PHẦN CẬP NHẬT ===

        @Override
        public String toString() {
            return "Itemset{" +
                    "items=" + items +
                    ", eu=" + String.format("%.2f", expectedUtility) +
                    '}';
        }
    }

    // ==================== ORIGINAL TRANSACTION CLASS ====================

    static class Transaction {
        final int tid;
        final Map<Integer, Integer> items;
        final Map<Integer, Double> probabilities;

        Transaction(int tid, Map<Integer, Integer> items, Map<Integer, Double> probabilities) {
            this.tid = tid;
            this.items = items;
            this.probabilities = probabilities;
        }
    }

    // ==================== MAIN METHOD ====================

    public static void main(String[] args) throws IOException {
        if (args.length != 4) {
            System.err.println("Usage: PTKHUIMEnhanced <database_file> <profit_file> <k> <min_probability>");
            System.exit(1);
        }

        String dbFile = args[0];
        String profitFile = args[1];
        int k = Integer.parseInt(args[2]);
        double minPro = Double.parseDouble(args[3]);

        // Read input files
        Map<Integer, Double> profits = readProfitTable(profitFile);
        List<Transaction> database = readDatabase(dbFile);

        System.out.println("=== PTK-HUIM-U± Enhanced Version 3.5 ===");
        System.out.println("Key improvements:");
        System.out.println("1. Removed duplicate database storage");
        System.out.println("2. Consolidated item ordering structures");
        System.out.println("3. Simplified TopK management");
        System.out.println("4. Single-pass utility list creation");
        System.out.println("5. On-demand calculation for EU and remaining");
        System.out.println("6. Direct iteration without duplicate lists");
        System.out.println();

        // Run enhanced algorithm
        ver6_1 algorithm = new ver6_1(profits, k, minPro);
        List<Itemset> topK = algorithm.mine(database);

        // Display results
        System.out.println("\n=== Top-" + k + " PHUIs ===");
        int rank = 1;
        for (Itemset itemset : topK) {
            System.out.printf("%d. %s\n", rank++, itemset);
        }
    }

    /**
     * Read profit table from file
     */
    static Map<Integer, Double> readProfitTable(String filename) throws IOException {
        Map<Integer, Double> profits = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.trim().split("\\s+");
                if (parts.length == 2) {
                    profits.put(Integer.parseInt(parts[0]), Double.parseDouble(parts[1]));
                }
            }
        }
        return profits;
    }

    /**
     * Read database from file
     */
    static List<Transaction> readDatabase(String filename) throws IOException {
        List<Transaction> database = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            int tid = 1;
            while ((line = br.readLine()) != null) {
                Map<Integer, Integer> items = new HashMap<>();
                Map<Integer, Double> probabilities = new HashMap<>();

                String[] entries = line.trim().split("\\s+");
                for (String entry : entries) {
                    String[] parts = entry.split(":");
                    if (parts.length == 3) {
                        int item = Integer.parseInt(parts[0]);
                        int quantity = Integer.parseInt(parts[1]);
                        double prob = Double.parseDouble(parts[2]);

                        items.put(item, quantity);
                        probabilities.put(item, prob);
                    }
                }

                if (!items.isEmpty()) {
                    database.add(new Transaction(tid++, items, probabilities));
                }
            }
        }
        return database;
    }
}