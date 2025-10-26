import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.stream.*;

/**
 * PTK-HUIM-U±: Parallel Top-K High-Utility Itemset Mining
 * from Uncertain Databases with Positive and Negative Utilities
 *
 * VARIANT 1: Weighted Balanced Split
 * Uses PTWU-based workload estimation for balanced task splitting
 */
public class variant1 {

    // ============================================================================
    // DATA STRUCTURES
    // ============================================================================

    /**
     * Transaction representation with item quantities and occurrence probabilities
     */
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

    /**
     * Itemset result with expected utility and existential probability
     */
    public static class Itemset {
        final Set<Integer> items;
        final double expectedUtility;
        final double probability;

        Itemset(Set<Integer> items, double eu, double p) {
            this.items = items;
            this.expectedUtility = eu;
            this.probability = p;
        }

        @Override
        public int hashCode() {
            return items.hashCode();
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            Itemset other = (Itemset) obj;
            return items.equals(other.items);
        }

        @Override
        public String toString() {
            return "Itemset{" +
                   "items=" + items +
                   ", eu=" + String.format(java.util.Locale.US, "%.4f", expectedUtility) +
                   '}';
        }
    }

    /**
     * Preprocessed item data for efficient suffix sum computation
     */
    private static class ItemData {
        final int item;
        final int quantity;
        final double profit;
        final double utility;
        final double logProb;

        ItemData(int item, int quantity, double profit, double logProb) {
            this.item = item;
            this.quantity = quantity;
            this.profit = profit;
            this.utility = profit * quantity;
            this.logProb = logProb;
        }
    }

    // ============================================================================
    // UPU-LIST (Utility-Projected Utility List with pre-computed aggregates)
    // ============================================================================

    /**
     * UPU-List (Utility-Projected Utility List) with array-based storage for memory efficiency
     * All aggregates (sumEU, sumRemaining, existentialProbability) pre-computed during construction
     */
    static class UPUList {

        /**
         * Temporary element structure used only during construction
         */
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
        final int[] tids;
        final double[] utilities;
        final double[] remainings;
        final double[] logProbabilities;
        final int size;
        final double ptwu;

        private final double sumEU;
        private final double sumRemaining;
        private final double existentialProbability;

        /**
         * Constructs utility list with array storage and single-pass aggregate computation
         * Significantly reduces memory allocations compared to list-based storage
         */
        UPUList(Set<Integer> itemset, List<Element> elements, double ptwu) {
            this.itemset = itemset;
            this.ptwu = ptwu;
            this.size = elements.size();

            this.tids = new int[size];
            this.utilities = new double[size];
            this.remainings = new double[size];
            this.logProbabilities = new double[size];

            // Single-pass: copy data and compute aggregates simultaneously
            double tempSumEU = 0.0;
            double tempSumRemaining = 0.0;
            double logComplement = 0.0;

            for (int i = 0; i < size; i++) {
                Element e = elements.get(i);
                tids[i] = e.tid;
                utilities[i] = e.utility;
                remainings[i] = e.remaining;
                logProbabilities[i] = e.logProbability;

                double prob = Math.exp(e.logProbability);
                tempSumEU += e.utility * prob;
                tempSumRemaining += e.remaining * prob;

                // Compute existential probability using log-space to prevent underflow
                if (e.logProbability > Math.log(1.0 - EPSILON)) {
                    logComplement = LOG_EPSILON;
                } else if (logComplement > LOG_EPSILON) {
                    double log1MinusP = prob < 0.5 ?
                        Math.log1p(-prob) :
                        Math.log(1.0 - prob);
                    logComplement += log1MinusP;
                    if (logComplement < LOG_EPSILON) {
                        logComplement = LOG_EPSILON;
                    }
                }
            }

            this.sumEU = tempSumEU;
            this.sumRemaining = tempSumRemaining;
            this.existentialProbability = logComplement < LOG_EPSILON ?
                1.0 : 1.0 - Math.exp(logComplement);
        }

        double getSumEU() {
            return sumEU;
        }

        double getSumRemaining() {
            return sumRemaining;
        }

        int getSize() {
            return size;
        }

        boolean isEmpty() {
            return size == 0;
        }
    }

    // ============================================================================
    // LOCK-FREE TOP-K MANAGER (CAS-based for thread safety)
    // ============================================================================

    /**
     * Thread-safe top-k manager using Compare-And-Swap operations
     * Maintains k highest-utility itemsets without locks for maximum parallelism
     */
    private class TopKManager {
        private final int k;
        private final AtomicReferenceArray<Itemset> topKArray;
        private final AtomicInteger size = new AtomicInteger(0);
        private final AtomicReference<Double> threshold = new AtomicReference<>(0.0);

        TopKManager(int k) {
            this.k = k;
            this.topKArray = new AtomicReferenceArray<>(k);
        }

        /**
         * Attempts to add itemset to top-k using lock-free CAS operations
         * Handles concurrent updates through retry mechanism with backoff
         */
        boolean tryAdd(Set<Integer> items, double eu, double ep) {
            final int MAX_RETRIES = 100;

            for (int attempt = 0; attempt < MAX_RETRIES; attempt++) {
                // Try empty slots first
                for (int i = 0; i < k; i++) {
                    if (topKArray.compareAndSet(i, null, new Itemset(items, eu, ep))) {
                        size.incrementAndGet();
                        if (DEBUG_VERBOSE) {
                            System.err.printf("[DEBUG-VERBOSE] Added to top-k (empty slot): %s, EU=%.4f, P=%.6f\n",
                                items, eu, ep);
                        }
                        updateThreshold();
                        return true;
                    }
                }

                // Check for duplicates and potential replacements
                for (int i = 0; i < k; i++) {
                    Itemset existing = topKArray.get(i);
                    if (existing != null && existing.items.equals(items)) {
                        if (eu > existing.expectedUtility + EPSILON) {
                            if (topKArray.compareAndSet(i, existing, new Itemset(items, eu, ep))) {
                                if (DEBUG_VERBOSE) {
                                    System.err.printf("[DEBUG-VERBOSE] Updated in top-k: %s, EU: %.4f -> %.4f\n",
                                        items, existing.expectedUtility, eu);
                                }
                                updateThreshold();
                                return true;
                            } else {
                                if (attempt % 10 == 0) Thread.yield();
                                break;
                            }
                        }
                        return false;
                    }
                }

                // Replace weakest itemset if array is full
                if (size.get() >= k) {
                    int weakestIndex = findWeakestIndex();
                    if (weakestIndex != -1) {
                        Itemset weakest = topKArray.get(weakestIndex);
                        if (weakest != null && eu > weakest.expectedUtility + EPSILON) {
                            if (topKArray.compareAndSet(weakestIndex, weakest, new Itemset(items, eu, ep))) {
                                if (DEBUG_VERBOSE) {
                                    System.err.printf("[DEBUG-VERBOSE] Replaced in top-k: %s (EU=%.4f) -> %s (EU=%.4f)\n",
                                        weakest.items, weakest.expectedUtility, items, eu);
                                }
                                updateThreshold();
                                return true;
                            } else {
                                if (attempt % 10 == 0) Thread.yield();
                                continue;
                            }
                        }
                    }
                }

                if (eu < getThreshold() - EPSILON) {
                    return false;
                }
            }

            return false;
        }

        private int findWeakestIndex() {
            double minEU = Double.MAX_VALUE;
            int minIndex = -1;

            for (int i = 0; i < k; i++) {
                Itemset item = topKArray.get(i);
                if (item != null && item.expectedUtility < minEU) {
                    minEU = item.expectedUtility;
                    minIndex = i;
                }
            }
            return minIndex;
        }

        /**
         * Updates threshold based on current minimum EU in top-k
         * Uses atomic snapshot to handle concurrent modifications
         */
        private void updateThreshold() {
            double oldThreshold = threshold.get();

            Itemset[] snapshot = new Itemset[k];
            for (int i = 0; i < k; i++) {
                snapshot[i] = topKArray.get(i);
            }

            double minEU = Double.MAX_VALUE;
            int count = 0;

            for (Itemset item : snapshot) {
                if (item != null) {
                    count++;
                    if (item.expectedUtility < minEU) {
                        minEU = item.expectedUtility;
                    }
                }
            }

            if (count >= k) {
                threshold.set(minEU);
                if (DEBUG_VERBOSE && Math.abs(minEU - oldThreshold) > EPSILON) {
                    System.err.printf("[DEBUG-VERBOSE] Threshold updated: %.4f -> %.4f\n",
                        oldThreshold, minEU);
                }
            }
        }

        double getThreshold() {
            return threshold.get();
        }

        List<Itemset> getTopK() {
            List<Itemset> result = new ArrayList<>();
            for (int i = 0; i < k; i++) {
                Itemset item = topKArray.get(i);
                if (item != null) {
                    result.add(item);
                }
            }
            result.sort((a, b) -> Double.compare(b.expectedUtility, a.expectedUtility));
            return result;
        }
    }

    // ============================================================================
    // FORKJOIN TASKS FOR PARALLEL PROCESSING (WEIGHTED BALANCED SPLIT)
    // ============================================================================

    /**
     * Parallel task for processing single-item prefixes
     * VARIANT 1: Uses PTWU-weighted balanced splitting instead of naive midpoint
     */
    private class PrefixMiningTask extends RecursiveAction {
        private final List<Integer> sortedItems;
        private final Map<Integer, UPUList> singleItemLists;
        private final int start, end;

        PrefixMiningTask(List<Integer> sortedItems,
                        Map<Integer, UPUList> singleItemLists,
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
                // WEIGHTED BALANCED SPLIT: Use PTWU as workload estimator
                int mid = findBalancedSplitForItems(start, end);
                PrefixMiningTask left = new PrefixMiningTask(sortedItems, singleItemLists, start, mid);
                PrefixMiningTask right = new PrefixMiningTask(sortedItems, singleItemLists, mid, end);

                left.fork();
                right.compute();
                left.join();
            }
        }

        /**
         * Finds balanced split point based on PTWU weights
         * Returns split index where accumulated weight ≈ totalWeight / 2
         * ENSURES: start < result < end (strictly between)
         */
        private int findBalancedSplitForItems(int start, int end) {
            // Step 1: Calculate total weight for range [start, end)
            double totalWeight = 0.0;
            for (int i = start; i < end; i++) {
                Integer item = sortedItems.get(i);
                totalWeight += itemPTWU.getOrDefault(item, 0.0);
            }

            // Step 2: Find split point where accumulated weight ≈ totalWeight / 2
            double targetWeight = totalWeight / 2.0;
            double accumulatedWeight = 0.0;

            for (int i = start; i < end - 1; i++) {  // Stop before last element
                Integer item = sortedItems.get(i);
                accumulatedWeight += itemPTWU.getOrDefault(item, 0.0);

                if (accumulatedWeight >= targetWeight) {
                    // Return next index, ensuring it's < end
                    int mid = i + 1;
                    // Safety: ensure mid is strictly between start and end
                    if (mid >= end) mid = end - 1;
                    if (mid <= start) mid = start + 1;
                    return mid;
                }
            }

            // Fallback to midpoint (always strictly between start and end)
            return start + (end - start) / 2;
        }

        private void processPrefix(int index) {
            Integer item = sortedItems.get(index);
            UPUList ul = singleItemLists.get(item);

            if (ul == null) return;

            double currentThreshold = topKManager.getThreshold();
            if (itemPTWU.get(item) < currentThreshold - EPSILON) {
                return;
            }

            // Collect valid extensions
            List<UPUList> extensions = new ArrayList<>();
            for (int j = index + 1; j < sortedItems.size(); j++) {
                Integer extItem = sortedItems.get(j);
                UPUList extUL = singleItemLists.get(extItem);

                if (extUL != null && itemPTWU.get(extItem) >= currentThreshold - EPSILON) {
                    extensions.add(extUL);
                }
            }

            if (!extensions.isEmpty()) {
                searchEnhanced(ul, extensions, singleItemLists);
            }
        }
    }

    /**
     * Parallel task for exploring itemset extensions
     * VARIANT 1: Uses PTWU-weighted balanced splitting instead of naive midpoint
     */
    private class ExtensionSearchTask extends RecursiveAction {
        private final UPUList prefix;
        private final List<UPUList> extensions;
        private final Map<Integer, UPUList> singleItemLists;
        private final int start, end;

        ExtensionSearchTask(UPUList prefix,
                           List<UPUList> extensions,
                           Map<Integer, UPUList> singleItemLists,
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

            if (size <= TASK_GRANULARITY) {
                for (int i = start; i < end; i++) {
                    processExtension(i);
                }
            } else {
                // WEIGHTED BALANCED SPLIT: Use PTWU as workload estimator
                int mid = findBalancedSplitForExtensions(start, end);
                ExtensionSearchTask left = new ExtensionSearchTask(
                    prefix, extensions, singleItemLists, start, mid
                );
                ExtensionSearchTask right = new ExtensionSearchTask(
                    prefix, extensions, singleItemLists, mid, end
                );

                invokeAll(left, right);
            }
        }

        /**
         * Finds balanced split point based on PTWU weights of extensions
         * Returns split index where accumulated weight ≈ totalWeight / 2
         * ENSURES: start < result < end (strictly between)
         */
        private int findBalancedSplitForExtensions(int start, int end) {
            // Step 1: Calculate total weight for range [start, end)
            double totalWeight = 0.0;
            for (int i = start; i < end; i++) {
                UPUList extUL = extensions.get(i);
                totalWeight += extUL.ptwu;
            }

            // Step 2: Find split point where accumulated weight ≈ totalWeight / 2
            double targetWeight = totalWeight / 2.0;
            double accumulatedWeight = 0.0;

            for (int i = start; i < end - 1; i++) {  // Stop before last element
                UPUList extUL = extensions.get(i);
                accumulatedWeight += extUL.ptwu;

                if (accumulatedWeight >= targetWeight) {
                    // Return next index, ensuring it's < end
                    int mid = i + 1;
                    // Safety: ensure mid is strictly between start and end
                    if (mid >= end) mid = end - 1;
                    if (mid <= start) mid = start + 1;
                    return mid;
                }
            }

            // Fallback to midpoint (always strictly between start and end)
            return start + (end - start) / 2;
        }

        private void processExtension(int index) {
            UPUList extension = extensions.get(index);

            double currentThreshold = topKManager.getThreshold();

            if (extension.ptwu < currentThreshold - EPSILON) {
                return;
            }

            UPUList joined = variant1.this.join(prefix, extension);

            if (joined == null || joined.isEmpty()) {
                return;
            }

            if (DEBUG) candidatesGenerated.incrementAndGet();

            double sumEU = joined.getSumEU();
            double sumRemaining = joined.getSumRemaining();

            if (DEBUG_VERBOSE) {
                System.err.printf("[DEBUG-VERBOSE] Evaluating candidate: %s\n", joined.itemset);
                System.err.printf("[DEBUG-VERBOSE]   EU=%.4f, Remaining=%.4f, Probability=%.6f, Threshold=%.4f\n",
                    sumEU, sumRemaining, joined.existentialProbability, currentThreshold);
            }

            // Pruning: EU + remaining utility upper bound
            if (sumEU + sumRemaining < currentThreshold - EPSILON) {
                if (DEBUG) prunedByEU.incrementAndGet();
                if (DEBUG_VERBOSE) {
                    System.err.printf("[DEBUG-VERBOSE]   PRUNED by EU+Remaining: %.4f < %.4f\n",
                        sumEU + sumRemaining, currentThreshold);
                }
                return;
            }

            // Pruning: existential probability threshold
            if (joined.existentialProbability < minPro - EPSILON) {
                if (DEBUG) prunedByProbability.incrementAndGet();
                if (DEBUG_VERBOSE) {
                    System.err.printf("[DEBUG-VERBOSE]   PRUNED by Probability: %.6f < %.6f\n",
                        joined.existentialProbability, minPro);
                }
                return;
            }

            if (DEBUG_VERBOSE) {
                System.err.println("[DEBUG-VERBOSE]   Passed all pruning checks");
            }

            // Add to top-k if qualified
            if (sumEU >= currentThreshold - EPSILON &&
                joined.existentialProbability >= minPro - EPSILON) {
                topKManager.tryAdd(joined.itemset, sumEU, joined.existentialProbability);
            }

            // Recursive exploration
            if (index < extensions.size() - 1) {
                List<UPUList> newExtensions = new ArrayList<>();
                double thresholdForRecursion = topKManager.getThreshold();

                for (int j = index + 1; j < extensions.size(); j++) {
                    UPUList ext = extensions.get(j);
                    if (ext.ptwu >= thresholdForRecursion - EPSILON) {
                        newExtensions.add(ext);
                    }
                }

                if (!newExtensions.isEmpty()) {
                    searchEnhanced(joined, newExtensions, singleItemLists);
                }
            }
        }
    }

    // ============================================================================
    // ALGORITHM CONFIGURATION
    // ============================================================================

    private final Map<Integer, Double> itemProfits;
    private final int k;
    private final double minPro;
    private static final double EPSILON = 1e-10;
    private static final double LOG_EPSILON = -700;
    private static final int TASK_GRANULARITY = 7;

    private final TopKManager topKManager;
    private final ForkJoinPool threadPool;

    // Item ordering for consistent itemset generation
    private Map<Integer, Integer> itemToRank;
    private Map<Integer, Double> itemPTWU;

    // Debug mode and statistics tracking
    private static boolean DEBUG = false;
    private static boolean DEBUG_VERBOSE = false;
    private final AtomicLong candidatesGenerated = new AtomicLong(0);
    private final AtomicLong utilityListJoins = new AtomicLong(0);
    private final AtomicLong prunedByPTWU = new AtomicLong(0);
    private final AtomicLong prunedByEU = new AtomicLong(0);
    private final AtomicLong prunedByProbability = new AtomicLong(0);

    // ============================================================================
    // CONSTRUCTOR
    // ============================================================================

    public variant1(Map<Integer, Double> itemProfits, int k, double minPro, boolean debug, boolean debugVerbose) {
        this.itemProfits = Collections.unmodifiableMap(new HashMap<>(itemProfits));
        this.k = k;
        this.minPro = minPro;
        this.topKManager = new TopKManager(k);
        this.threadPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());
        DEBUG = debug || debugVerbose; // Enable basic debug if verbose is enabled
        DEBUG_VERBOSE = debugVerbose;
    }

    // ============================================================================
    // MAIN MINING ALGORITHM
    // ============================================================================

    /**
     * Mines top-k high-utility itemsets from uncertain database
     */
    public List<Itemset> mine(List<Transaction> rawDatabase) {
        if (DEBUG) {
            System.err.println("[DEBUG] ===== PTK-HUIM-U± Algorithm Started (VARIANT 1: Weighted Split) =====");
            System.err.println("[DEBUG] Task Splitting: PTWU-Weighted Balanced Split");
            System.err.printf("[DEBUG] Database: %d transactions\n", rawDatabase.size());
            System.err.printf("[DEBUG] Parameters: k=%d, minProbability=%.10f\n", k, minPro);
        }

        // Phase 1: Initialize single-item utility lists with PTWU-based ordering
        if (DEBUG) System.err.println("[DEBUG] Phase 1: Initialization...");
        Map<Integer, UPUList> singleItemLists = optimizedInitialization(rawDatabase);
        List<Integer> sortedItems = getSortedItemsByRank(singleItemLists.keySet());

        if (DEBUG) {
            System.err.printf("[DEBUG] Found %d unique items after PTWU filtering\n", singleItemLists.size());
        }

        // Add qualifying single items to top-k
        if (DEBUG) System.err.println("[DEBUG] Phase 2: Processing single-item candidates...");
        int singleItemAdded = 0;
        for (Integer item : sortedItems) {
            UPUList ul = singleItemLists.get(item);
            if (ul != null) {
                double sumEU = ul.getSumEU();
                if (sumEU >= topKManager.getThreshold() - EPSILON &&
                    ul.existentialProbability >= minPro - EPSILON) {
                    topKManager.tryAdd(ul.itemset, sumEU, ul.existentialProbability);
                    if (DEBUG) singleItemAdded++;
                }
            }
        }
        if (DEBUG) {
            System.err.printf("[DEBUG] Added %d single-item candidates to top-k\n", singleItemAdded);
            System.err.printf("[DEBUG] Initial threshold: %.4f\n", topKManager.getThreshold());
        }

        // Phase 2: Parallel mining of multi-item patterns
        if (DEBUG) {
            System.err.println("[DEBUG] Phase 3: Parallel mining of multi-item patterns...");
            System.err.printf("[DEBUG] Using %d threads\n", threadPool.getParallelism());
        }
        PrefixMiningTask rootTask = new PrefixMiningTask(
            sortedItems, singleItemLists, 0, sortedItems.size()
        );
        threadPool.invoke(rootTask);

        if (DEBUG) {
            System.err.println("[DEBUG] Parallel mining completed");
            System.err.printf("[DEBUG] Final threshold: %.4f\n", topKManager.getThreshold());
        }

        List<Itemset> results = topKManager.getTopK();

        // Print statistics
        if (DEBUG) {
            System.err.println("[DEBUG] ===== Algorithm Statistics =====");
            System.err.printf("[DEBUG] Candidates Generated: %d\n", candidatesGenerated.get());
            System.err.printf("[DEBUG] Utility List Joins: %d\n", utilityListJoins.get());
            System.err.printf("[DEBUG] Pruned by PTWU: %d\n", prunedByPTWU.get());
            System.err.printf("[DEBUG] Pruned by EU+Remaining: %d\n", prunedByEU.get());
            System.err.printf("[DEBUG] Pruned by Probability: %d\n", prunedByProbability.get());
            System.err.printf("[DEBUG] Total Pruned: %d\n",
                prunedByPTWU.get() + prunedByEU.get() + prunedByProbability.get());
            System.err.printf("[DEBUG] Top-K Results Found: %d\n", results.size());
            System.err.println("[DEBUG] ====================================");
        }

        // Cleanup
        threadPool.shutdown();
        try {
            if (!threadPool.awaitTermination(60, TimeUnit.SECONDS)) {
                threadPool.shutdownNow();
            }
        } catch (InterruptedException e) {
            threadPool.shutdownNow();
            Thread.currentThread().interrupt();
        }

        return results;
    }

    /**
     * Recursively searches for high-utility itemsets using extension pattern
     */
    private void searchEnhanced(UPUList prefix, List<UPUList> extensions,
                               Map<Integer, UPUList> singleItemLists) {

        if (extensions == null || extensions.isEmpty()) {
            return;
        }

        ExtensionSearchTask task = new ExtensionSearchTask(
            prefix, extensions, singleItemLists, 0, extensions.size()
        );
        task.invoke();
    }

    // ============================================================================
    // UTILITY LIST OPERATIONS
    // ============================================================================

    /**
     * Joins two utility lists using tid-based intersection
     * Returns null if join produces empty result or fails PTWU pruning
     */
    private UPUList join(UPUList ul1, UPUList ul2) {
        if (DEBUG) utilityListJoins.incrementAndGet();

        if (DEBUG_VERBOSE) {
            System.err.printf("[DEBUG-VERBOSE] Joining: %s + %s\n", ul1.itemset, ul2.itemset);
        }

        int size1 = ul1.size;
        int size2 = ul2.size;
        if (size1 == 0 || size2 == 0) return null;

        double joinedPTWU = Math.min(ul1.ptwu, ul2.ptwu);
        double currentThreshold = topKManager.getThreshold();
        if (joinedPTWU < currentThreshold - EPSILON) {
            if (DEBUG) prunedByPTWU.incrementAndGet();
            if (DEBUG_VERBOSE) {
                System.err.printf("[DEBUG-VERBOSE]   Join PRUNED by PTWU: %.4f < %.4f\n",
                    joinedPTWU, currentThreshold);
            }
            return null;
        }

        int estimatedCapacity = Math.min(Math.min(size1, size2), 2048);
        estimatedCapacity = Math.max(estimatedCapacity, 16);

        List<UPUList.Element> joinedElements = new ArrayList<>(estimatedCapacity);

        // Merge-join on transaction IDs
        int i = 0, j = 0;
        while (i < size1 && j < size2) {
            int tid1 = ul1.tids[i];
            int tid2 = ul2.tids[j];

            if (tid1 == tid2) {
                double newUtility = ul1.utilities[i] + ul2.utilities[j];
                double newRemaining = Math.min(ul1.remainings[i], ul2.remainings[j]);
                double newLogProbability = ul1.logProbabilities[i] + ul2.logProbabilities[j];

                if (newLogProbability > LOG_EPSILON + 1) {
                    joinedElements.add(new UPUList.Element(
                        tid1, newUtility, newRemaining, newLogProbability
                    ));
                }
                i++;
                j++;
            } else if (tid1 < tid2) {
                i++;
            } else {
                j++;
            }
        }

        // Trim oversized array for small result sets
        if (joinedElements instanceof ArrayList &&
            joinedElements.size() < estimatedCapacity / 3 &&
            joinedElements.size() < 100) {
            ((ArrayList<UPUList.Element>) joinedElements).trimToSize();
        }

        Set<Integer> newItemset = createItemsetUnion(ul1.itemset, ul2.itemset);
        UPUList result = new UPUList(newItemset, joinedElements, joinedPTWU);

        if (DEBUG_VERBOSE) {
            System.err.printf("[DEBUG-VERBOSE]   Join result: %s (size=%d, EU=%.4f, PTWU=%.4f)\n",
                result.itemset, result.size, result.getSumEU(), result.ptwu);
        }

        return result;
    }

    /**
     * Creates union of two itemsets with size-optimized strategy
     */
    private Set<Integer> createItemsetUnion(Set<Integer> set1, Set<Integer> set2) {
        int size1 = set1.size();
        int size2 = set2.size();
        int totalSize = size1 + size2;

        if (totalSize <= 4) {
            Set<Integer> result = new HashSet<>(totalSize + 1, 1.0f);
            result.addAll(set1);
            result.addAll(set2);
            return result;
        }

        if (totalSize <= 20) {
            Set<Integer> larger = (size1 >= size2) ? set1 : set2;
            Set<Integer> smaller = (size1 >= size2) ? set2 : set1;

            Set<Integer> result = new HashSet<>(totalSize, 0.75f);
            result.addAll(larger);
            result.addAll(smaller);
            return result;
        }

        Set<Integer> result = new HashSet<>(totalSize, 0.75f);
        result.addAll(set1);
        result.addAll(set2);
        return result;
    }

    // ============================================================================
    // INITIALIZATION AND PREPROCESSING
    // ============================================================================

    /**
     * Optimized initialization using single-pass PTWU calculation and suffix sum preprocessing
     * Eliminates O(T²) nested loops with O(T) linear scan
     */
    private Map<Integer, UPUList> optimizedInitialization(List<Transaction> rawDatabase) {
        // Pass 1: Calculate PTWU and establish global ordering
        this.itemPTWU = calculatePTWUSinglePass(rawDatabase);
        this.itemToRank = buildItemRanking(itemPTWU);

        // Pass 2: Build utility lists using suffix sum optimization
        Map<Integer, List<UPUList.Element>> tempElements =
            buildUtilityListsWithSuffixSum(rawDatabase);

        // Pass 3: Create final utility lists with pre-computed aggregates
        Map<Integer, UPUList> singleItemLists = new HashMap<>();

        for (Map.Entry<Integer, List<UPUList.Element>> entry : tempElements.entrySet()) {
            Integer item = entry.getKey();
            List<UPUList.Element> elements = entry.getValue();

            if (!elements.isEmpty()) {
                Set<Integer> itemset = Collections.singleton(item);
                Double ptwu = itemPTWU.get(item);

                UPUList ul = new UPUList(itemset, elements, ptwu);

                if (ul.existentialProbability >= minPro - EPSILON) {
                    singleItemLists.put(item, ul);
                }
            }
        }

        return singleItemLists;
    }

    /**
     * Calculates Positive Transaction-Weighted Utility in single pass
     * PTWU = sum of positive utilities in transactions containing each item
     */
    private Map<Integer, Double> calculatePTWUSinglePass(List<Transaction> rawDatabase) {
        Map<Integer, Double> itemPTWU = new HashMap<>();

        for (Transaction rawTrans : rawDatabase) {
            // Calculate PTU (positive utilities only)
            double ptu = 0;
            for (Map.Entry<Integer, Integer> entry : rawTrans.items.entrySet()) {
                Integer item = entry.getKey();
                Integer quantity = entry.getValue();
                Double profit = itemProfits.get(item);
                if (profit != null && profit > 0) {
                    ptu += profit * quantity;
                }
            }

            // Accumulate PTWU for all items with positive probabilities
            for (Map.Entry<Integer, Integer> entry : rawTrans.items.entrySet()) {
                Integer item = entry.getKey();
                Double prob = rawTrans.probabilities.get(item);
                if (prob != null && prob > 0) {
                    itemPTWU.merge(item, ptu, Double::sum);
                }
            }
        }

        return itemPTWU;
    }

    /**
     * Builds global item ranking based on PTWU values
     */
    private Map<Integer, Integer> buildItemRanking(Map<Integer, Double> itemPTWU) {
        Map<Integer, Integer> itemToRank = new HashMap<>();

        List<Integer> rankedItems = itemPTWU.entrySet().stream()
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

        return itemToRank;
    }

    /**
     * Builds utility lists using suffix sum preprocessing
     * Eliminates O(T²) remaining utility calculation with O(T) preprocessing
     */
    private Map<Integer, List<UPUList.Element>> buildUtilityListsWithSuffixSum(
        List<Transaction> rawDatabase) {

        Map<Integer, List<UPUList.Element>> tempElements = new HashMap<>();

        for (Transaction rawTrans : rawDatabase) {
            List<ItemData> validItems = extractAndSortValidItems(rawTrans);
            if (validItems.isEmpty()) continue;

            double[] suffixSums = computeSuffixSums(validItems);

            for (int i = 0; i < validItems.size(); i++) {
                ItemData itemData = validItems.get(i);

                if (itemData.logProb > LOG_EPSILON) {
                    tempElements.computeIfAbsent(itemData.item, k -> new ArrayList<>())
                        .add(new UPUList.Element(
                            rawTrans.tid,
                            itemData.utility,
                            suffixSums[i],
                            itemData.logProb
                        ));
                }
            }
        }

        return tempElements;
    }

    /**
     * Extracts and sorts transaction items by global PTWU rank
     */
    private List<ItemData> extractAndSortValidItems(Transaction rawTrans) {
        List<ItemData> validItems = new ArrayList<>();

        for (Map.Entry<Integer, Integer> entry : rawTrans.items.entrySet()) {
            Integer item = entry.getKey();
            Integer quantity = entry.getValue();

            if (!itemToRank.containsKey(item)) continue;

            Double profit = itemProfits.get(item);
            Double prob = rawTrans.probabilities.get(item);

            if (profit != null && prob != null && prob > 0) {
                double logProb = prob > 0 ? Math.log(prob) : LOG_EPSILON;
                validItems.add(new ItemData(item, quantity, profit, logProb));
            }
        }

        validItems.sort((a, b) -> {
            Integer rankA = itemToRank.get(a.item);
            Integer rankB = itemToRank.get(b.item);
            return rankA.compareTo(rankB);
        });

        return validItems;
    }

    /**
     * Computes suffix sums for remaining utility calculation
     * suffixSum[i] = sum of positive utilities from position i+1 to end
     */
    private double[] computeSuffixSums(List<ItemData> validItems) {
        int n = validItems.size();
        double[] suffixSums = new double[n];

        suffixSums[n - 1] = 0.0;

        for (int i = n - 2; i >= 0; i--) {
            ItemData nextItem = validItems.get(i + 1);
            double nextUtility = nextItem.profit > 0 ? nextItem.utility : 0.0;
            suffixSums[i] = suffixSums[i + 1] + nextUtility;
        }

        return suffixSums;
    }

    /**
     * Returns items sorted by global PTWU rank
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

    // ============================================================================
    // FILE I/O AND MAIN ENTRY POINT
    // ============================================================================

    /**
     * Reads profit table: item -> profit (positive or negative)
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
     * Reads uncertain database: format per line is "item:quantity:probability"
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

    public static void main(String[] args) throws IOException {
        if (args.length < 4 || args.length > 5) {
            System.err.println("Usage: variant1 <database_file> <profit_file> <k> <min_probability> [--debug | --debug-verbose]");
            System.exit(1);
        }

        String dbFile = args[0];
        String profitFile = args[1];
        int k = Integer.parseInt(args[2]);
        double minPro = Double.parseDouble(args[3]);

        boolean debug = false;
        boolean debugVerbose = false;
        if (args.length == 5) {
            if (args[4].equals("--debug")) {
                debug = true;
            } else if (args[4].equals("--debug-verbose")) {
                debugVerbose = true;
            }
        }

        Map<Integer, Double> profits = readProfitTable(profitFile);
        List<Transaction> database = readDatabase(dbFile);

        variant1 algorithm = new variant1(profits, k, minPro, debug, debugVerbose);

        // Force garbage collection and measure baseline memory
        Runtime runtime = Runtime.getRuntime();
        System.gc();
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();

        // Measure execution time
        long startTime = System.nanoTime();
        List<Itemset> topK = algorithm.mine(database);
        long endTime = System.nanoTime();

        // Force garbage collection and measure final memory
        System.gc();
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        long memoryAfter = runtime.totalMemory() - runtime.freeMemory();

        // Calculate statistics
        double executionTimeMs = (endTime - startTime) / 1_000_000.0;
        double memoryUsedMB = (memoryAfter - memoryBefore) / (1024.0 * 1024.0);

        // Print performance statistics
        System.out.println("=== Performance Statistics ===");
        System.out.printf("Execution Time: %.2f ms\n", executionTimeMs);
        System.out.printf("Memory Used: %.2f MB\n", memoryUsedMB);
        System.out.println();

        // Print results
        System.out.println("=== Top-" + k + " PHUIs ===");
        int rank = 1;
        for (Itemset itemset : topK) {
            System.out.printf("%d. %s\n", rank++, itemset);
        }
    }
}
