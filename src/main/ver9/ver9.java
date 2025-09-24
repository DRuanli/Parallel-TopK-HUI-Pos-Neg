package main.ver9;

import java.io.*;
import java.time.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.stream.*;

/**
 * PTK-HUIM-UÂ±: Enhanced Parallel Top-K High-Utility Itemset Mining
 * from Uncertain Databases with Positive and Negative Utilities
 *
 * VERSION 9
 * - CAS-based TopKManager from
 * - Pre-computed EnhancedUtilityList
 * - Optimized initialization
 * - Best algorithmic AND runtime optimizations combined!
 *
 * @author Elio
 * @version 9
 */
public class ver9 {
    private final Map<Integer, Double> itemProfits;
    private final int k;
    private final double minPro;
    private static final double EPSILON = 1e-10;
    private static final double LOG_EPSILON = -700;

    // Thread-safe top-K management - CAS-based from ver5_2
    private final TopKManager topKManager;

    // Item ordering structures
    private Map<Integer, Integer> itemToRank;
    private Map<Integer, Double> itemRTWU;

    // Enhanced statistics - thread-safe
    private final AtomicLong candidatesGenerated = new AtomicLong(0);
    private final AtomicLong candidatesPruned = new AtomicLong(0);
    private final AtomicLong utilityListsCreated = new AtomicLong(0);
    private final AtomicLong euPruned = new AtomicLong(0);
    private final AtomicLong epPruned = new AtomicLong(0);
    private final AtomicLong rtwuPruned = new AtomicLong(0);
    private final AtomicLong branchPruned = new AtomicLong(0);
    private final AtomicLong bulkBranchPruned = new AtomicLong(0);

    // Parallel execution control
    private final ForkJoinPool customThreadPool;
    private static final int PARALLEL_THRESHOLD = 30;
    private static final int TASK_GRANULARITY = 7;

    // Memory monitoring
    private final long maxMemory;
    private final AtomicLong peakMemoryUsage = new AtomicLong(0);

    /**
     * Enhanced Transaction with efficient storage (from ver5_2)
     */
    static class EnhancedTransaction {
        final int tid;
        final int[] items;
        final int[] quantities;
        final double[] logProbabilities;
        final double rtu;

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

            // Convert to arrays
            int size = sortedItems.size();
            this.items = new int[size];
            this.quantities = new int[size];
            this.logProbabilities = new double[size];

            int idx = 0;
            double rtu = 0;

            for (Integer item : sortedItems) {
                items[idx] = item;
                quantities[idx] = itemMap.get(item);

                double prob = probMap.getOrDefault(item, 0.0);

                if (prob < EPSILON) {
                    logProbabilities[idx] = LOG_EPSILON;
                } else if (prob > 1.0 - EPSILON) {
                    logProbabilities[idx] = 0.0;
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
    }

    /**
     * PRE-COMPUTED EnhancedUtilityList from ver5_2 (MAJOR OPTIMIZATION)
     * - Eliminates lazy evaluation overhead
     * - O(1) access for getSumEU() and getSumRemaining()
     * - Thread-safe immutable design
     * - Bug fixes from ver5_2 included
     */
    static class EnhancedUtilityList {
        /**
         * Original Element class - unchanged for full compatibility
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

        // ============= ORIGINAL STORAGE (PRESERVED) =============
        final Set<Integer> itemset;
        final List<Element> elements;
        final double rtwu;

        // ============= OPTIMIZATION: PRE-COMPUTED AGGREGATES =============
        private final double sumEU;                    // Pre-computed instead of cached
        private final double sumRemaining;             // Pre-computed instead of cached
        private final double existentialProbability;  // Pre-computed

        /**
         * OPTIMIZED CONSTRUCTOR - ver5_2's approach with bug fixes
         */
        EnhancedUtilityList(Set<Integer> itemset, List<Element> elements, double rtwu) {
            // ============= PRESERVE ORIGINAL STORAGE =============
            this.itemset = Collections.unmodifiableSet(new HashSet<>(itemset));
            this.elements = Collections.unmodifiableList(elements);
            this.rtwu = rtwu;

            // ============= BUG FIX: SEPARATE CALCULATIONS =============
            // Fixed version from ver5_2 - calculate sums and existential probability separately

            double tempSumEU = 0.0;
            double tempSumRemaining = 0.0;

            // STEP 1: Calculate sums - MUST process ALL elements, no early termination
            for (Element e : elements) {
                double prob = Math.exp(e.logProbability);
                tempSumEU += e.utility * prob;
                tempSumRemaining += e.remaining * prob;
            }

            // STEP 2: Calculate existential probability separately - can use early termination
            double logComplement = 0.0;
            for (Element e : elements) {
                // Early termination for existential probability (this was the bug cause)
                if (e.logProbability > Math.log(1.0 - EPSILON)) {
                    // Probability is very close to 1, existential probability = 1.0
                    logComplement = LOG_EPSILON;
                    break;  // Safe: sums already calculated above
                } else {
                    double prob = Math.exp(e.logProbability);
                    // Use log1p for better numerical stability when prob is small
                    double log1MinusP = prob < 0.5 ?
                        Math.log1p(-prob) :           // More accurate for small prob
                        Math.log(1.0 - prob);        // Standard calculation
                    logComplement += log1MinusP;

                    // Early termination if complement becomes negligible
                    if (logComplement < LOG_EPSILON) {
                        logComplement = LOG_EPSILON;
                        break;  // Safe: sums already calculated above
                    }
                }
            }

            // Store final computed values (eliminates need for lazy evaluation)
            this.sumEU = tempSumEU;
            this.sumRemaining = tempSumRemaining;

            // Calculate final existential probability with enhanced numerical stability
            if (logComplement < LOG_EPSILON) {
                this.existentialProbability = 1.0;
            } else {
                this.existentialProbability = 1.0 - Math.exp(logComplement);
            }
        }

        // ============= MAJOR OPTIMIZATION: O(1) ACCESSORS =============

        /**
         * O(1) access - pre-computed during construction
         * ELIMINATES lazy evaluation overhead from ver4_6/ver4_7
         */
        double getSumEU() {
            return sumEU;  // Direct return, no computation needed
        }

        /**
         * O(1) access - pre-computed during construction
         * ELIMINATES lazy evaluation overhead from ver4_6/ver4_7
         */
        double getSumRemaining() {
            return sumRemaining;  // Direct return, no computation needed
        }

        /**
         * Get existential probability - enhanced numerical stability from ver5_2
         */
        double getExistentialProbability() {
            return existentialProbability;
        }

        /**
         * Get number of elements - O(1)
         */
        int getSize() {
            return elements.size();
        }

        /**
         * Check if utility list is empty - O(1)
         */
        boolean isEmpty() {
            return elements.isEmpty();
        }

        @Override
        public String toString() {
            return String.format("EnhancedUtilityList{itemset=%s, elements=%d, sumEU=%.2f, sumRemaining=%.2f, existProb=%.4f, rtwu=%.2f}",
                itemset, elements.size(), sumEU, sumRemaining, existentialProbability, rtwu);
        }
    }

    /**
     * CAS-based TopKManager from ver5_2 (MAXIMUM PERFORMANCE)
     * - Lock-free operations using AtomicReferenceArray
     * - CAS performance tracking
     * - Optimal for parallel mining workloads
     */
    private class TopKManager {
        private final int k;
        private final AtomicReferenceArray<Itemset> topKArray;
        private final AtomicInteger size = new AtomicInteger(0);
        private final AtomicReference<Double> threshold = new AtomicReference<>(0.0);

        // Performance tracking from ver5_2
        private final AtomicLong casRetries = new AtomicLong(0);
        private final AtomicLong successfulUpdates = new AtomicLong(0);

        private volatile double cachedThreshold = 0.0;

        TopKManager(int k) {
            this.k = k;
            this.topKArray = new AtomicReferenceArray<>(k);
        }

        boolean tryAdd(Set<Integer> items, double eu, double ep) {
            // Fast path - check cached threshold first
            if (eu < cachedThreshold - EPSILON) {
                return false;
            }

            // Check if we can add to any empty slot
            for (int i = 0; i < k; i++) {
                if (topKArray.compareAndSet(i, null, new Itemset(items, eu, ep))) {
                    size.incrementAndGet();
                    successfulUpdates.incrementAndGet();
                    updateThreshold();
                    return true;
                }
            }

            // Check for duplicates and better items to replace
            for (int i = 0; i < k; i++) {
                Itemset existing = topKArray.get(i);
                if (existing != null && existing.items.equals(items)) {
                    if (eu > existing.expectedUtility + EPSILON) {
                        Itemset newItemset = new Itemset(items, eu, ep);
                        if (topKArray.compareAndSet(i, existing, newItemset)) {
                            successfulUpdates.incrementAndGet();
                            updateThreshold();
                            return true;
                        } else {
                            casRetries.incrementAndGet();
                        }
                    }
                    return false;
                }
            }

            // Find weakest item to replace
            if (size.get() >= k) {
                int weakestIndex = findWeakestIndex();
                if (weakestIndex != -1) {
                    Itemset weakest = topKArray.get(weakestIndex);
                    if (weakest != null && eu > weakest.expectedUtility + EPSILON) {
                        Itemset newItemset = new Itemset(items, eu, ep);
                        if (topKArray.compareAndSet(weakestIndex, weakest, newItemset)) {
                            successfulUpdates.incrementAndGet();
                            updateThreshold();
                            return true;
                        } else {
                            casRetries.incrementAndGet();
                        }
                    }
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

        private void updateThreshold() {
            double minEU = Double.MAX_VALUE;
            int count = 0;

            for (int i = 0; i < k; i++) {
                Itemset item = topKArray.get(i);
                if (item != null) {
                    count++;
                    if (item.expectedUtility < minEU) {
                        minEU = item.expectedUtility;
                    }
                }
            }

            if (count >= k) {
                threshold.set(minEU);
                cachedThreshold = minEU;
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

        // Performance metrics from ver5_2!
        long getSuccessfulUpdates() {
            return successfulUpdates.get();
        }

        long getCASRetries() {
            return casRetries.get();
        }
    }

    /**
     * MERGED: Helper class from ver4_9 for efficient suffix sum processing
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

    /**
     * ForkJoin task for parallel prefix mining (from ver5_2)
     */
    private class PrefixMiningTask extends RecursiveAction {
        private final List<Integer> sortedItems;
        private final Map<Integer, EnhancedUtilityList> singleItemLists;
        private final int start, end;

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
                PrefixMiningTask left = new PrefixMiningTask(sortedItems, singleItemLists, start, mid);
                PrefixMiningTask right = new PrefixMiningTask(sortedItems, singleItemLists, mid, end);

                left.fork();
                right.compute();
                left.join();
            }
        }

        private void processPrefix(int index) {
            Integer item = sortedItems.get(index);
            EnhancedUtilityList ul = singleItemLists.get(item);

            if (ul == null) return;

            double currentThreshold = topKManager.getThreshold();
            if (itemRTWU.get(item) < currentThreshold - EPSILON) {
                branchPruned.incrementAndGet();
                return;
            }

            List<EnhancedUtilityList> extensions = new ArrayList<>();
            for (int j = index + 1; j < sortedItems.size(); j++) {
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

            if (index % 10 == 0) {
                long usedMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                peakMemoryUsage.updateAndGet(peak -> Math.max(peak, usedMemory));
            }
        }
    }

    /**
     * ForkJoin task for parallel extension search (from ver5_2)
     */
    private class ExtensionSearchTask extends RecursiveAction {
        private final EnhancedUtilityList prefix;
        private final List<EnhancedUtilityList> extensions;
        private final Map<Integer, EnhancedUtilityList> singleItemLists;
        private final int start, end;

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

            // Bulk branch pruning for parallel tasks
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
                ExtensionSearchTask left = new ExtensionSearchTask(
                    prefix, extensions, singleItemLists, start, mid
                );
                ExtensionSearchTask right = new ExtensionSearchTask(
                    prefix, extensions, singleItemLists, mid, end
                );

                invokeAll(left, right);
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

            EnhancedUtilityList joined = ver9.this.join(prefix, extension);

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

            // Pruning Strategy 2: EU + remaining - NOW O(1) thanks to ver5_2 optimization!
            double sumEU = joined.getSumEU();        // O(1) - no computation needed!
            double sumRemaining = joined.getSumRemaining(); // O(1) - no computation needed!

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

            // Recursive search
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

    // Constructor
    public ver9(Map<Integer, Double> itemProfits, int k, double minPro) {
        this.itemProfits = Collections.unmodifiableMap(new HashMap<>(itemProfits));
        this.k = k;
        this.minPro = minPro;
        this.topKManager = new TopKManager(k);

        int numThreads = Runtime.getRuntime().availableProcessors();
        this.customThreadPool = new ForkJoinPool(numThreads);

        this.maxMemory = Runtime.getRuntime().maxMemory();
    }

    /**
     * Main mining method
     */
    public List<Itemset> mine(List<Transaction> rawDatabase) {
        Instant start = Instant.now();

        System.out.println("=== PTK-HUIM-UÂ± v5.2 OPTIMIZED (Ultimate Hybrid) ===");
        System.out.println("CAS TopKManager + Pre-computed UtilityList + Suffix Sum Optimization!");
        System.out.println("Database size: " + rawDatabase.size());
        System.out.println("Number of items: " + itemProfits.size());
        System.out.println("K: " + k + ", MinPro: " + minPro);
        System.out.println("Available memory: " + (maxMemory / 1024 / 1024) + " MB");
        System.out.println("Thread pool size: " + customThreadPool.getParallelism());

        // Phase 1: OPTIMIZED Initialization (from ver4_9)
        System.out.println("\nPhase 1: Optimized initialization with suffix sum preprocessing...");
        Map<Integer, EnhancedUtilityList> singleItemLists = optimizedInitialization(rawDatabase);

        List<Integer> sortedItems = getSortedItemsByRank(singleItemLists.keySet());

        System.out.println("Items after filtering: " + sortedItems.size());

        // Process single items
        for (Integer item : sortedItems) {
            EnhancedUtilityList ul = singleItemLists.get(item);
            if (ul != null) {
                double sumEU = ul.getSumEU(); // O(1) access!
                if (sumEU >= topKManager.getThreshold() - EPSILON &&
                    ul.existentialProbability >= minPro - EPSILON) {
                    topKManager.tryAdd(ul.itemset, sumEU, ul.existentialProbability);
                }
            }
        }

        System.out.println("\nPhase 2: Enhanced parallel mining...");

        // Use ForkJoinPool for parallel mining
        if (sortedItems.size() >= PARALLEL_THRESHOLD) {
            System.out.println("Using parallel processing for " + sortedItems.size() + " items");

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
            System.out.println("Using sequential processing for " + sortedItems.size() + " items");
            sequentialMining(sortedItems, singleItemLists);
        }

        List<Itemset> results = topKManager.getTopK();

        Instant end = Instant.now();

        System.out.println("\n=== Mining Complete ===");
        System.out.println("Execution time: " + Duration.between(start, end).toMillis() + " ms");
        System.out.println("Candidates generated: " + candidatesGenerated.get());
        System.out.println("Utility lists created: " + utilityListsCreated.get());
        System.out.println("Enhanced pruning statistics:");
        System.out.println("  - RTWU pruned: " + rtwuPruned.get());
        System.out.println("  - Branches pruned: " + branchPruned.get());
        System.out.println("  - Bulk branches pruned: " + bulkBranchPruned.get());
        System.out.println("  - EU+remaining pruned: " + euPruned.get());
        System.out.println("  - Existential probability pruned: " + epPruned.get());
        System.out.println("  - Total pruned: " + candidatesPruned.get());
        System.out.println("Peak memory usage: " + (peakMemoryUsage.get() / 1024 / 1024) + " MB");
        System.out.println("Final threshold: " + String.format("%.4f", topKManager.getThreshold()));
        System.out.println("Top-K found: " + results.size());

        // CAS Performance statistics - restored from ver5_2!
        System.out.println("Lock-free TopK statistics:");
        System.out.println("  - Successful updates: " + topKManager.getSuccessfulUpdates());
        System.out.println("  - CAS retries: " + topKManager.getCASRetries());
        if (topKManager.getSuccessfulUpdates() + topKManager.getCASRetries() > 0) {
            double casEfficiency = (double) topKManager.getSuccessfulUpdates() /
                (topKManager.getSuccessfulUpdates() + topKManager.getCASRetries());
            System.out.println("  - CAS efficiency: " + String.format("%.2f%%", casEfficiency * 100));
        }

        // Shutdown thread pool
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
     * Sequential mining - same logic as ver5_2
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
     * Search with enhanced pruning - same logic as ver5_2
     */
    private void searchEnhanced(EnhancedUtilityList prefix, List<EnhancedUtilityList> extensions,
                               Map<Integer, EnhancedUtilityList> singleItemLists) {

        if (extensions == null || extensions.isEmpty()) {
            return;
        }

        double currentThreshold = topKManager.getThreshold();

        // Check minimum RTWU for bulk pruning
        double minExtensionRTWU = Double.MAX_VALUE;
        for (EnhancedUtilityList ext : extensions) {
            if (ext.rtwu < minExtensionRTWU) {
                minExtensionRTWU = ext.rtwu;
            }
        }

        double maxPossibleRTWU = Math.min(prefix.rtwu, minExtensionRTWU);

        if (maxPossibleRTWU < currentThreshold - EPSILON) {
            bulkBranchPruned.incrementAndGet();
            return;
        }

        // Filter viable extensions
        List<EnhancedUtilityList> viableExtensions = new ArrayList<>();
        for (EnhancedUtilityList ext : extensions) {
            if (ext.rtwu >= currentThreshold - EPSILON) {
                viableExtensions.add(ext);
            } else {
                rtwuPruned.incrementAndGet();
            }
        }
        viableExtensions.sort((a, b) -> Double.compare(b.rtwu, a.rtwu));

        // Determine parallel or sequential processing
        if (viableExtensions.size() >= PARALLEL_THRESHOLD && ForkJoinTask.inForkJoinPool()) {
            ExtensionSearchTask task = new ExtensionSearchTask(
                prefix, viableExtensions, singleItemLists, 0, viableExtensions.size()
            );
            task.invoke();
        } else {
            // Sequential processing
            List<EnhancedUtilityList> newExtensions = new ArrayList<>(extensions.size());

            for (int i = 0; i < extensions.size(); i++) {
                EnhancedUtilityList extension = extensions.get(i);

                if (extension.rtwu < currentThreshold - EPSILON) {
                    rtwuPruned.incrementAndGet();
                    candidatesPruned.incrementAndGet();
                    continue;
                }

                EnhancedUtilityList joined = join(prefix, extension);

                if (joined == null || joined.elements.isEmpty()) {
                    continue;
                }

                utilityListsCreated.incrementAndGet();
                candidatesGenerated.incrementAndGet();

                double threshold = topKManager.getThreshold();

                if (joined.existentialProbability < minPro - EPSILON) {
                    epPruned.incrementAndGet();
                    candidatesPruned.incrementAndGet();
                    continue;
                }

                // O(1) utility access thanks to pre-computation!
                double sumEU = joined.getSumEU();
                double sumRemaining = joined.getSumRemaining();

                if (sumEU + sumRemaining < threshold - EPSILON) {
                    euPruned.incrementAndGet();
                    candidatesPruned.incrementAndGet();
                    continue;
                }

                if (sumEU >= threshold - EPSILON &&
                    joined.existentialProbability >= minPro - EPSILON) {
                    topKManager.tryAdd(joined.itemset, sumEU, joined.existentialProbability);
                }

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
    }

    /**
     * Join two utility-lists - same optimization as ver5_2
     */
    private EnhancedUtilityList join(EnhancedUtilityList ul1, EnhancedUtilityList ul2) {
        double joinedRTWU = Math.min(ul1.rtwu, ul2.rtwu);

        double currentThreshold = topKManager.getThreshold();
        if (joinedRTWU < currentThreshold - EPSILON) {
            rtwuPruned.incrementAndGet();
            return null;
        }

        int maxPossibleSize = Math.min(ul1.elements.size(), ul2.elements.size());

        if (currentThreshold > 0 && joinedRTWU < currentThreshold * 0.1) {
            return null;
        }

        int size1 = ul1.elements.size();
        int size2 = ul2.elements.size();

        if (size1 == 0 || size2 == 0) return null;

        // Optimized initial capacity estimation
        int estimatedCapacity = Math.min(size1, size2) / 3;
        estimatedCapacity = Math.max(estimatedCapacity, 4);
        estimatedCapacity = Math.min(estimatedCapacity, 32);

        List<EnhancedUtilityList.Element> joinedElements = new ArrayList<>(estimatedCapacity);

        int i = 0, j = 0;
        int consecutiveMisses = 0;
        while (i < size1 && j < size2) {
            EnhancedUtilityList.Element e1 = ul1.elements.get(i);
            EnhancedUtilityList.Element e2 = ul2.elements.get(j);

            if (e1.tid == e2.tid) {
                double newUtility = e1.utility + e2.utility;
                double newRemaining = Math.min(e1.remaining, e2.remaining);
                double newLogProbability = e1.logProbability + e2.logProbability;

                if (newLogProbability > LOG_EPSILON + 1) {
                    joinedElements.add(new EnhancedUtilityList.Element(
                        e1.tid, newUtility, newRemaining, newLogProbability
                    ));
                }
                consecutiveMisses = 0;
                i++;
                j++;
            } else if (e1.tid < e2.tid) {
                i++;
                consecutiveMisses++;
            } else {
                j++;
                consecutiveMisses++;
            }
        }

        if (consecutiveMisses > 50 && joinedElements.isEmpty() && (i + j) > 100) {
            return null;
        }

        if (joinedElements instanceof ArrayList &&
            joinedElements.size() < estimatedCapacity /3 &&
            joinedElements.size() < 100) {
            ((ArrayList<EnhancedUtilityList.Element>) joinedElements).trimToSize();
        }

        Set<Integer> newItemset = createSafeItemsetUnion(ul1.itemset, ul2.itemset);
        return new EnhancedUtilityList(newItemset, joinedElements, joinedRTWU);
    }

    private Set<Integer> createSafeItemsetUnion(Set<Integer> set1, Set<Integer> set2) {
        int size1 = set1.size();
        int size2 = set2.size();
        int totalSize = size1 + size2;

        // Very small sets - direct copy
        if (totalSize <= 4) {
            Set<Integer> result = new HashSet<>(totalSize + 1, 1.0f);
            result.addAll(set1);
            result.addAll(set2);
            return result;
        }

        // Size-optimized addition (larger first)
        if (totalSize <= 20) {
            Set<Integer> larger = (size1 >= size2) ? set1 : set2;
            Set<Integer> smaller = (size1 >= size2) ? set2 : set1;

            Set<Integer> result = new HashSet<>(totalSize, 0.75f);
            result.addAll(larger);
            result.addAll(smaller);
            return result;
        }

        // Default for larger sets
        Set<Integer> result = new HashSet<>(totalSize, 0.75f);
        result.addAll(set1);
        result.addAll(set2);
        return result;
    }

    /**
     * MERGED: Optimized initialization from ver4_9 with ver5_2's pre-computed utilities
     * - Single-pass RTWU calculation
     * - Suffix sum preprocessing for O(T) utility list building
     * - Combined with pre-computed EnhancedUtilityList from ver5_2
     */
    private Map<Integer, EnhancedUtilityList> optimizedInitialization(List<Transaction> rawDatabase) {
        // PASS 1: Single-pass RTWU calculation (from ver4_9)
        System.out.println("Pass 1: Single-pass RTWU calculation with optimization...");
        this.itemRTWU = calculateRTWUSinglePass(rawDatabase);

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

        // PASS 2: Build utility-lists with suffix sum preprocessing (from ver4_9)
        System.out.println("Pass 2: Building utility-lists with suffix sum preprocessing...");
        Map<Integer, List<EnhancedUtilityList.Element>> tempElements =
            buildUtilityListsWithSuffixSum(rawDatabase);

        // Create final utility lists with ver4_9's pre-computed values
        Map<Integer, EnhancedUtilityList> singleItemLists = new HashMap<>();

        for (Map.Entry<Integer, List<EnhancedUtilityList.Element>> entry : tempElements.entrySet()) {
            Integer item = entry.getKey();
            List<EnhancedUtilityList.Element> elements = entry.getValue();

            if (!elements.isEmpty()) {
                Set<Integer> itemset = Collections.singleton(item);
                Double rtwu = itemRTWU.get(item);

                // Pre-computed EnhancedUtilityList - O(1) access after construction!
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
     * MERGED: Single-pass RTWU calculation from ver4_9
     * Optimization: Eliminates double iteration over transaction items
     * Reduces complexity from O(2 * |DB| * T_avg) to O(|DB| * T_avg)
     */
    private Map<Integer, Double> calculateRTWUSinglePass(List<Transaction> rawDatabase) {
        Map<Integer, Double> itemRTWU = new HashMap<>();

        for (Transaction rawTrans : rawDatabase) {
            // Calculate RTU (positive utilities only - matching original logic)
            double rtu = 0;
            for (Map.Entry<Integer, Integer> entry : rawTrans.items.entrySet()) {
                Integer item = entry.getKey();
                Integer quantity = entry.getValue();
                Double profit = itemProfits.get(item);
                if (profit != null && profit > 0) {
                    rtu += profit * quantity;
                }
            }

            // Update RTWU for ALL items with positive probabilities
            for (Map.Entry<Integer, Integer> entry : rawTrans.items.entrySet()) {
                Integer item = entry.getKey();
                Double prob = rawTrans.probabilities.get(item);
                if (prob != null && prob > 0) {
                    itemRTWU.merge(item, rtu, Double::sum);
                }
            }
        }

        return itemRTWU;
    }

    /**
     * MERGED: Utility list building with suffix sum preprocessing from ver4_9
     * Major optimization: Eliminates O(T²) nested loops with O(T) suffix sum computation
     * This is the most critical optimization for performance improvement
     */
    private Map<Integer, List<EnhancedUtilityList.Element>> buildUtilityListsWithSuffixSum(
        List<Transaction> rawDatabase) {

        Map<Integer, List<EnhancedUtilityList.Element>> tempElements = new HashMap<>();

        for (Transaction rawTrans : rawDatabase) {
            // Step 1: Extract and sort valid items by rank
            List<ItemData> validItems = extractAndSortValidItems(rawTrans);

            if (validItems.isEmpty()) continue;

            // Step 2: OPTIMIZATION - Precompute all suffix sums in O(T) time
            double[] suffixSums = computeSuffixSums(validItems);

            // Step 3: OPTIMIZATION - Single pass element creation with O(1) remaining lookup
            for (int i = 0; i < validItems.size(); i++) {
                ItemData itemData = validItems.get(i);

                if (itemData.logProb > LOG_EPSILON) {
                    // CRITICAL OPTIMIZATION: O(1) remaining utility lookup instead of O(T) calculation
                    double remainingUtility = suffixSums[i];

                    tempElements.computeIfAbsent(itemData.item, k -> new ArrayList<>())
                        .add(new EnhancedUtilityList.Element(
                            rawTrans.tid,
                            itemData.utility,
                            remainingUtility, // Pre-calculated in O(1) time!
                            itemData.logProb
                        ));
                }
            }
        }

        return tempElements;
    }

    /**
     * MERGED: Extract and sort valid items by RTWU rank from ver4_9
     * Helper method for suffix sum processing
     */
    private List<ItemData> extractAndSortValidItems(Transaction rawTrans) {
        List<ItemData> validItems = new ArrayList<>();

        for (Map.Entry<Integer, Integer> entry : rawTrans.items.entrySet()) {
            Integer item = entry.getKey();
            Integer quantity = entry.getValue();

            // Only process items that exist in our ranking
            if (!itemToRank.containsKey(item)) continue;

            Double profit = itemProfits.get(item);
            Double prob = rawTrans.probabilities.get(item);

            if (profit != null && prob != null && prob > 0) {
                double logProb = prob > 0 ? Math.log(prob) : LOG_EPSILON;
                validItems.add(new ItemData(item, quantity, profit, logProb));
            }
        }

        // Sort by RTWU rank for consistent suffix sum calculation
        validItems.sort((a, b) -> {
            Integer rankA = itemToRank.get(a.item);
            Integer rankB = itemToRank.get(b.item);
            return rankA.compareTo(rankB);
        });

        return validItems;
    }

    /**
     * MERGED: Core suffix sum computation from ver4_9 - THE KEY OPTIMIZATION
     * Replaces O(T²) nested loops with O(T) preprocessing
     *
     * Algorithm explanation:
     * - suffixSum[i] = sum of all positive utilities from position i+1 to end
     * - Computed right-to-left in single pass
     * - Each item's remaining utility = suffixSum[i]
     */
    private double[] computeSuffixSums(List<ItemData> validItems) {
        int n = validItems.size();
        double[] suffixSums = new double[n];

        // Base case: last item has no remaining utility
        suffixSums[n - 1] = 0.0;

        // OPTIMIZATION: Fill suffix sums from right to left in single pass
        for (int i = n - 2; i >= 0; i--) {
            ItemData nextItem = validItems.get(i + 1);

            // Only count positive utilities as "remaining"
            double nextUtility = nextItem.profit > 0 ? nextItem.utility : 0.0;

            suffixSums[i] = suffixSums[i + 1] + nextUtility;
        }

        return suffixSums;
    }

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
                   ", eu=" + String.format("%.2f", expectedUtility) +
                   '}';
        }
    }

    /**
     * Transaction class
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

    // Main method and file I/O - same as other versions
    public static void main(String[] args) throws IOException {
        if (args.length != 4) {
            System.err.println("Usage: ver4_9 <database_file> <profit_file> <k> <min_probability>");
            System.exit(1);
        }

        String dbFile = args[0];
        String profitFile = args[1];
        int k = Integer.parseInt(args[2]);
        double minPro = Double.parseDouble(args[3]);

        Map<Integer, Double> profits = readProfitTable(profitFile);
        List<Transaction> database = readDatabase(dbFile);

        System.out.println("=== PTK-HUIM-UÂ± Version 5.2 OPTIMIZED ===");
        System.out.println("Ultimate hybrid: CAS TopKManager + Pre-computed utilities + Suffix sum optimization");
        System.out.println();

        ver9 algorithm = new ver9(profits, k, minPro);
        List<Itemset> topK = algorithm.mine(database);

        System.out.println("\n=== Top-" + k + " PHUIs ===");
        int rank = 1;
        for (Itemset itemset : topK) {
            System.out.printf("%d. %s\n", rank++, itemset);
        }
    }

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