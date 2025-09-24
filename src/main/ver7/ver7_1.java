package main.ver7;

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
 * VERSION 7.1 - MATCHING VER7 OUTPUT EXACTLY
 *
 * PRUNING STRATEGIES IMPLEMENTED:
 * 1. RTWU Pruning - Items/itemsets with RTWU below threshold
 * 2. Existential Probability Pruning - Probability below minPro
 * 3. Expected Utility Upper Bound Pruning - EU + Remaining < threshold
 * 4. Branch Pruning - Entire branch pruned if prefix RTWU too low
 * 5. Bulk Branch Pruning - All extensions pruned if max possible RTWU too low
 * 6. Dynamic Threshold Pruning - Continuous threshold updates
 *
 * @author Elio (Matching ver7)
 * @version 7.1-matching
 */
public class ver7_1 {
    private final Map<Integer, Double> itemProfits;
    private final int k;
    private final double minPro;
    private static final double EPSILON = 1e-10;
    private static final double LOG_EPSILON = -700;

    // Thread-safe top-K management
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
     * Enhanced Transaction with efficient storage
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
     * Optimized Utility-List with lazy computation
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

        private Double cachedSumEU = null;
        private Double cachedSumRemaining = null;

        EnhancedUtilityList(Set<Integer> itemset, List<Element> elements, double rtwu) {
            this.itemset = Collections.unmodifiableSet(new HashSet<>(itemset));
            this.elements = Collections.unmodifiableList(elements);
            this.rtwu = rtwu;
            this.existentialProbability = calculateLogSpaceExistentialProbability();
        }

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

        private double calculateLogSpaceExistentialProbability() {
            if (elements.isEmpty()) return 0.0;

            double logComplement = 0.0;

            for (Element e : elements) {
                double logP = e.logProbability;

                if (logP > -1e-15) {
                    return 1.0;
                }

                double log1MinusP;

                if (logP < -10) {
                    log1MinusP = -Math.exp(logP);
                } else if (logP < -0.693) {
                    log1MinusP = Math.log1p(-Math.exp(logP));
                } else {
                    double p = Math.exp(logP);
                    if (p > 0.9999999999) {
                        return 1.0;
                    }
                    log1MinusP = Math.log(1.0 - p);
                }

                logComplement += log1MinusP;

                if (logComplement < -30) {
                    return 1.0;
                }
            }

            if (logComplement < LOG_EPSILON) {
                return 1.0;
            }

            double complement = Math.exp(logComplement);
            return 1.0 - complement;
        }
    }

    /**
     * ForkJoin task for parallel prefix mining
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
     * ForkJoin task for parallel extension search
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

            EnhancedUtilityList joined = ver7_1.this.join(prefix, extension);

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

            // Pruning Strategy 2: EU + remaining
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
    public ver7_1(Map<Integer, Double> itemProfits, int k, double minPro) {
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

        System.out.println("=== PTK-HUIM-U± v7.1 (Matching ver7) ===");
        System.out.println("Database size: " + rawDatabase.size());
        System.out.println("Number of items: " + itemProfits.size());
        System.out.println("K: " + k + ", MinPro: " + minPro);
        System.out.println("Available memory: " + (maxMemory / 1024 / 1024) + " MB");
        System.out.println("Thread pool size: " + customThreadPool.getParallelism());

        // Phase 1: Initialization
        System.out.println("\nPhase 1: Single-pass initialization...");
        Map<Integer, EnhancedUtilityList> singleItemLists = optimizedInitialization(rawDatabase);

        List<Integer> sortedItems = getSortedItemsByRank(singleItemLists.keySet());

        System.out.println("Items after filtering: " + sortedItems.size());

        // Process single items
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
     * Sequential mining with MATCHING ver7 behavior
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
     * Search with enhanced pruning - MATCHING ver7 behavior exactly
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
            // Sequential processing - MATCHING ver7's bug/behavior
            List<EnhancedUtilityList> newExtensions = new ArrayList<>(extensions.size());

            // IMPORTANT: Process original extensions list to match ver7's behavior
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
     * Join two utility-lists
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

        // OPTIMIZATION: Better initial capacity estimation
        int estimatedCapacity = Math.min(size1, size2) / 3; // Conservative estimate
        estimatedCapacity = Math.max(estimatedCapacity, 4); // Minimum capacity
        estimatedCapacity = Math.min(estimatedCapacity, 32); // Maximum initial capacity

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

        // STRATEGY A: Very small sets - direct copy
        if (totalSize <= 4) {
            Set<Integer> result = new HashSet<>(totalSize + 1, 1.0f);
            result.addAll(set1);
            result.addAll(set2);
            return result;
        }

        // STRATEGY B: Size-optimized addition (larger first)
        if (totalSize <= 20) {
            Set<Integer> larger = (size1 >= size2) ? set1 : set2;
            Set<Integer> smaller = (size1 >= size2) ? set2 : set1;

            Set<Integer> result = new HashSet<>(totalSize, 0.75f);
            result.addAll(larger);
            result.addAll(smaller);
            return result;
        }

        // STRATEGY C: Default for larger sets
        Set<Integer> result = new HashSet<>(totalSize, 0.75f);
        result.addAll(set1);
        result.addAll(set2);
        return result;
    }


    /**
     * Optimized single-pass initialization
     */
    private Map<Integer, EnhancedUtilityList> optimizedInitialization(List<Transaction> rawDatabase) {
        // Pass 1: Calculate RTWU
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

        // Build global ordering
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

        // Pass 2: Build utility-lists
        System.out.println("Pass 2: Building utility-lists directly...");

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

        // Create final utility lists
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
     * Thread-safe Top-K Manager
     */
    private class TopKManager {
        private final int k;
        private final ConcurrentSkipListSet<Itemset> topKSet;
        private final ConcurrentHashMap<Set<Integer>, Itemset> topKMap;
        private final AtomicReference<Double> threshold;
        private volatile double cachedThreshold = 0.0;
        private final Object lock = new Object();

        TopKManager(int k) {
            this.k = k;
            this.topKSet = new ConcurrentSkipListSet<>((a, b) -> {
                int cmp = Double.compare(b.expectedUtility, a.expectedUtility);
                if (cmp != 0) return cmp;
                return Integer.compare(a.hashCode(), b.hashCode());
            });
            this.topKMap = new ConcurrentHashMap<>();
            this.threshold = new AtomicReference<>(0.0);
        }

        boolean tryAdd(Set<Integer> items, double eu, double ep) {
            if (eu < cachedThreshold - EPSILON) {
                return false;
            }

            if (eu < threshold.get() - EPSILON) {
                return false;
            }

            Itemset existingItemset = topKMap.get(items);
            if (existingItemset != null && existingItemset.expectedUtility >= eu - EPSILON) {
                return false;
            }

            synchronized (lock) {
                if (topKSet.size() >= k && eu < threshold.get() - EPSILON) {
                    return false;
                }

                Itemset currentVersionInMap = topKMap.get(items);
                if (currentVersionInMap != null) {
                    if (currentVersionInMap.expectedUtility >= eu - EPSILON) {
                        return false;
                    }
                    topKSet.remove(currentVersionInMap);
                }

                Itemset newItemset = new Itemset(items, eu, ep);
                topKSet.add(newItemset);
                topKMap.put(items, newItemset);

                if (topKSet.size() > k) {
                    Itemset removed = topKSet.pollLast();
                    if (removed != null) {
                        topKMap.remove(removed.items);
                    }
                }

                if (topKSet.size() >= k) {
                    double newThreshold = topKSet.last().expectedUtility;
                    threshold.set(newThreshold);
                    cachedThreshold = newThreshold;
                }
            }
            return true;
        }

        double getThreshold() {
            return cachedThreshold;
        }

        List<Itemset> getTopK() {
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

    // Main method and file I/O
    public static void main(String[] args) throws IOException {
        if (args.length != 4) {
            System.err.println("Usage: ver7_1 <database_file> <profit_file> <k> <min_probability>");
            System.exit(1);
        }

        String dbFile = args[0];
        String profitFile = args[1];
        int k = Integer.parseInt(args[2]);
        double minPro = Double.parseDouble(args[3]);

        Map<Integer, Double> profits = readProfitTable(profitFile);
        List<Transaction> database = readDatabase(dbFile);

        System.out.println("=== PTK-HUIM-U± Version 7.1 ===");
        System.out.println("Matching ver7 output behavior");
        System.out.println();

        ver7_1 algorithm = new ver7_1(profits, k, minPro);
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