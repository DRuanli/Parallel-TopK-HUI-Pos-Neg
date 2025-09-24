package testing.mining;

import testing.core.*;
import testing.parallel.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.stream.*;

/**
 * Core mining engine for PTK-HUIM-U±
 */
public class MiningEngine {
    private final Map<Integer, Double> itemProfits;
    private final int k;
    private final double minPro;

    // Core components
    private final TopKManager topKManager;
    private final MiningStatistics statistics;
    private final PruningStrategy pruningStrategy;
    private final JoinOperator joinOperator;
    private final TaskScheduler taskScheduler;

    // Item ordering structures
    private Map<Integer, Integer> itemToRank;
    private Map<Integer, Double> itemRTWU;

    // Memory monitoring
    private final long maxMemory;
    private final AtomicLong peakMemoryUsage;

    public MiningEngine(Map<Integer, Double> itemProfits, int k, double minPro) {
        this.itemProfits = Collections.unmodifiableMap(new HashMap<>(itemProfits));
        this.k = k;
        this.minPro = minPro;
        this.topKManager = new TopKManager(k);
        this.statistics = new MiningStatistics();
        this.pruningStrategy = new PruningStrategy(topKManager, minPro, statistics);
        this.joinOperator = new JoinOperator(topKManager, statistics);

        this.maxMemory = Runtime.getRuntime().maxMemory();
        this.peakMemoryUsage = new AtomicLong(0);

        int numThreads = Runtime.getRuntime().availableProcessors();
        this.taskScheduler = new TaskScheduler(numThreads, peakMemoryUsage, this);
    }

    /**
     * Main mining method
     */
    public List<Itemset> mine(List<Transaction> rawDatabase) {
        System.out.println("\nPhase 1: Optimized initialization with suffix sum preprocessing...");
        Map<Integer, UtilityList> singleItemLists = optimizedInitialization(rawDatabase);

        List<Integer> sortedItems = getSortedItemsByRank(singleItemLists.keySet());

        System.out.println("Items after filtering: " + sortedItems.size());

        // Process single items
        for (Integer item : sortedItems) {
            UtilityList ul = singleItemLists.get(item);
            if (ul != null) {
                double sumEU = ul.getSumEU(); // O(1) access!
                if (sumEU >= topKManager.getThreshold() - MiningConstants.EPSILON &&
                    ul.existentialProbability >= minPro - MiningConstants.EPSILON) {
                    topKManager.tryAdd(ul.itemset, sumEU, ul.existentialProbability);
                }
            }
        }

        System.out.println("\nPhase 2: Enhanced parallel mining...");
        taskScheduler.executePrefixMining(sortedItems, singleItemLists);

        return topKManager.getTopK();
    }

    /**
     * Sequential mining - same logic as ver5_2
     */
    public void sequentialMining(List<Integer> sortedItems,
                                Map<Integer, UtilityList> singleItemLists) {
        for (int i = 0; i < sortedItems.size(); i++) {
            Integer item = sortedItems.get(i);
            UtilityList ul = singleItemLists.get(item);

            if (ul == null) continue;

            double currentThreshold = topKManager.getThreshold();
            if (itemRTWU.get(item) < currentThreshold - MiningConstants.EPSILON) {
                statistics.incrementBranchPruned();
                continue;
            }

            List<UtilityList> extensions = new ArrayList<>();
            for (int j = i + 1; j < sortedItems.size(); j++) {
                Integer extItem = sortedItems.get(j);
                UtilityList extUL = singleItemLists.get(extItem);

                if (extUL == null) continue;

                if (itemRTWU.get(extItem) < currentThreshold - MiningConstants.EPSILON) {
                    statistics.incrementRtwuPruned();
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
    public void searchEnhanced(UtilityList prefix, List<UtilityList> extensions,
                              Map<Integer, UtilityList> singleItemLists) {

        if (extensions == null || extensions.isEmpty()) {
            return;
        }

        double currentThreshold = topKManager.getThreshold();

        // Check minimum RTWU for bulk pruning
        double minExtensionRTWU = Double.MAX_VALUE;
        for (UtilityList ext : extensions) {
            if (ext.rtwu < minExtensionRTWU) {
                minExtensionRTWU = ext.rtwu;
            }
        }

        double maxPossibleRTWU = Math.min(prefix.rtwu, minExtensionRTWU);

        if (pruningStrategy.shouldBulkPrune(maxPossibleRTWU)) {
            return;
        }

        // Filter viable extensions
        List<UtilityList> viableExtensions = new ArrayList<>();
        for (UtilityList ext : extensions) {
            if (ext.rtwu >= currentThreshold - MiningConstants.EPSILON) {
                viableExtensions.add(ext);
            } else {
                statistics.incrementRtwuPruned();
            }
        }
        viableExtensions.sort((a, b) -> Double.compare(b.rtwu, a.rtwu));

        // Determine parallel or sequential processing
        if (!taskScheduler.executeExtensionSearch(prefix, viableExtensions, singleItemLists)) {
            // Sequential processing
            List<UtilityList> newExtensions = new ArrayList<>(extensions.size());

            for (int i = 0; i < extensions.size(); i++) {
                UtilityList extension = extensions.get(i);

                if (pruningStrategy.shouldPruneByRTWU(extension.rtwu)) {
                    statistics.incrementCandidatesPruned();
                    continue;
                }

                UtilityList joined = joinOperator.join(prefix, extension);

                if (joined == null || joined.elements.isEmpty()) {
                    continue;
                }

                statistics.incrementUtilityListsCreated();
                statistics.incrementCandidatesGenerated();

                if (pruningStrategy.shouldPruneByExistentialProbability(joined.existentialProbability)) {
                    continue;
                }

                // O(1) utility access thanks to pre-computation!
                double sumEU = joined.getSumEU();
                double sumRemaining = joined.getSumRemaining();

                if (pruningStrategy.shouldPruneByEUUpperBound(sumEU, sumRemaining)) {
                    continue;
                }

                if (pruningStrategy.qualifiesForTopK(sumEU, joined.existentialProbability)) {
                    topKManager.tryAdd(joined.itemset, sumEU, joined.existentialProbability);
                }

                if (i < extensions.size() - 1) {
                    newExtensions.clear();
                    double currentThresholdForFilter = topKManager.getThreshold();

                    for (int j = i + 1; j < extensions.size(); j++) {
                        UtilityList ext = extensions.get(j);
                        if (ext.rtwu >= currentThresholdForFilter - MiningConstants.EPSILON) {
                            newExtensions.add(ext);
                        } else {
                            statistics.incrementRtwuPruned();
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
     * Join method delegated to JoinOperator
     */
    public UtilityList join(UtilityList ul1, UtilityList ul2) {
        return joinOperator.join(ul1, ul2);
    }

    /**
     * MERGED: Optimized initialization from ver4_9 with ver5_2's pre-computed utilities
     * - Single-pass RTWU calculation
     * - Suffix sum preprocessing for O(T) utility list building
     * - Combined with pre-computed EnhancedUtilityList from ver5_2
     */
    private Map<Integer, UtilityList> optimizedInitialization(List<Transaction> rawDatabase) {
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
        Map<Integer, List<UtilityList.Element>> tempElements =
            buildUtilityListsWithSuffixSum(rawDatabase);

        // Create final utility lists with ver4_9's pre-computed values
        Map<Integer, UtilityList> singleItemLists = new HashMap<>();

        for (Map.Entry<Integer, List<UtilityList.Element>> entry : tempElements.entrySet()) {
            Integer item = entry.getKey();
            List<UtilityList.Element> elements = entry.getValue();

            if (!elements.isEmpty()) {
                Set<Integer> itemset = Collections.singleton(item);
                Double rtwu = itemRTWU.get(item);

                // Pre-computed EnhancedUtilityList - O(1) access after construction!
                UtilityList ul = new UtilityList(itemset, elements, rtwu);

                if (ul.existentialProbability >= minPro - MiningConstants.EPSILON) {
                    singleItemLists.put(item, ul);
                    statistics.incrementUtilityListsCreated();
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
    private Map<Integer, List<UtilityList.Element>> buildUtilityListsWithSuffixSum(
        List<Transaction> rawDatabase) {

        Map<Integer, List<UtilityList.Element>> tempElements = new HashMap<>();

        for (Transaction rawTrans : rawDatabase) {
            // Step 1: Extract and sort valid items by rank
            List<ItemData> validItems = extractAndSortValidItems(rawTrans);

            if (validItems.isEmpty()) continue;

            // Step 2: OPTIMIZATION - Precompute all suffix sums in O(T) time
            double[] suffixSums = computeSuffixSums(validItems);

            // Step 3: OPTIMIZATION - Single pass element creation with O(1) remaining lookup
            for (int i = 0; i < validItems.size(); i++) {
                ItemData itemData = validItems.get(i);

                if (itemData.logProb > MiningConstants.LOG_EPSILON) {
                    // CRITICAL OPTIMIZATION: O(1) remaining utility lookup instead of O(T) calculation
                    double remainingUtility = suffixSums[i];

                    tempElements.computeIfAbsent(itemData.item, k -> new ArrayList<>())
                        .add(new UtilityList.Element(
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
                double logProb = prob > 0 ? Math.log(prob) : MiningConstants.LOG_EPSILON;
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

    // Getters for components
    public TopKManager getTopKManager() { return topKManager; }
    public MiningStatistics getStatistics() { return statistics; }
    public Map<Integer, Double> getItemRTWU() { return itemRTWU; }
    public double getMinPro() { return minPro; }
    public TaskScheduler getTaskScheduler() { return taskScheduler; }
    public long getPeakMemoryUsage() { return peakMemoryUsage.get(); }
}