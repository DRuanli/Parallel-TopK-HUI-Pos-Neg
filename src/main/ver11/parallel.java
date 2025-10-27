import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.stream.*;

/**
 * PTK-HUIM-U±: Parallel Top-K High-Utility Itemset Mining
 * from Uncertain Databases with Positive and Negative Utilities
 *
 *
 * @author Võ Gia Huy, Lê Đăng Nguyễn
 * @advisor Prof. Nguyễn Chí Thiện
 */
public class parallel {

    // ============================================================================
    // CONSTANTS AND CONFIGURATION
    // ============================================================================

    /** Small value to handle numerical precision issues */
    private static final double EPSILON = 1e-10;

    /** Log-space epsilon to prevent underflow in probability calculations */
    private static final double LOG_EPSILON = -700.0;

    /** Task granularity threshold for parallel decomposition */
    private static final int TASK_GRANULARITY = 4;

    /** Debug mode flags */
    private static boolean DEBUG = false;
    private static boolean DEBUG_VERBOSE = false;

    // ============================================================================
    // CORE DATA STRUCTURES
    // ============================================================================

    /**
     * Transaction: Represents a single transaction in the uncertain database.
     *
     * Each transaction contains:
     * - A unique transaction ID (tid)
     * - Items with their quantities
     * - Occurrence probabilities for each item (TRANSACTION-LEVEL probabilities)
     *
     * Example: Transaction T1 = {(a:2, 0.8), (b:3, 0.9)}
     * - Item 'a' appears with quantity 2 and probability 0.8 IN THIS TRANSACTION
     * - Item 'b' appears with quantity 3 and probability 0.9 IN THIS TRANSACTION
     */
    static class Transaction {
        /**
         * Transaction identifier (unique ID)
         */
        final int tid;

        /**
         * Map: item ID -> quantity in this transaction
         * Example: {1 -> 2, 3 -> 5} means item 1 has quantity 2, item 3 has quantity 5
         */
        final Map<Integer, Integer> items;

        /**
         * Map: item ID -> occurrence probability in this transaction (TRANSACTION-LEVEL)
         * Example: {1 -> 0.8, 3 -> 0.9} means:
         * - Item 1 appears in this transaction with probability 0.8
         * - Item 3 appears in this transaction with probability 0.9
         *
         * Note: This is NOT the database-level existential probability!
         */
        final Map<Integer, Double> probabilities;

        /**
         * Constructor for Transaction
         *
         * @param tid Transaction ID
         * @param items Map from item ID to quantity
         * @param probabilities Map from item ID to transaction-level occurrence probability
         */
        Transaction(int tid, Map<Integer, Integer> items, Map<Integer, Double> probabilities) {
            this.tid = tid;
            this.items = items;
            this.probabilities = probabilities;
        }
    }

    /**
     * PatternResult: Represents a discovered high-utility itemset pattern.
     * This is the FINAL OUTPUT structure containing database-level statistics.
     *
     * IMPORTANT: This class represents COMPLETE PATTERNS found during mining,
     * not intermediate data structures used during the mining process.
     */
    public static class PatternResult {
        /**
         * Set of item IDs that form this pattern
         * Example: {1, 3, 5} represents the itemset containing items 1, 3, and 5
         */
        final Set<Integer> itemIDs;

        /**
         * Expected Utility (EU) of this pattern across the ENTIRE DATABASE
         * This is the sum of (probability × utility) across all transactions
         *
         * Formula: EU(X) = Σ[Ti contains X] P(X,Ti) × u(X,Ti)
         */
        final double expectedUtility;

        /**
         * Existential Probability (EP) across the ENTIRE DATABASE
         * This is the probability that this pattern appears in AT LEAST ONE transaction
         *
         * Formula: EP(X) = 1 - Π[Ti contains X] (1 - P(X,Ti))
         *
         * Note: This is DATABASE-LEVEL probability, NOT transaction-level!
         */
        final double existentialProbability;

        /**
         * Constructor for PatternResult
         *
         * @param itemIDs Set of item IDs forming this pattern
         * @param expectedUtility Expected utility across entire database
         * @param existentialProbability Probability of existence across entire database
         */
        PatternResult(Set<Integer> itemIDs, double expectedUtility, double existentialProbability) {
            this.itemIDs = itemIDs;
            this.expectedUtility = expectedUtility;
            this.existentialProbability = existentialProbability;
        }

        @Override
        public int hashCode() {
            return itemIDs.hashCode();
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            PatternResult other = (PatternResult) obj;
            return itemIDs.equals(other.itemIDs);
        }

        @Override
        public String toString() {
            return "Pattern{" +
                   "items=" + itemIDs +
                   ", EU=" + String.format(Locale.US, "%.4f", expectedUtility) +
                   ", EP=" + String.format(Locale.US, "%.6f", existentialProbability) +
                   '}';
        }
    }

    /**
     * ItemData: Preprocessed item information for a single item in a transaction.
     * Used internally during suffix sum computation and utility list construction.
     */
    private static class ItemData {
        /** Item ID */
        final int item;

        /** Quantity of this item in the transaction */
        final int quantity;

        /** External profit/utility of this item (can be positive or negative) */
        final double profit;

        /** Computed utility = profit × quantity */
        final double utility;

        /** Natural logarithm of the transaction-level probability
         *  (stored in log-space to prevent underflow) */
        final double logProb;

        /**
         * Constructor for ItemData
         *
         * @param item Item ID
         * @param quantity Item quantity in transaction
         * @param profit External profit value (positive or negative)
         * @param logProb Log of transaction-level probability
         */
        ItemData(int item, int quantity, double profit, double logProb) {
            this.item = item;
            this.quantity = quantity;
            this.profit = profit;
            this.utility = profit * quantity;
            this.logProb = logProb;
        }
    }

    // ============================================================================
    // UPU-LIST: Unified Probability-Utility List Data Structure
    // ============================================================================

    /**
     * UPUList: Core data structure for efficient mining with pre-computed aggregates.
     *
     * This structure maintains transaction-level information for a SPECIFIC ITEMSET
     * and pre-computes all necessary statistics (EU, RU, EP) during construction
     * to enable O(1) access during the mining phase.
     *
     * Key Design Principles:
     * 1. Array-based storage for memory efficiency
     * 2. Single-pass aggregate computation during construction
     * 3. Immutable after construction (thread-safe for parallel access)
     */
    static class UPUList {

        /**
         * Element: Temporary structure used ONLY during UPUList construction.
         * Not exposed outside the UPUList class.
         */
        static class Element {
            /** Transaction ID where this itemset appears */
            final int tid;

            /** Utility of the itemset in this specific transaction */
            final double utility;

            /** Remaining utility: sum of utilities of items AFTER this itemset
             *  (used for upper bound pruning) */
            final double remaining;

            /** Log of probability that this itemset occurs in this transaction */
            final double logProbability;

            /**
             * Constructor for Element
             *
             * @param tid Transaction ID
             * @param utility Itemset utility in this transaction
             * @param remaining Remaining utility for pruning bounds
             * @param logProbability Log-space transaction-level probability
             */
            Element(int tid, double utility, double remaining, double logProbability) {
                this.tid = tid;
                this.utility = utility;
                this.remaining = remaining;
                this.logProbability = logProbability;
            }
        }

        // --- Core Fields: Itemset Identification ---

        /**
         * Set of item IDs that this UPUList represents.
         *
         * IMPORTANT CLARIFICATION (addressing professor's comment):
         * - This is NOT the same as PatternResult class!
         * - PatternResult is a FINAL OUTPUT with database-level statistics
         * - This Set<Integer> is just IDENTIFICATION of which items we're tracking
         *
         * Example: {1, 3} means this UPUList tracks information for itemset {1,3}
         */
        final Set<Integer> itemIDs;

        // --- Per-Transaction Data (Array Storage) ---

        /**
         * Array of transaction IDs where this itemset appears
         * Example: [1, 3, 5] means transactions T1, T3, T5 contain this itemset
         */
        final int[] tids;

        /**
         * Array of utilities for each transaction (parallel to tids array)
         * utilities[i] = utility of itemset in transaction tids[i]
         */
        final double[] utilities;

        /**
         * Array of remaining utilities (parallel to tids array)
         * remainings[i] = sum of positive utilities after this itemset in transaction tids[i]
         */
        final double[] remainings;

        /**
         * Array of log-probabilities (parallel to tids array)
         * logProbabilities[i] = log(P(itemset, tids[i]))
         */
        final double[] logProbabilities;

        /**
         * Number of transactions containing this itemset
         */
        final int size;

        // --- Upper Bound Values ---

        /**
         * Positive Transaction-Weighted Utility: upper bound for pruning
         * PTWU = sum of PTU values over all transactions containing this itemset
         */
        final double ptwu;

        // --- Pre-computed Aggregate Statistics (DATABASE-LEVEL) ---

        /**
         * Sum of Expected Utilities across all transactions
         * sumEU = Σ[all transactions] P(itemset,Ti) × u(itemset,Ti)
         *
         * This is the EXPECTED UTILITY at DATABASE LEVEL
         */
        private final double sumEU;

        /**
         * Sum of Expected Remaining Utilities across all transactions
         * sumRemaining = Σ[all transactions] P(itemset,Ti) × ru(itemset,Ti)
         *
         * Used for upper bound pruning: EU(any extension) ≤ sumEU + sumRemaining
         */
        private final double sumRemaining;

        /**
         * Existential Probability across the entire database
         * EP = 1 - Π[all transactions] (1 - P(itemset,Ti))
         *
         * This represents the probability that the itemset appears in AT LEAST ONE
         * transaction in the entire database.
         *
         * IMPORTANT: This is DATABASE-LEVEL probability, not transaction-level!
         */
        private final double existentialProbability;

        /**
         * Constructs a UPUList with array-based storage and pre-computed aggregates.
         *
         * This constructor performs a SINGLE PASS through the elements to:
         * 1. Copy data into efficient array storage
         * 2. Compute all aggregate statistics (sumEU, sumRemaining, EP)
         *
         * After construction, all fields are immutable (thread-safe).
         *
         * @param itemIDs Set of item IDs this list represents
         * @param elements List of per-transaction elements
         * @param ptwu Positive Transaction-Weighted Utility upper bound
         */
        UPUList(Set<Integer> itemIDs, List<Element> elements, double ptwu) {
            this.itemIDs = itemIDs;
            this.ptwu = ptwu;
            this.size = elements.size();

            // Allocate arrays for efficient storage
            this.tids = new int[size];
            this.utilities = new double[size];
            this.remainings = new double[size];
            this.logProbabilities = new double[size];

            // Single-pass: copy data AND compute aggregates simultaneously
            double tempSumEU = 0.0;
            double tempSumRemaining = 0.0;
            double logComplement = 0.0;  // For computing existential probability

            for (int i = 0; i < size; i++) {
                Element e = elements.get(i);

                // Store per-transaction data
                tids[i] = e.tid;
                utilities[i] = e.utility;
                remainings[i] = e.remaining;
                logProbabilities[i] = e.logProbability;

                // Compute transaction-level probability from log-space
                double prob = Math.exp(e.logProbability);

                // Accumulate expected utilities
                tempSumEU += e.utility * prob;
                tempSumRemaining += e.remaining * prob;

                // Accumulate for existential probability calculation (in log-space)
                // EP = 1 - Π(1 - pi) = 1 - exp(Σ log(1 - pi))
                if (e.logProbability > Math.log(1.0 - EPSILON)) {
                    // Probability ≈ 1, itemset certainly exists
                    logComplement = LOG_EPSILON;
                } else if (logComplement > LOG_EPSILON) {
                    // Use log1p for numerical stability
                    double log1MinusP = prob < 0.5 ?
                        Math.log1p(-prob) :  // More accurate for small prob
                        Math.log(1.0 - prob);
                    logComplement += log1MinusP;

                    // Prevent underflow
                    if (logComplement < LOG_EPSILON) {
                        logComplement = LOG_EPSILON;
                    }
                }
            }

            // Store final aggregate values
            this.sumEU = tempSumEU;
            this.sumRemaining = tempSumRemaining;
            this.existentialProbability = logComplement < LOG_EPSILON ?
                1.0 : 1.0 - Math.exp(logComplement);
        }

        /**
         * Returns the sum of expected utilities (DATABASE-LEVEL)
         *
         * @return Sum of P(itemset,Ti) × u(itemset,Ti) across all transactions
         */
        double getSumEU() {
            return sumEU;
        }

        /**
         * Returns the sum of expected remaining utilities (for pruning bounds)
         *
         * @return Sum of P(itemset,Ti) × ru(itemset,Ti) across all transactions
         */
        double getSumRemaining() {
            return sumRemaining;
        }

        /**
         * Returns the existential probability (DATABASE-LEVEL)
         *
         * @return Probability that itemset appears in at least one transaction
         */
        double getExistentialProbability() {
            return existentialProbability;
        }

        /**
         * Returns number of transactions containing this itemset
         *
         * @return Transaction count
         */
        int getSize() {
            return size;
        }

        /**
         * Checks if this itemset appears in any transaction
         *
         * @return true if size == 0 (no transactions contain this itemset)
         */
        boolean isEmpty() {
            return size == 0;
        }
    }

    // ============================================================================
    // LOCK-FREE TOP-K MANAGER
    // ============================================================================

    /**
     * TopKManager: Thread-safe manager for maintaining top-k patterns.
     *
     * Uses Compare-And-Swap (CAS) operations for lock-free concurrent updates,
     * enabling multiple worker threads to propose updates simultaneously without
     * blocking each other.
     *
     * Thread Safety Strategy:
     * - AtomicReferenceArray for lock-free slot updates
     * - AtomicInteger for lock-free size tracking
     * - AtomicReference for lock-free threshold updates
     * - Retry mechanism with exponential backoff for conflict resolution
     */
    private class TopKManager {
        /** Target number of patterns to maintain */
        private final int k;

        /** Lock-free array storing current top-k patterns */
        private final AtomicReferenceArray<PatternResult> topKArray;

        /** Current number of patterns stored (0 to k) */
        private final AtomicInteger size = new AtomicInteger(0);

        /** Current k-th best expected utility (threshold for admission) */
        private final AtomicReference<Double> threshold = new AtomicReference<>(0.0);

        /**
         * Constructor for TopKManager
         *
         * @param k Number of top patterns to maintain
         */
        TopKManager(int k) {
            this.k = k;
            this.topKArray = new AtomicReferenceArray<>(k);
        }

        /**
         * Attempts to add a pattern to top-k using lock-free CAS operations.
         *
         * Strategy:
         * 1. Try to fill empty slots first (fast path)
         * 2. Try to update existing duplicates with better utility
         * 3. Try to replace weakest pattern if array is full
         *
         * Uses retry mechanism with backoff to handle concurrent conflicts.
         *
         * @param itemIDs Set of item IDs forming this pattern
         * @param eu Expected utility of the pattern
         * @param ep Existential probability of the pattern
         * @return true if successfully added/updated, false otherwise
         */
        boolean tryAdd(Set<Integer> itemIDs, double eu, double ep) {
            final int MAX_RETRIES = 100;

            for (int attempt = 0; attempt < MAX_RETRIES; attempt++) {
                // Strategy 1: Try empty slots first (fastest)
                for (int i = 0; i < k; i++) {
                    if (topKArray.compareAndSet(i, null, new PatternResult(itemIDs, eu, ep))) {
                        size.incrementAndGet();
                        if (DEBUG_VERBOSE) {
                            System.err.printf("[VERBOSE] Added to empty slot %d: %s, EU=%.4f\n",
                                i, itemIDs, eu);
                        }
                        updateThreshold();
                        return true;
                    }
                }

                // Strategy 2: Update existing duplicate with better utility
                for (int i = 0; i < k; i++) {
                    PatternResult existing = topKArray.get(i);
                    if (existing != null && existing.itemIDs.equals(itemIDs) &&
                        eu > existing.expectedUtility + EPSILON) {
                        if (topKArray.compareAndSet(i, existing, new PatternResult(itemIDs, eu, ep))) {
                            if (DEBUG_VERBOSE) {
                                System.err.printf("[VERBOSE] Updated duplicate at slot %d: %s, EU %.4f -> %.4f\n",
                                    i, itemIDs, existing.expectedUtility, eu);
                            }
                            updateThreshold();
                            return true;
                        }
                    }
                }

                // Strategy 3: Replace weakest if array is full
                if (size.get() >= k) {
                    int weakestIndex = findMinEUIndex();
                    PatternResult weakest = topKArray.get(weakestIndex);

                    if (weakest != null && eu > weakest.expectedUtility + EPSILON) {
                        if (topKArray.compareAndSet(weakestIndex, weakest, new PatternResult(itemIDs, eu, ep))) {
                            if (DEBUG_VERBOSE) {
                                System.err.printf("[VERBOSE] Replaced weakest at slot %d: %s (EU=%.4f) with %s (EU=%.4f)\n",
                                    weakestIndex, weakest.itemIDs, weakest.expectedUtility, itemIDs, eu);
                            }
                            updateThreshold();
                            return true;
                        }
                    }
                }

                // Exponential backoff before retry
                if (attempt > 10) {
                    try {
                        Thread.sleep(1);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return false;
                    }
                }
            }

            if (DEBUG) {
                System.err.printf("[DEBUG] Failed to add after %d attempts: %s\n",
                    MAX_RETRIES, itemIDs);
            }
            return false;
        }

        /**
         * Finds the index of the pattern with minimum expected utility.
         *
         * @return Index of weakest pattern (0 to k-1)
         */
        private int findMinEUIndex() {
            int minIndex = 0;
            double minEU = Double.MAX_VALUE;

            for (int i = 0; i < k; i++) {
                PatternResult pattern = topKArray.get(i);
                if (pattern != null && pattern.expectedUtility < minEU) {
                    minEU = pattern.expectedUtility;
                    minIndex = i;
                }
            }

            return minIndex;
        }

        /**
         * Updates the admission threshold (k-th best EU) atomically.
         * Called after any successful add/update operation.
         */
        private void updateThreshold() {
            double newThreshold = Double.MAX_VALUE;

            for (int i = 0; i < k; i++) {
                PatternResult pattern = topKArray.get(i);
                if (pattern != null && pattern.expectedUtility < newThreshold) {
                    newThreshold = pattern.expectedUtility;
                }
            }

            // If array not full yet, threshold remains 0.0
            if (size.get() < k) {
                newThreshold = 0.0;
            }

            threshold.set(newThreshold);
        }

        /**
         * Returns current admission threshold (k-th best EU).
         *
         * @return Current threshold value
         */
        double getThreshold() {
            return threshold.get();
        }

        /**
         * Collects and returns all current top-k patterns, sorted by EU descending.
         *
         * @return List of top-k patterns sorted by expected utility (highest first)
         */
        List<PatternResult> getTopK() {
            List<PatternResult> result = new ArrayList<>();

            for (int i = 0; i < k; i++) {
                PatternResult pattern = topKArray.get(i);
                if (pattern != null) {
                    result.add(pattern);
                }
            }

            // Sort by expected utility descending
            result.sort((a, b) -> Double.compare(b.expectedUtility, a.expectedUtility));

            return result;
        }
    }

    // ============================================================================
    // PARALLEL MINING TASKS
    // ============================================================================

    /**
     * PrefixMiningTask: ForkJoin task for parallel prefix-based mining.
     *
     * Divides the search space by prefix items and processes each prefix
     * independently using work-stealing for dynamic load balancing.
     */
    private class PrefixMiningTask extends RecursiveAction {
        private final List<Integer> sortedItems;
        private final Map<Integer, UPUList> singleItemLists;
        private final int start;
        private final int end;

        /**
         * Constructor for PrefixMiningTask
         *
         * @param sortedItems List of items sorted by PTWU ascending
         * @param singleItemLists Map from item ID to its UPUList
         * @param start Start index (inclusive) in sortedItems
         * @param end End index (exclusive) in sortedItems
         */
        PrefixMiningTask(List<Integer> sortedItems, Map<Integer, UPUList> singleItemLists,
                        int start, int end) {
            this.sortedItems = sortedItems;
            this.singleItemLists = singleItemLists;
            this.start = start;
            this.end = end;
        }

        /**
         * Executes the mining task.
         *
         * For small ranges (≤ TASK_GRANULARITY), processes sequentially.
         * For larger ranges, splits into two subtasks and processes in parallel.
         */
        @Override
        protected void compute() {
            int size = end - start;

            if (size <= TASK_GRANULARITY) {
                // Base case: process sequentially
                for (int i = start; i < end; i++) {
                    processPrefix(i);
                }
            } else {
                // Recursive case: split into two subtasks
                int mid = start + size / 2;
                PrefixMiningTask left = new PrefixMiningTask(sortedItems, singleItemLists, start, mid);
                PrefixMiningTask right = new PrefixMiningTask(sortedItems, singleItemLists, mid, end);

                invokeAll(left, right);
            }
        }

        /**
         * Processes a single prefix item and explores all its extensions.
         *
         * @param index Index of the prefix item in sortedItems
         */
        private void processPrefix(int index) {
            int item = sortedItems.get(index);
            double currentThreshold = topKManager.getThreshold();

            UPUList itemList = singleItemLists.get(item);
            if (itemList == null || itemList.isEmpty()) {
                return;
            }

            // Early pruning: if PTWU < threshold, skip this prefix
            if (itemList.ptwu < currentThreshold - EPSILON) {
                if (DEBUG) {
                    System.err.printf("[DEBUG] Pruned prefix {%d}: PTWU=%.4f < threshold=%.4f\n",
                        item, itemList.ptwu, currentThreshold);
                }
                return;
            }

            // Collect extension candidates (items appearing after this one)
            List<Integer> extensions = new ArrayList<>();
            for (int j = index + 1; j < sortedItems.size(); j++) {
                int extItem = sortedItems.get(j);
                UPUList extList = singleItemLists.get(extItem);
                if (extList != null && extList.ptwu >= currentThreshold - EPSILON) {
                    extensions.add(extItem);
                }
            }

            // Recursively mine extensions
            if (!extensions.isEmpty()) {
                searchEnhanced(itemList, extensions, minPro, singleItemLists);
            }
        }
    }

    // ============================================================================
    // CORE MINING ALGORITHM
    // ============================================================================

    /** Item profit table: item ID -> external utility (profit) */
    private final Map<Integer, Double> itemProfits;

    /** Target number of top patterns to find */
    private final int k;

    /** Minimum existential probability threshold */
    private final double minPro;

    /** Top-k manager instance */
    private final TopKManager topKManager;

    /** Global item ranking: item ID -> rank (based on PTWU ascending) */
    private Map<Integer, Integer> itemToRank;

    /** ForkJoin pool for parallel execution */
    private final ForkJoinPool forkJoinPool;

    /**
     * Constructor for parallel mining algorithm
     *
     * @param itemProfits Map from item ID to external utility/profit
     * @param k Number of top patterns to find
     * @param minPro Minimum existential probability threshold
     * @param debug Enable debug output
     * @param debugVerbose Enable verbose debug output
     */
    public parallel(Map<Integer, Double> itemProfits, int k, double minPro,
                   boolean debug, boolean debugVerbose) {
        this.itemProfits = itemProfits;
        this.k = k;
        this.minPro = minPro;
        this.topKManager = new TopKManager(k);

        DEBUG = debug;
        DEBUG_VERBOSE = debugVerbose;

        // Use available processors for parallel execution
        int parallelism = Runtime.getRuntime().availableProcessors();
        this.forkJoinPool = new ForkJoinPool(parallelism);

        if (DEBUG) {
            System.err.printf("[DEBUG] Initialized with k=%d, minPro=%.2f, parallelism=%d\n",
                k, minPro, parallelism);
        }
    }

    /**
     * Streaming mining method: discovers top-k patterns from large databases.
     *
     * Processes transactions in batches to minimize memory footprint.
     * Maintains accumulated state across batches for consistent results.
     *
     * Mining Process (per batch):
     * 1. Process batch transactions with current state
     * 2. Accumulate statistics (PTWU, probabilities)
     * 3. Maintain intermediate UPU-lists
     * 4. Final pass: build complete UPU-lists and mine
     *
     * @param dbFile Path to database file
     * @param batchSize Number of transactions per batch (recommended: 10000-100000)
     * @return List of top-k patterns sorted by expected utility (descending)
     * @throws IOException if file reading fails
     */
    public List<PatternResult> mineStreaming(String dbFile, int batchSize) throws IOException {
        long startTime = System.nanoTime();

        if (DEBUG) {
            System.err.printf("[DEBUG] Starting streaming mining with batch size %d\n", batchSize);
        }

        // Phase 1: Stream through database to accumulate statistics
        if (DEBUG) System.err.println("[DEBUG] Phase 1: Streaming through database...");

        StreamingMiningState state = new StreamingMiningState();
        int batchCount = 0;

        try (StreamingDatabaseReader reader = new StreamingDatabaseReader(dbFile, batchSize)) {
            List<Transaction> batch;
            while (!(batch = reader.nextBatch()).isEmpty()) {
                batchCount++;

                // Compute batch statistics
                Map<Integer, Double> batchPTWU = computePTWU(batch);
                Map<Integer, Double> batchLogComplement = computeLogComplementBatch(batch);
                Map<Integer, List<UPUList.Element>> batchElements =
                    buildUtilityListsStreamingBatch(batch);

                // Accumulate state
                state.updateWithBatch(batchPTWU, batchLogComplement, batchElements);
                state.totalTransactions += batch.size();

                if (DEBUG) {
                    System.err.printf("[DEBUG] Processed batch %d (%d transactions, total: %d)\n",
                        batchCount, batch.size(), state.totalTransactions);
                }
            }

            if (DEBUG) {
                System.err.printf("[DEBUG] Finished streaming: %d batches, %d total transactions\n",
                    batchCount, state.totalTransactions);
            }
        }

        // Phase 2: Mine from accumulated state (same as batch mining)
        if (DEBUG) System.err.println("[DEBUG] Phase 2: Mining accumulated patterns...");

        // Filter by probability threshold
        Map<Integer, Double> itemProbabilities = state.computeItemProbabilities();
        Set<Integer> validItems = itemProbabilities.entrySet().stream()
            .filter(e -> e.getValue() >= minPro - EPSILON)
            .map(Map.Entry::getKey)
            .collect(Collectors.toSet());

        if (DEBUG) {
            System.err.printf("[DEBUG] Valid items (prob >= %.2f): %d / %d\n",
                minPro, validItems.size(), itemProbabilities.size());
        }

        // Rank items by PTWU ascending
        this.itemToRank = computeGlobalRanking(state.accumulatedPTWU, validItems);
        List<Integer> sortedItems = getSortedItemsByRank(validItems);

        // Build UPU-lists from accumulated elements
        Map<Integer, UPUList> singleItemLists = new HashMap<>();
        for (int item : sortedItems) {
            List<UPUList.Element> elements = state.allElements.get(item);
            if (elements != null && !elements.isEmpty()) {
                Set<Integer> itemSet = new HashSet<>();
                itemSet.add(item);
                UPUList upuList = new UPUList(itemSet, elements, state.accumulatedPTWU.get(item));

                // Check probability threshold
                if (upuList.getExistentialProbability() >= minPro - EPSILON) {
                    singleItemLists.put(item, upuList);
                }
            }
        }

        if (DEBUG) {
            System.err.printf("[DEBUG] Created %d single-item UPU-lists\n", singleItemLists.size());
        }

        // Phase 3: Check single items for top-k admission
        if (DEBUG) System.err.println("[DEBUG] Phase 3: Checking single items...");

        for (int item : sortedItems) {
            UPUList itemList = singleItemLists.get(item);
            if (itemList == null) continue;

            double eu = itemList.getSumEU();
            double ep = itemList.getExistentialProbability();

            if (ep >= minPro - EPSILON && eu >= topKManager.getThreshold() - EPSILON) {
                topKManager.tryAdd(itemList.itemIDs, eu, ep);
            }
        }

        // Phase 4: Parallel prefix mining
        if (DEBUG) System.err.println("[DEBUG] Phase 4: Parallel mining extensions...");

        PrefixMiningTask rootTask = new PrefixMiningTask(sortedItems, singleItemLists, 0, sortedItems.size());
        forkJoinPool.invoke(rootTask);

        long endTime = System.nanoTime();

        if (DEBUG) {
            double miningTimeMs = (endTime - startTime) / 1_000_000.0;
            System.err.printf("[DEBUG] Streaming mining completed in %.2f ms\n", miningTimeMs);
        }

        return topKManager.getTopK();
    }

    /**
     * Computes log-complement batch for probability accumulation.
     *
     * @param batch Batch of transactions
     * @return Map from item ID to accumulated log-complement
     */
    private Map<Integer, Double> computeLogComplementBatch(List<Transaction> batch) {
        Map<Integer, Double> itemLogComplement = new HashMap<>();

        for (Transaction trans : batch) {
            for (Map.Entry<Integer, Double> entry : trans.probabilities.entrySet()) {
                int item = entry.getKey();
                double prob = entry.getValue();

                double logComplement = prob < 0.5 ?
                    Math.log1p(-prob) : Math.log(1.0 - prob);

                itemLogComplement.merge(item, logComplement, Double::sum);
            }
        }

        return itemLogComplement;
    }

    /**
     * Builds utility lists from batch without requiring global ranking.
     *
     * This variant is used during streaming to accumulate elements
     * before global ranking is determined.
     *
     * @param batch Batch of transactions
     * @return Map from item ID to list of UPU-list elements
     */
    private Map<Integer, List<UPUList.Element>> buildUtilityListsStreamingBatch(
        List<Transaction> batch) {

        Map<Integer, List<UPUList.Element>> tempElements = new HashMap<>();

        for (Transaction rawTrans : batch) {
            // Extract items without sorting (we'll sort later after global ranking)
            List<ItemData> validItems = extractItemsWithoutRanking(rawTrans);
            if (validItems.isEmpty()) continue;

            // Compute suffix sums for remaining utility
            double[] suffixSums = computeSuffixSums(validItems);

            // Create elements for each item
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
     * Extracts valid items from transaction without requiring global ranking.
     *
     * @param rawTrans Transaction to process
     * @return List of ItemData in natural item order (not sorted by rank)
     */
    private List<ItemData> extractItemsWithoutRanking(Transaction rawTrans) {
        List<ItemData> validItems = new ArrayList<>();

        for (Map.Entry<Integer, Integer> entry : rawTrans.items.entrySet()) {
            Integer item = entry.getKey();
            Integer quantity = entry.getValue();

            Double profit = itemProfits.get(item);
            Double prob = rawTrans.probabilities.get(item);

            if (profit != null && prob != null && prob > 0) {
                double logProb = prob > 0 ? Math.log(prob) : LOG_EPSILON;
                validItems.add(new ItemData(item, quantity, profit, logProb));
            }
        }

        // Sort by item ID for consistency
        validItems.sort((a, b) -> Integer.compare(a.item, b.item));

        return validItems;
    }

    /**
     * Main mining method: discovers top-k high-utility patterns from uncertain database.
     *
     * Mining Process:
     * 1. Compute PTWU for all items
     * 2. Filter items by probability and rank by PTWU
     * 3. Build UPU-lists with suffix sum preprocessing
     * 4. Check single items against top-k
     * 5. Mine extensions in parallel using ForkJoin
     *
     * @param database List of transactions forming the uncertain database
     * @return List of top-k patterns sorted by expected utility (descending)
     */
    public List<PatternResult> mine(List<Transaction> database) {
        long startTime = System.nanoTime();

        // Phase 1: Compute PTWU and filter items
        if (DEBUG) System.err.println("[DEBUG] Phase 1: Computing PTWU...");

        Map<Integer, Double> itemPTWU = computePTWU(database);
        Map<Integer, Double> itemProbability = computeItemProbabilities(database);

        // Filter by probability threshold
        Set<Integer> validItems = itemProbability.entrySet().stream()
            .filter(e -> e.getValue() >= minPro - EPSILON)
            .map(Map.Entry::getKey)
            .collect(Collectors.toSet());

        if (DEBUG) {
            System.err.printf("[DEBUG] Valid items (prob >= %.2f): %d / %d\n",
                minPro, validItems.size(), itemProbability.size());
        }

        // Phase 2: Rank items by PTWU ascending
        if (DEBUG) System.err.println("[DEBUG] Phase 2: Ranking items by PTWU...");

        this.itemToRank = computeGlobalRanking(itemPTWU, validItems);
        List<Integer> sortedItems = getSortedItemsByRank(validItems);

        // Phase 3: Build UPU-lists with suffix sum preprocessing
        if (DEBUG) System.err.println("[DEBUG] Phase 3: Building UPU-lists...");

        Map<Integer, List<UPUList.Element>> tempElements = buildUtilityListsWithSuffixSum(database);
        Map<Integer, UPUList> singleItemLists = new HashMap<>();

        for (int item : sortedItems) {
            List<UPUList.Element> elements = tempElements.get(item);
            if (elements != null && !elements.isEmpty()) {
                Set<Integer> itemSet = new HashSet<>();
                itemSet.add(item);
                UPUList upuList = new UPUList(itemSet, elements, itemPTWU.get(item));

                // Check probability threshold at database level
                if (upuList.getExistentialProbability() >= minPro - EPSILON) {
                    singleItemLists.put(item, upuList);
                }
            }
        }

        if (DEBUG) {
            System.err.printf("[DEBUG] Created %d single-item UPU-lists\n", singleItemLists.size());
        }

        // Phase 4: Check single items for top-k admission
        if (DEBUG) System.err.println("[DEBUG] Phase 4: Checking single items...");

        for (int item : sortedItems) {
            UPUList itemList = singleItemLists.get(item);
            if (itemList == null) continue;

            double eu = itemList.getSumEU();
            double ep = itemList.getExistentialProbability();

            if (ep >= minPro - EPSILON && eu >= topKManager.getThreshold() - EPSILON) {
                topKManager.tryAdd(itemList.itemIDs, eu, ep);
            }
        }

        // Phase 5: Parallel prefix mining
        if (DEBUG) System.err.println("[DEBUG] Phase 5: Parallel mining...");

        PrefixMiningTask rootTask = new PrefixMiningTask(sortedItems, singleItemLists, 0, sortedItems.size());
        forkJoinPool.invoke(rootTask);

        long endTime = System.nanoTime();

        if (DEBUG) {
            double miningTimeMs = (endTime - startTime) / 1_000_000.0;
            System.err.printf("[DEBUG] Mining completed in %.2f ms\n", miningTimeMs);
        }

        return topKManager.getTopK();
    }

    /**
     * Recursive mining procedure for exploring itemset extensions.
     *
     * For each extension candidate:
     * 1. Join UPU-lists to form extended itemset
     * 2. Apply multi-tier pruning
     * 3. Check top-k admission if qualified
     * 4. Recursively mine further extensions
     *
     * @param prefixList UPU-list of current prefix itemset
     * @param extensions List of candidate extension items
     * @param minPro Minimum probability threshold
     * @param singleItemLists Map from item ID to UPUList for single items
     */
    private void searchEnhanced(UPUList prefixList, List<Integer> extensions, double minPro,
                               Map<Integer, UPUList> singleItemLists) {
        if (extensions.isEmpty()) return;

        for (int i = 0; i < extensions.size(); i++) {
            int extItem = extensions.get(i);
            double currentThreshold = topKManager.getThreshold();

            // Pre-join pruning: check PTWU of extension item
            UPUList extList = singleItemLists.get(extItem);
            if (extList == null || extList.ptwu < currentThreshold - EPSILON) {
                continue;
            }

            // Join UPU-lists
            UPUList joinedList = joinLists(prefixList, extList);
            if (joinedList == null || joinedList.isEmpty()) {
                continue;
            }

            // Multi-tier pruning
            double sumEU = joinedList.getSumEU();
            double sumRU = joinedList.getSumRemaining();
            double ep = joinedList.getExistentialProbability();

            // Tier 1: Probability pruning
            if (ep < minPro - EPSILON) {
                continue;
            }

            // Tier 2: PTWU pruning
            if (joinedList.ptwu < currentThreshold - EPSILON) {
                continue;
            }

            // Tier 3: EU + RU upper bound pruning
            if (sumEU + sumRU < currentThreshold - EPSILON) {
                continue;
            }

            // Check top-k admission
            if (sumEU >= currentThreshold - EPSILON && ep >= minPro - EPSILON) {
                topKManager.tryAdd(joinedList.itemIDs, sumEU, ep);
            }

            // Recursively mine further extensions
            if (i < extensions.size() - 1) {
                List<Integer> remainingExtensions = new ArrayList<>();
                for (int j = i + 1; j < extensions.size(); j++) {
                    int candItem = extensions.get(j);
                    UPUList candList = singleItemLists.get(candItem);
                    if (candList != null && candList.ptwu >= topKManager.getThreshold() - EPSILON) {
                        remainingExtensions.add(candItem);
                    }
                }

                if (!remainingExtensions.isEmpty()) {
                    searchEnhanced(joinedList, remainingExtensions, minPro, singleItemLists);
                }
            }
        }
    }

    /**
     * Joins two UPU-lists to form UPU-list for extended itemset.
     *
     * Uses two-pointer merge to efficiently find transactions containing both itemsets.
     *
     * @param list1 First UPU-list
     * @param list2 Second UPU-list
     * @return Joined UPU-list or null if no common transactions
     */
    private UPUList joinLists(UPUList list1, UPUList list2) {
        // Joined PTWU is minimum of two lists
        double joinedPTWU = Math.min(list1.ptwu, list2.ptwu);

        // Early pruning: if joined PTWU < threshold, no need to join
        if (joinedPTWU < topKManager.getThreshold() - EPSILON) {
            return null;
        }

        List<UPUList.Element> joinedElements = new ArrayList<>();
        int i = 0, j = 0;

        // Two-pointer merge on transaction IDs
        while (i < list1.size && j < list2.size) {
            int tid1 = list1.tids[i];
            int tid2 = list2.tids[j];

            if (tid1 == tid2) {
                // Found common transaction: compute joined values
                double utility = list1.utilities[i] + list2.utilities[j];
                double remaining = Math.min(list1.remainings[i], list2.remainings[j]);
                double logProb = list1.logProbabilities[i] + list2.logProbabilities[j];

                joinedElements.add(new UPUList.Element(tid1, utility, remaining, logProb));
                i++;
                j++;
            } else if (tid1 < tid2) {
                i++;
            } else {
                j++;
            }
        }

        if (joinedElements.isEmpty()) {
            return null;
        }

        // Create joined itemset ID
        Set<Integer> joinedItemIDs = new HashSet<>(list1.itemIDs);
        joinedItemIDs.addAll(list2.itemIDs);

        return new UPUList(joinedItemIDs, joinedElements, joinedPTWU);
    }

    // ============================================================================
    // UTILITY COMPUTATION METHODS
    // ============================================================================

    /**
     * Computes Positive Transaction-Weighted Utility (PTWU) for all items.
     *
     * PTWU(item) = sum of PTU values over all transactions containing that item
     * PTU(transaction) = sum of utilities of positive-profit items in transaction
     *
     * @param database Transaction database
     * @return Map from item ID to PTWU value
     */
    private Map<Integer, Double> computePTWU(List<Transaction> database) {
        Map<Integer, Double> itemPTWU = new HashMap<>();

        for (Transaction trans : database) {
            // Compute PTU: sum of positive utilities in this transaction
            double ptu = 0.0;
            for (Map.Entry<Integer, Integer> entry : trans.items.entrySet()) {
                int item = entry.getKey();
                int quantity = entry.getValue();
                Double profit = itemProfits.get(item);

                if (profit != null && profit > 0) {
                    ptu += profit * quantity;
                }
            }

            // Add PTU to PTWU of each item in transaction
            for (int item : trans.items.keySet()) {
                itemPTWU.merge(item, ptu, Double::sum);
            }
        }

        return itemPTWU;
    }

    /**
     * Computes database-level existential probability for each item.
     *
     * EP(item) = 1 - Π[Ti contains item] (1 - P(item,Ti))
     *
     * @param database Transaction database
     * @return Map from item ID to existential probability
     */
    private Map<Integer, Double> computeItemProbabilities(List<Transaction> database) {
        Map<Integer, Double> itemLogComplement = new HashMap<>();

        for (Transaction trans : database) {
            for (Map.Entry<Integer, Double> entry : trans.probabilities.entrySet()) {
                int item = entry.getKey();
                double prob = entry.getValue();

                double logComplement = prob < 0.5 ?
                    Math.log1p(-prob) : Math.log(1.0 - prob);

                itemLogComplement.merge(item, logComplement, Double::sum);
            }
        }

        Map<Integer, Double> itemProbability = new HashMap<>();
        for (Map.Entry<Integer, Double> entry : itemLogComplement.entrySet()) {
            int item = entry.getKey();
            double logComp = entry.getValue();
            double ep = logComp < LOG_EPSILON ? 1.0 : 1.0 - Math.exp(logComp);
            itemProbability.put(item, ep);
        }

        return itemProbability;
    }

    /**
     * Computes global item ranking based on PTWU values (ascending order).
     *
     * Lower PTWU → Lower rank (processed first in mining)
     * This ordering optimizes pruning effectiveness.
     *
     * @param itemPTWU Map from item ID to PTWU value
     * @param validItems Set of items passing probability threshold
     * @return Map from item ID to rank (0-based)
     */
    private Map<Integer, Integer> computeGlobalRanking(Map<Integer, Double> itemPTWU,
                                                       Set<Integer> validItems) {
        // Sort items by PTWU ascending
        List<Integer> rankedItems = validItems.stream()
            .filter(itemPTWU::containsKey)
            .sorted((a, b) -> {
                double ptwuA = itemPTWU.get(a);
                double ptwuB = itemPTWU.get(b);
                int cmp = Double.compare(ptwuA, ptwuB);
                return cmp != 0 ? cmp : Integer.compare(a, b);  // Tie-break by item ID
            })
            .collect(Collectors.toList());

        // Create rank map
        Map<Integer, Integer> itemToRank = new HashMap<>();
        for (int i = 0; i < rankedItems.size(); i++) {
            itemToRank.put(rankedItems.get(i), i);
        }

        return itemToRank;
    }

    /**
     * Builds utility lists using suffix sum preprocessing.
     *
     * Suffix sum eliminates O(T²) remaining utility calculation cost
     * by precomputing cumulative sums in O(T) time per transaction.
     *
     * @param rawDatabase Transaction database
     * @return Map from item ID to list of UPU-list elements
     */
    private Map<Integer, List<UPUList.Element>> buildUtilityListsWithSuffixSum(
        List<Transaction> rawDatabase) {

        Map<Integer, List<UPUList.Element>> tempElements = new HashMap<>();

        for (Transaction rawTrans : rawDatabase) {
            // Extract and sort items by global rank
            List<ItemData> validItems = extractAndSortValidItems(rawTrans);
            if (validItems.isEmpty()) continue;

            // Compute suffix sums for remaining utility
            double[] suffixSums = computeSuffixSums(validItems);

            // Create elements for each item
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
     * Extracts valid items from transaction and sorts by global rank.
     *
     * @param rawTrans Transaction to process
     * @return List of ItemData sorted by global ranking
     */
    private List<ItemData> extractAndSortValidItems(Transaction rawTrans) {
        List<ItemData> validItems = new ArrayList<>();

        for (Map.Entry<Integer, Integer> entry : rawTrans.items.entrySet()) {
            Integer item = entry.getKey();
            Integer quantity = entry.getValue();

            // Skip items not in global ranking
            if (!itemToRank.containsKey(item)) continue;

            Double profit = itemProfits.get(item);
            Double prob = rawTrans.probabilities.get(item);

            if (profit != null && prob != null && prob > 0) {
                double logProb = prob > 0 ? Math.log(prob) : LOG_EPSILON;
                validItems.add(new ItemData(item, quantity, profit, logProb));
            }
        }

        // Sort by global rank
        validItems.sort((a, b) -> {
            Integer rankA = itemToRank.get(a.item);
            Integer rankB = itemToRank.get(b.item);
            return rankA.compareTo(rankB);
        });

        return validItems;
    }

    /**
     * Computes suffix sums for remaining utility calculation.
     *
     * suffixSum[i] = sum of positive utilities from position i+1 to end
     *
     * This enables O(1) remaining utility lookup during UPU-list construction.
     *
     * @param validItems Sorted list of items in transaction
     * @return Array of suffix sums (same length as validItems)
     */
    private double[] computeSuffixSums(List<ItemData> validItems) {
        int n = validItems.size();
        double[] suffixSums = new double[n];

        suffixSums[n - 1] = 0.0;  // Last item has no remaining utility

        for (int i = n - 2; i >= 0; i--) {
            ItemData nextItem = validItems.get(i + 1);
            // Only count positive utilities
            double nextUtility = nextItem.profit > 0 ? nextItem.utility : 0.0;
            suffixSums[i] = suffixSums[i + 1] + nextUtility;
        }

        return suffixSums;
    }

    /**
     * Returns items sorted by global rank.
     *
     * @param items Set of item IDs
     * @return List sorted by rank (ascending)
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
    // STREAMING PROCESSING COMPONENTS
    // ============================================================================

    /**
     * TransactionBatchBuffer: Buffers transactions for efficient batch processing.
     *
     * This class manages a buffer of transactions and provides methods to:
     * - Add transactions incrementally
     * - Process when batch size is reached
     * - Flush remaining transactions on EOF
     */
    static class TransactionBatchBuffer {
        private final int batchSize;
        private final List<Transaction> buffer;

        /**
         * Constructor for TransactionBatchBuffer
         *
         * @param batchSize Number of transactions per batch
         */
        TransactionBatchBuffer(int batchSize) {
            this.batchSize = Math.max(1, batchSize);
            this.buffer = new ArrayList<>(batchSize);
        }

        /**
         * Adds a transaction to the buffer.
         *
         * @param transaction Transaction to add
         * @return true if batch is full and ready to process, false otherwise
         */
        synchronized boolean add(Transaction transaction) {
            buffer.add(transaction);
            return isFull();
        }

        /**
         * Checks if buffer is full.
         *
         * @return true if buffer size >= batchSize
         */
        boolean isFull() {
            return buffer.size() >= batchSize;
        }

        /**
         * Retrieves and clears current buffer contents.
         *
         * @return List of buffered transactions
         */
        synchronized List<Transaction> flush() {
            List<Transaction> result = new ArrayList<>(buffer);
            buffer.clear();
            return result;
        }

        /**
         * Returns current buffer size without clearing.
         *
         * @return Number of transactions in buffer
         */
        int size() {
            return buffer.size();
        }

        /**
         * Checks if buffer is empty.
         *
         * @return true if buffer contains no transactions
         */
        boolean isEmpty() {
            return buffer.isEmpty();
        }
    }

    /**
     * StreamingDatabaseReader: Reads transactions from file in streaming fashion.
     *
     * Yields transactions one at a time with buffering capability.
     * Memory efficient for large databases.
     */
    static class StreamingDatabaseReader implements Closeable {
        private final BufferedReader reader;
        private final int batchSize;
        private int tid = 1;
        private boolean closed = false;

        /**
         * Constructor for StreamingDatabaseReader
         *
         * @param filename Path to database file
         * @param batchSize Batch size for internal buffering
         * @throws IOException if file cannot be opened
         */
        StreamingDatabaseReader(String filename, int batchSize) throws IOException {
            this.reader = new BufferedReader(new FileReader(filename));
            this.batchSize = batchSize;
        }

        /**
         * Reads next batch of transactions from file.
         *
         * @return List of transactions (size up to batchSize), empty list if EOF
         * @throws IOException if reading fails
         */
        List<Transaction> nextBatch() throws IOException {
            if (closed) {
                return new ArrayList<>();
            }

            List<Transaction> batch = new ArrayList<>(batchSize);
            String line;
            int count = 0;

            while (count < batchSize && (line = reader.readLine()) != null) {
                Transaction trans = parseLine(line);
                if (trans != null) {
                    batch.add(trans);
                    count++;
                }
            }

            return batch;
        }

        /**
         * Parses a single transaction line from database file.
         *
         * Format: item:quantity:probability item:quantity:probability ...
         *
         * @param line Line to parse
         * @return Transaction object or null if line is empty/malformed
         */
        private Transaction parseLine(String line) {
            if (line == null || line.trim().isEmpty()) {
                return null;
            }

            Map<Integer, Integer> items = new HashMap<>();
            Map<Integer, Double> probabilities = new HashMap<>();

            String[] entries = line.trim().split("\\s+");
            for (String entry : entries) {
                String[] parts = entry.split(":");
                if (parts.length == 3) {
                    try {
                        int item = Integer.parseInt(parts[0]);
                        int quantity = Integer.parseInt(parts[1]);
                        double prob = Double.parseDouble(parts[2]);

                        items.put(item, quantity);
                        probabilities.put(item, prob);
                    } catch (NumberFormatException e) {
                        // Skip malformed entry
                    }
                }
            }

            if (!items.isEmpty()) {
                return new Transaction(tid++, items, probabilities);
            }
            return null;
        }

        /**
         * Returns total transactions read so far.
         *
         * @return Transaction count
         */
        int getTransactionCount() {
            return tid - 1;
        }

        @Override
        public void close() throws IOException {
            if (!closed) {
                closed = true;
                reader.close();
            }
        }
    }

    /**
     * StreamingMiningState: Maintains state across streaming batches.
     *
     * This class accumulates statistics from multiple batches and provides
     * consistent view of data seen so far.
     */
    private static class StreamingMiningState {
        /** Accumulated PTWU values across all batches */
        Map<Integer, Double> accumulatedPTWU = new HashMap<>();

        /** Accumulated log-complement for existential probability */
        Map<Integer, Double> accumulatedLogComplement = new HashMap<>();

        /** Total transactions processed */
        int totalTransactions = 0;

        /** All temporary elements collected so far */
        Map<Integer, List<UPUList.Element>> allElements = new HashMap<>();

        /**
         * Updates state with batch statistics.
         *
         * @param batchPTWU PTWU from this batch
         * @param batchLogComplementMap Log-complement map from this batch
         * @param batchElements Temporary elements from this batch
         */
        void updateWithBatch(Map<Integer, Double> batchPTWU,
                           Map<Integer, Double> batchLogComplementMap,
                           Map<Integer, List<UPUList.Element>> batchElements) {

            // Merge PTWU
            for (Map.Entry<Integer, Double> entry : batchPTWU.entrySet()) {
                accumulatedPTWU.merge(entry.getKey(), entry.getValue(), Double::sum);
            }

            // Merge log-complement
            for (Map.Entry<Integer, Double> entry : batchLogComplementMap.entrySet()) {
                accumulatedLogComplement.merge(entry.getKey(), entry.getValue(), Double::sum);
            }

            // Merge elements
            for (Map.Entry<Integer, List<UPUList.Element>> entry : batchElements.entrySet()) {
                allElements.computeIfAbsent(entry.getKey(), k -> new ArrayList<>())
                    .addAll(entry.getValue());
            }
        }

        /**
         * Computes item probabilities from accumulated log-complements.
         *
         * @return Map from item ID to existential probability
         */
        Map<Integer, Double> computeItemProbabilities() {
            Map<Integer, Double> result = new HashMap<>();
            for (Map.Entry<Integer, Double> entry : accumulatedLogComplement.entrySet()) {
                int item = entry.getKey();
                double logComp = entry.getValue();
                double ep = logComp < LOG_EPSILON ? 1.0 : 1.0 - Math.exp(logComp);
                result.put(item, ep);
            }
            return result;
        }
    }

    // ============================================================================
    // FILE I/O METHODS
    // ============================================================================

    /**
     * Reads profit table from file.
     *
     * Format: Each line contains "item_id profit"
     * Example: "1 5.0" means item 1 has profit 5.0
     *
     * @param filename Path to profit table file
     * @return Map from item ID to profit value
     * @throws IOException if file reading fails
     */
    static Map<Integer, Double> readProfitTable(String filename) throws IOException {
        Map<Integer, Double> profits = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.trim().split("\\s+");
                if (parts.length == 2) {
                    int item = Integer.parseInt(parts[0]);
                    double profit = Double.parseDouble(parts[1]);
                    profits.put(item, profit);
                }
            }
        }
        return profits;
    }

    /**
     * Reads uncertain database from file.
     *
     * Format: Each line is one transaction with format "item:quantity:probability"
     * Example: "1:2:0.8 3:5:0.9" means:
     * - Item 1 with quantity 2 and probability 0.8
     * - Item 3 with quantity 5 and probability 0.9
     *
     * @param filename Path to database file
     * @return List of transactions
     * @throws IOException if file reading fails
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

    // ============================================================================
    // MAIN ENTRY POINT
    // ============================================================================

    /**
     * Main method for running the algorithm.
     *
     * Usage (Batch mode): java parallel <database_file> <profit_file> <k> <min_probability> [--debug | --debug-verbose]
     * Usage (Streaming mode): java parallel <database_file> <profit_file> <k> <min_probability> --streaming [batch_size] [--debug | --debug-verbose]
     *
     * Arguments:
     * - database_file: Path to uncertain transaction database
     * - profit_file: Path to item profit table
     * - k: Number of top patterns to find
     * - min_probability: Minimum existential probability threshold (0-1)
     * - --streaming: Enable streaming mode (optional)
     * - batch_size: Batch size for streaming (default: 10000, optional)
     * - --debug: Enable debug output (optional)
     * - --debug-verbose: Enable verbose debug output (optional)
     *
     * Examples:
     *   Batch mode:
     *     java parallel data.txt profits.txt 10 0.1
     *     java parallel data.txt profits.txt 10 0.1 --debug
     *
     *   Streaming mode (default batch size 10000):
     *     java parallel data.txt profits.txt 10 0.1 --streaming
     *
     *   Streaming mode (custom batch size 50000):
     *     java parallel data.txt profits.txt 10 0.1 --streaming 50000
     *     java parallel data.txt profits.txt 10 0.1 --streaming 50000 --debug
     *
     * @param args Command-line arguments
     * @throws IOException if file operations fail
     */
    public static void main(String[] args) throws IOException {
        if (args.length < 4) {
            System.err.println("Usage (Batch): parallel <database_file> <profit_file> <k> <min_probability> [--debug | --debug-verbose]");
            System.err.println("Usage (Streaming): parallel <database_file> <profit_file> <k> <min_probability> --streaming [batch_size] [--debug | --debug-verbose]");
            System.exit(1);
        }

        String dbFile = args[0];
        String profitFile = args[1];
        int k = Integer.parseInt(args[2]);
        double minPro = Double.parseDouble(args[3]);

        boolean debug = false;
        boolean debugVerbose = false;
        boolean streamingMode = false;
        int batchSize = 10000;

        // Parse optional arguments
        for (int i = 4; i < args.length; i++) {
            if (args[i].equals("--debug")) {
                debug = true;
            } else if (args[i].equals("--debug-verbose")) {
                debugVerbose = true;
            } else if (args[i].equals("--streaming")) {
                streamingMode = true;
                // Check if next argument is a batch size
                if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                    try {
                        batchSize = Integer.parseInt(args[i + 1]);
                        i++;  // Skip this argument
                    } catch (NumberFormatException e) {
                        // Use default batch size
                    }
                }
            }
        }

        // Read profit table
        Map<Integer, Double> profits = readProfitTable(profitFile);

        // Create algorithm instance
        parallel algorithm = new parallel(profits, k, minPro, debug, debugVerbose);

        // Measure memory usage
        Runtime runtime = Runtime.getRuntime();
        System.gc();
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();

        // Execute mining
        long startTime = System.nanoTime();
        List<PatternResult> topK;

        if (streamingMode) {
            if (debug) {
                System.err.printf("[DEBUG] Running in STREAMING mode with batch size %d\n", batchSize);
            }
            topK = algorithm.mineStreaming(dbFile, batchSize);
        } else {
            if (debug) {
                System.err.println("[DEBUG] Running in BATCH mode");
            }
            List<Transaction> database = readDatabase(dbFile);
            topK = algorithm.mine(database);
        }

        long endTime = System.nanoTime();

        // Measure memory after mining
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
        System.out.printf("Mode: %s\n", streamingMode ? "STREAMING (batch size: " + batchSize + ")" : "BATCH");
        System.out.printf("Execution Time: %.2f ms\n", executionTimeMs);
        System.out.printf("Memory Used: %.2f MB\n", memoryUsedMB);
        System.out.println();

        // Print results
        System.out.println("=== Top-" + k + " High-Utility Patterns ===");
        int rank = 1;
        for (PatternResult pattern : topK) {
            System.out.printf("%d. %s\n", rank++, pattern);
        }
    }
}