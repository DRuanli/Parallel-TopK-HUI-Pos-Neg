package testing.core;

import java.util.*;

/**
 * Transaction data structures for PTK-HUIM-U±
 */
public class Transaction {
    public final int tid;
    public final Map<Integer, Integer> items;
    public final Map<Integer, Double> probabilities;

    public Transaction(int tid, Map<Integer, Integer> items, Map<Integer, Double> probabilities) {
        this.tid = tid;
        this.items = items;
        this.probabilities = probabilities;
    }

    /**
     * Enhanced Transaction with efficient storage (from ver5_2)
     */
    public static class EnhancedTransaction {
        private static final double EPSILON = 1e-10;
        private static final double LOG_EPSILON = -700;

        public final int tid;
        public final int[] items;
        public final int[] quantities;
        public final double[] logProbabilities;
        public final double rtu;

        public EnhancedTransaction(int tid, Map<Integer, Integer> itemMap,
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
}