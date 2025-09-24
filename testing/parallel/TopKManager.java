package testing.parallel;

import testing.core.Itemset;
import java.util.*;
import java.util.concurrent.atomic.*;

/**
 * CAS-based TopKManager from ver5_2 (MAXIMUM PERFORMANCE)
 * - Lock-free operations using AtomicReferenceArray
 * - CAS performance tracking
 * - Optimal for parallel mining workloads
 */
public class TopKManager {
    private static final double EPSILON = 1e-10;
    private final int k;
    private final AtomicReferenceArray<Itemset> topKArray;
    private final AtomicInteger size = new AtomicInteger(0);
    private final AtomicReference<Double> threshold = new AtomicReference<>(0.0);

    // Performance tracking from ver5_2
    private final AtomicLong casRetries = new AtomicLong(0);
    private final AtomicLong successfulUpdates = new AtomicLong(0);

    private volatile double cachedThreshold = 0.0;

    public TopKManager(int k) {
        this.k = k;
        this.topKArray = new AtomicReferenceArray<>(k);
    }

    public boolean tryAdd(Set<Integer> items, double eu, double ep) {
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

    public double getThreshold() {
        return threshold.get();
    }

    public List<Itemset> getTopK() {
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
    public long getSuccessfulUpdates() {
        return successfulUpdates.get();
    }

    public long getCASRetries() {
        return casRetries.get();
    }
}