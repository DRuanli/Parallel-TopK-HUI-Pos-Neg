package testing.mining;

import testing.core.UtilityList;
import testing.parallel.TopKManager;
import java.util.*;

/**
 * Join operations for utility list mining
 * FIXED VERSION: Early termination removed to prevent missing candidates
 */
public class JoinOperator {
    private final TopKManager topKManager;
    private final MiningStatistics statistics;

    public JoinOperator(TopKManager topKManager, MiningStatistics statistics) {
        this.topKManager = topKManager;
        this.statistics = statistics;
    }

    /**
     * Join two utility-lists 
     * FIXED: Removed early termination that was causing missed candidates
     */
    public UtilityList join(UtilityList ul1, UtilityList ul2) {
        double joinedRTWU = Math.min(ul1.rtwu, ul2.rtwu);

        double currentThreshold = topKManager.getThreshold();
        if (joinedRTWU < currentThreshold - MiningConstants.EPSILON) {
            statistics.incrementRtwuPruned();
            return null;
        }

        // REMOVED: The problematic aggressive pruning
        // This was causing issues when threshold was high early in mining
        /*
        if (currentThreshold > 0 && joinedRTWU < currentThreshold * 0.1) {
            return null;
        }
        */

        int size1 = ul1.elements.size();
        int size2 = ul2.elements.size();

        if (size1 == 0 || size2 == 0) return null;

        // Optimized initial capacity estimation
        int estimatedCapacity = Math.min(size1, size2) / 3;
        estimatedCapacity = Math.max(estimatedCapacity, 4);
        estimatedCapacity = Math.min(estimatedCapacity, 32);

        List<UtilityList.Element> joinedElements = new ArrayList<>(estimatedCapacity);

        int i = 0, j = 0;
        // REMOVED: consecutiveMisses tracking that was causing early termination
        
        // Main join loop - now processes ALL elements without early termination
        while (i < size1 && j < size2) {
            UtilityList.Element e1 = ul1.elements.get(i);
            UtilityList.Element e2 = ul2.elements.get(j);

            if (e1.tid == e2.tid) {
                double newUtility = e1.utility + e2.utility;
                double newRemaining = Math.min(e1.remaining, e2.remaining);  // This is CORRECT
                double newLogProbability = e1.logProbability + e2.logProbability;

                // Check if probability is meaningful
                if (newLogProbability > MiningConstants.LOG_EPSILON + 1) {
                    joinedElements.add(new UtilityList.Element(
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
            
            // REMOVED: The problematic early termination check
            // The following block has been completely removed:
            /*
            if (consecutiveMisses > 50 && joinedElements.isEmpty() && (i + j) > 100) {
                return null;  // This was causing missed candidates!
            }
            */
        }

        // Optimize ArrayList capacity if significantly overallocated
        if (joinedElements instanceof ArrayList &&
            joinedElements.size() < estimatedCapacity / 3 &&
            joinedElements.size() < 100) {
            ((ArrayList<UtilityList.Element>) joinedElements).trimToSize();
        }

        // Create the joined itemset
        Set<Integer> newItemset = createSafeItemsetUnion(ul1.itemset, ul2.itemset);
        
        // Return the new utility list (or null if no joined elements)
        if (joinedElements.isEmpty()) {
            return null;
        }
        
        return new UtilityList(newItemset, joinedElements, joinedRTWU);
    }

    /**
     * Create union of two itemsets with optimized memory allocation
     */
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
}