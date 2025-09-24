package testing.mining;

import testing.core.UtilityList;
import testing.parallel.TopKManager;

/**
 * Pruning strategies for PTK-HUIM-U±
 */
public class PruningStrategy {
    private final TopKManager topKManager;
    private final double minPro;
    private final MiningStatistics statistics;

    public PruningStrategy(TopKManager topKManager, double minPro, MiningStatistics statistics) {
        this.topKManager = topKManager;
        this.minPro = minPro;
        this.statistics = statistics;
    }

    /**
     * Check if a utility list should be pruned based on RTWU
     */
    public boolean shouldPruneByRTWU(double rtwu) {
        double threshold = topKManager.getThreshold();
        if (rtwu < threshold - MiningConstants.EPSILON) {
            statistics.incrementRtwuPruned();
            return true;
        }
        return false;
    }

    /**
     * Check if a utility list should be pruned based on existential probability
     */
    public boolean shouldPruneByExistentialProbability(double existentialProbability) {
        if (existentialProbability < minPro - MiningConstants.EPSILON) {
            statistics.incrementEpPruned();
            statistics.incrementCandidatesPruned();
            return true;
        }
        return false;
    }

    /**
     * Check if a utility list should be pruned based on expected utility upper bound
     */
    public boolean shouldPruneByEUUpperBound(double sumEU, double sumRemaining) {
        double threshold = topKManager.getThreshold();
        if (sumEU + sumRemaining < threshold - MiningConstants.EPSILON) {
            statistics.incrementEuPruned();
            statistics.incrementCandidatesPruned();
            return true;
        }
        return false;
    }

    /**
     * Check for bulk pruning of multiple extensions
     */
    public boolean shouldBulkPrune(double maxPossibleRTWU) {
        double threshold = topKManager.getThreshold();
        if (maxPossibleRTWU < threshold - MiningConstants.EPSILON) {
            statistics.incrementBulkBranchPruned();
            return true;
        }
        return false;
    }

    /**
     * Check if an itemset qualifies for top-k
     */
    public boolean qualifiesForTopK(double sumEU, double existentialProbability) {
        double threshold = topKManager.getThreshold();
        return sumEU >= threshold - MiningConstants.EPSILON &&
               existentialProbability >= minPro - MiningConstants.EPSILON;
    }
}