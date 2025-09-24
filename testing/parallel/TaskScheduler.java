package testing.parallel;

import testing.core.*;
import testing.mining.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * TaskScheduler for parallel execution with ForkJoinPool
 */
public class TaskScheduler {
    private static final int PARALLEL_THRESHOLD = 30;
    private static final int TASK_GRANULARITY = 7;

    private final ForkJoinPool customThreadPool;
    private final AtomicLong peakMemoryUsage;
    private final MiningEngine miningEngine;

    public TaskScheduler(int numThreads, AtomicLong peakMemoryUsage, MiningEngine miningEngine) {
        this.customThreadPool = new ForkJoinPool(numThreads);
        this.peakMemoryUsage = peakMemoryUsage;
        this.miningEngine = miningEngine;
    }

    /**
     * ForkJoin task for parallel prefix mining (from ver5_2)
     */
    public class PrefixMiningTask extends RecursiveAction {
        private final List<Integer> sortedItems;
        private final Map<Integer, UtilityList> singleItemLists;
        private final int start, end;

        public PrefixMiningTask(List<Integer> sortedItems,
                        Map<Integer, UtilityList> singleItemLists,
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
            UtilityList ul = singleItemLists.get(item);

            if (ul == null) return;

            double currentThreshold = miningEngine.getTopKManager().getThreshold();
            Map<Integer, Double> itemRTWU = miningEngine.getItemRTWU();

            if (itemRTWU.get(item) < currentThreshold - MiningConstants.EPSILON) {
                miningEngine.getStatistics().incrementBranchPruned();
                return;
            }

            List<UtilityList> extensions = new ArrayList<>();
            for (int j = index + 1; j < sortedItems.size(); j++) {
                Integer extItem = sortedItems.get(j);
                UtilityList extUL = singleItemLists.get(extItem);

                if (extUL == null) continue;

                if (itemRTWU.get(extItem) < currentThreshold - MiningConstants.EPSILON) {
                    miningEngine.getStatistics().incrementRtwuPruned();
                    continue;
                }

                extensions.add(extUL);
            }

            if (!extensions.isEmpty()) {
                miningEngine.searchEnhanced(ul, extensions, singleItemLists);
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
    public class ExtensionSearchTask extends RecursiveAction {
        private final UtilityList prefix;
        private final List<UtilityList> extensions;
        private final Map<Integer, UtilityList> singleItemLists;
        private final int start, end;

        public ExtensionSearchTask(UtilityList prefix,
                           List<UtilityList> extensions,
                           Map<Integer, UtilityList> singleItemLists,
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
                double currentThreshold = miningEngine.getTopKManager().getThreshold();
                double minRTWU = Double.MAX_VALUE;
                for (int i = start; i < end; i++) {
                    UtilityList ext = extensions.get(i);
                    if (ext.rtwu < minRTWU) {
                        minRTWU = ext.rtwu;
                    }
                }

                if (minRTWU < currentThreshold - MiningConstants.EPSILON) {
                    miningEngine.getStatistics().incrementBulkBranchPruned();
                    miningEngine.getStatistics().addCandidatesPruned(size);
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
            UtilityList extension = extensions.get(index);

            double currentThreshold = miningEngine.getTopKManager().getThreshold();
            if (extension.rtwu < currentThreshold - MiningConstants.EPSILON) {
                miningEngine.getStatistics().incrementRtwuPruned();
                miningEngine.getStatistics().incrementCandidatesPruned();
                return;
            }

            UtilityList joined = miningEngine.join(prefix, extension);

            if (joined == null || joined.elements.isEmpty()) {
                return;
            }

            miningEngine.getStatistics().incrementUtilityListsCreated();
            miningEngine.getStatistics().incrementCandidatesGenerated();

            double threshold = miningEngine.getTopKManager().getThreshold();

            // Pruning Strategy 1: Existential probability
            if (joined.existentialProbability < miningEngine.getMinPro() - MiningConstants.EPSILON) {
                miningEngine.getStatistics().incrementEpPruned();
                miningEngine.getStatistics().incrementCandidatesPruned();
                return;
            }

            // Pruning Strategy 2: EU + remaining - NOW O(1) thanks to ver5_2 optimization!
            double sumEU = joined.getSumEU();        // O(1) - no computation needed!
            double sumRemaining = joined.getSumRemaining(); // O(1) - no computation needed!

            if (sumEU + sumRemaining < threshold - MiningConstants.EPSILON) {
                miningEngine.getStatistics().incrementEuPruned();
                miningEngine.getStatistics().incrementCandidatesPruned();
                return;
            }

            // Update top-k if qualified
            if (sumEU >= threshold - MiningConstants.EPSILON &&
                joined.existentialProbability >= miningEngine.getMinPro() - MiningConstants.EPSILON) {
                miningEngine.getTopKManager().tryAdd(joined.itemset, sumEU, joined.existentialProbability);
            }

            // Recursive search
            if (index < extensions.size() - 1) {
                List<UtilityList> newExtensions = new ArrayList<>();
                double currentThresholdForFilter = miningEngine.getTopKManager().getThreshold();

                for (int j = index + 1; j < extensions.size(); j++) {
                    UtilityList ext = extensions.get(j);

                    if (ext.rtwu >= currentThresholdForFilter - MiningConstants.EPSILON) {
                        newExtensions.add(ext);
                    } else {
                        miningEngine.getStatistics().incrementRtwuPruned();
                    }
                }

                if (!newExtensions.isEmpty()) {
                    miningEngine.searchEnhanced(joined, newExtensions, singleItemLists);
                }
            }
        }
    }

    public void executePrefixMining(List<Integer> sortedItems, Map<Integer, UtilityList> singleItemLists) {
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
                miningEngine.sequentialMining(sortedItems, singleItemLists);
            }
        } else {
            System.out.println("Using sequential processing for " + sortedItems.size() + " items");
            miningEngine.sequentialMining(sortedItems, singleItemLists);
        }
    }

    public boolean executeExtensionSearch(UtilityList prefix, List<UtilityList> viableExtensions,
                                         Map<Integer, UtilityList> singleItemLists) {
        if (viableExtensions.size() >= PARALLEL_THRESHOLD && ForkJoinTask.inForkJoinPool()) {
            ExtensionSearchTask task = new ExtensionSearchTask(
                prefix, viableExtensions, singleItemLists, 0, viableExtensions.size()
            );
            task.invoke();
            return true;
        }
        return false;
    }

    public void shutdown() {
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
    }

    public int getParallelism() {
        return customThreadPool.getParallelism();
    }

    public static int getParallelThreshold() {
        return PARALLEL_THRESHOLD;
    }
}