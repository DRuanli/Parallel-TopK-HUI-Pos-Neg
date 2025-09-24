package testing.mining;

import java.util.concurrent.atomic.*;

/**
 * Thread-safe statistics tracking for mining operations
 */
public class MiningStatistics {
    private final AtomicLong candidatesGenerated = new AtomicLong(0);
    private final AtomicLong candidatesPruned = new AtomicLong(0);
    private final AtomicLong utilityListsCreated = new AtomicLong(0);
    private final AtomicLong euPruned = new AtomicLong(0);
    private final AtomicLong epPruned = new AtomicLong(0);
    private final AtomicLong rtwuPruned = new AtomicLong(0);
    private final AtomicLong branchPruned = new AtomicLong(0);
    private final AtomicLong bulkBranchPruned = new AtomicLong(0);

    public void incrementCandidatesGenerated() {
        candidatesGenerated.incrementAndGet();
    }

    public void incrementCandidatesPruned() {
        candidatesPruned.incrementAndGet();
    }

    public void addCandidatesPruned(long count) {
        candidatesPruned.addAndGet(count);
    }

    public void incrementUtilityListsCreated() {
        utilityListsCreated.incrementAndGet();
    }

    public void incrementEuPruned() {
        euPruned.incrementAndGet();
    }

    public void incrementEpPruned() {
        epPruned.incrementAndGet();
    }

    public void incrementRtwuPruned() {
        rtwuPruned.incrementAndGet();
    }

    public void incrementBranchPruned() {
        branchPruned.incrementAndGet();
    }

    public void incrementBulkBranchPruned() {
        bulkBranchPruned.incrementAndGet();
    }

    public long getCandidatesGenerated() {
        return candidatesGenerated.get();
    }

    public long getUtilityListsCreated() {
        return utilityListsCreated.get();
    }

    public long getRtwuPruned() {
        return rtwuPruned.get();
    }

    public long getBranchPruned() {
        return branchPruned.get();
    }

    public long getBulkBranchPruned() {
        return bulkBranchPruned.get();
    }

    public long getEuPruned() {
        return euPruned.get();
    }

    public long getEpPruned() {
        return epPruned.get();
    }

    public long getCandidatesPruned() {
        return candidatesPruned.get();
    }

    public void printStatistics() {
        System.out.println("Candidates generated: " + getCandidatesGenerated());
        System.out.println("Utility lists created: " + getUtilityListsCreated());
        System.out.println("Enhanced pruning statistics:");
        System.out.println("  - RTWU pruned: " + getRtwuPruned());
        System.out.println("  - Branches pruned: " + getBranchPruned());
        System.out.println("  - Bulk branches pruned: " + getBulkBranchPruned());
        System.out.println("  - EU+remaining pruned: " + getEuPruned());
        System.out.println("  - Existential probability pruned: " + getEpPruned());
        System.out.println("  - Total pruned: " + getCandidatesPruned());
    }
}