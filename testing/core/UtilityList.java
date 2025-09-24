package testing.core;

import java.util.*;

/**
 * PRE-COMPUTED EnhancedUtilityList from ver5_2 (MAJOR OPTIMIZATION)
 * - Eliminates lazy evaluation overhead
 * - O(1) access for getSumEU() and getSumRemaining()
 * - Thread-safe immutable design
 * - Bug fixes from ver5_2 included
 */
public class UtilityList {
    private static final double EPSILON = 1e-10;
    private static final double LOG_EPSILON = -700;

    /**
     * Original Element class - unchanged for full compatibility
     */
    public static class Element {
        public final int tid;
        public final double utility;
        public final double remaining;
        public final double logProbability;

        public Element(int tid, double utility, double remaining, double logProbability) {
            this.tid = tid;
            this.utility = utility;
            this.remaining = remaining;
            this.logProbability = logProbability;
        }
    }

    // ============= ORIGINAL STORAGE (PRESERVED) =============
    public final Set<Integer> itemset;
    public final List<Element> elements;
    public final double rtwu;

    // ============= OPTIMIZATION: PRE-COMPUTED AGGREGATES =============
    private final double sumEU;                    // Pre-computed instead of cached
    private final double sumRemaining;             // Pre-computed instead of cached
    public final double existentialProbability;    // Pre-computed

    /**
     * OPTIMIZED CONSTRUCTOR - ver5_2's approach with bug fixes
     */
    public UtilityList(Set<Integer> itemset, List<Element> elements, double rtwu) {
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
    public double getSumEU() {
        return sumEU;  // Direct return, no computation needed
    }

    /**
     * O(1) access - pre-computed during construction
     * ELIMINATES lazy evaluation overhead from ver4_6/ver4_7
     */
    public double getSumRemaining() {
        return sumRemaining;  // Direct return, no computation needed
    }

    /**
     * Get existential probability - enhanced numerical stability from ver5_2
     */
    public double getExistentialProbability() {
        return existentialProbability;
    }

    /**
     * Get number of elements - O(1)
     */
    public int getSize() {
        return elements.size();
    }

    /**
     * Check if utility list is empty - O(1)
     */
    public boolean isEmpty() {
        return elements.isEmpty();
    }

    @Override
    public String toString() {
        return String.format("EnhancedUtilityList{itemset=%s, elements=%d, sumEU=%.2f, sumRemaining=%.2f, existProb=%.4f, rtwu=%.2f}",
            itemset, elements.size(), sumEU, sumRemaining, existentialProbability, rtwu);
    }
}