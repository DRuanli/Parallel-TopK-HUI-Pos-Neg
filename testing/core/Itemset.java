package testing.core;

import java.util.Set;

/**
 * Itemset result class
 */
public class Itemset {
    public final Set<Integer> items;
    public final double expectedUtility;
    public final double probability;

    public Itemset(Set<Integer> items, double eu, double p) {
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