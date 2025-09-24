package testing.mining;

/**
 * MERGED: Helper class from ver4_9 for efficient suffix sum processing
 */
public class ItemData {
    public final int item;
    public final int quantity;
    public final double profit;
    public final double utility;
    public final double logProb;

    public ItemData(int item, int quantity, double profit, double logProb) {
        this.item = item;
        this.quantity = quantity;
        this.profit = profit;
        this.utility = profit * quantity;
        this.logProb = logProb;
    }
}