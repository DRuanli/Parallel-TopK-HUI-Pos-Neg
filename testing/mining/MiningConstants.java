package testing.mining;

/**
 * Global constants for mining operations
 */
public class MiningConstants {
    public static final double EPSILON = 1e-10;
    public static final double LOG_EPSILON = -700;

    // Private constructor to prevent instantiation
    private MiningConstants() {
        throw new AssertionError("Cannot instantiate constants class");
    }
}