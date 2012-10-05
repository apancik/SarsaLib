package uk.ac.ox.pancik.utils;

import com.google.common.base.Preconditions;

public class RandomSingleton {
	private static final java.util.Random random;

	static {
		random = new java.util.Random();
	}

	public static double nextDouble() {
		return random.nextDouble();
	}

	public static double nextDouble(final double max) {
		return nextDouble(0.0, max);
	}

	public static double nextDouble(final double from, final double to) {
		Preconditions.checkArgument(from < to, "From must be smaller than to");
		return from + (to - from) * random.nextDouble();
	}

	public static float nextFloat() {
		return random.nextFloat();
	}

	public static float nextFloat(final float from, final float to) {
		Preconditions.checkArgument(from < to, "From must be smaller than to");
		return from + (to - from) * random.nextFloat();
	}

	public static int nextInt(final int max) {
		return random.nextInt(max);
	}

	public static int nextInt(final int from, final int to) {
		Preconditions.checkArgument(from < to, "From must be smaller than from");
		return from + Math.abs(random.nextInt()) % (to - from);
	}
}
