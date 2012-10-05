package uk.ac.ox.pancik.utils;

// Generic 2D vectors factory
public class Vectors {
	public static Vector2D generateRadialVector(final double angle, final double step) {
		return new Vector2D(Math.cos(angle) * step, Math.sin(angle) * step);
	}

	public static Vector2D generateRandomVector(final double x1, final double y1, final double x2, final double y2) {
		return new Vector2D(RandomSingleton.nextDouble(x1, x2), RandomSingleton.nextDouble(y1, y2));
	}
}
