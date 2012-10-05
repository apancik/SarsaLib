package uk.ac.ox.pancik.utils;

import com.google.common.base.Objects;

// Generic immutable vector class
public class Vector2D {
	private final double x;
	private final double y;

	private static final double EPS = 1.0E-4;

	public Vector2D(final double x, final double y) {
		this.x = x;
		this.y = y;
	}

	public double distanceTo(final Vector2D to) {
		return this.minus(to).length();
	}

	public double dot(final Vector2D vector) {
		return this.x * vector.x + this.y * vector.y;
	}

	public double getX() {
		return this.x;
	}

	public double getY() {
		return this.y;
	}

	public double length() {
		return Math.sqrt(this.dot(this));
	}

	public Vector2D minus(final Vector2D subtrahend) {
		return new Vector2D(this.x - subtrahend.x, this.y - subtrahend.y);
	}

	public Vector2D normalize() {
		final double length = this.length();
		return this.times(Math.abs(length) < EPS ? Double.POSITIVE_INFINITY : 1.0 / length); // TODO Check
	}

	public Vector2D plus(final Vector2D addend) {
		return new Vector2D(this.x + addend.x, this.y + addend.y);
	}

	public Vector2D times(final double multiplier) {
		return new Vector2D(this.x * multiplier, this.y * multiplier);
	}

	@Override
	public String toString() {
		return Objects.toStringHelper(this.getClass()).add("X", this.x).add("y", this.y).toString();
	}
}
