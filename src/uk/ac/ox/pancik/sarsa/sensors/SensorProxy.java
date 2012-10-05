package uk.ac.ox.pancik.sarsa.sensors;

public class SensorProxy implements Sensor {

	private double value;

	@Override
	public double detect() {
		return this.value;
	}

	public void setValue(final double value) {
		this.value = value;
	}
}
