package uk.ac.ox.pancik.sarsa.sensors;

public interface Sensor {
	// should be between -1 and 1
	public double detect();
}