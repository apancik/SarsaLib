package uk.ac.ox.pancik.sarsa.actions;

public class ActionProxy implements Action {
	private boolean selected;

	@Override
	public void execute() {
		this.selected = true;
	}

	public boolean isSelected() {
		final boolean tmp = this.selected;
		this.selected = false;
		return tmp;
	}
}
