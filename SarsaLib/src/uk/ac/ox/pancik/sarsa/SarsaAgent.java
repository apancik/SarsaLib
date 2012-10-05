package uk.ac.ox.pancik.sarsa;

import java.util.List;

import uk.ac.ox.pancik.sarsa.sensors.Sensor;
import uk.ac.ox.pancik.utils.RandomSingleton;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;

public class SarsaAgent {

	// random action ratio is the epsilon in epsilon-greedy action selection
	// if equals 0, no random actions are selected
	private static double RANDOM_ACTIONS_RATIO;

	// boolean variable determining whether to use boltzmann selection or not
	private static boolean USING_BOLTZMANN;

	// boltzmann temperature regulates the extent of randomness of the selection
	// of the action with the largest Q-value in boltzmann selection
	private static double TEMPERATURE;

	// future discount rate - gamma
	private static final double GAMMA_DEFAULT = 0.9;
	private double futureDiscountRate = GAMMA_DEFAULT;
	
	public double modifyFutureDiscountRateWith(double delta) {
		futureDiscountRate += delta;
		
		for(NeuralNetwork nn : neuralNetworksArray)
			nn.setDiscountFactor(futureDiscountRate);
		
		return futureDiscountRate;
	}

	// number of discreet actions considered
	private final int numberOfActions;

	// array containing one neural network for each action from actionsArray
	private final NeuralNetwork[] neuralNetworksArray;

	// variables holding the state between steps
	private int selectedActionIndex;

	private double lastPredictedUtility;
	private int stepCounter;

	private final List<Sensor> sensors;

	public SarsaAgent(final List<Sensor> sensors, final int[] numberOfHiddenNeurons, final int numberOfActions) {
		this.sensors = sensors;

		this.numberOfActions = numberOfActions;

		this.neuralNetworksArray = new NeuralNetwork[numberOfActions];

		// for each action create a neural network approximating Q(state, action)
		for (int action = 0; action < numberOfActions; action++) {
			this.neuralNetworksArray[action] = new NeuralNetwork(numberOfHiddenNeurons, sensors.size(), futureDiscountRate);
		}

		System.out.println("Initialising Brain with [" + Ints.join(", ", numberOfHiddenNeurons) + "]");
	}

	// constructor creating a copy of another SarsaAgent
	public SarsaAgent(final SarsaAgent original) {
		// copy the parameters
		this.sensors = original.sensors;

		this.numberOfActions = original.numberOfActions;

		this.futureDiscountRate = original.futureDiscountRate;

		this.neuralNetworksArray = new NeuralNetwork[this.numberOfActions];

		for (int action = 0; action < this.numberOfActions; action++) {
			// create a copy of neural network representing the action
			this.neuralNetworksArray[action] = new NeuralNetwork(original.neuralNetworksArray[action]);
		}
	}

	public NeuralNetwork[] getNeuralNetworks() {
		return this.neuralNetworksArray;
	}

	public int getSelectedActionIndex() {
		return this.selectedActionIndex;
	}

	// TODO Extract to Selector class such as epsilonGreedySelector or
	// Selection method for the appropriate action based on the predicted Q values
	private int selectAction(final double[] utilitiesArray) {
		// =======================================================
		// EPSILON GREEDY - PICK RANDOM ACTION WITH EPSILON CHANCE
		// =======================================================
		if (RandomSingleton.nextDouble() < this.RANDOM_ACTIONS_RATIO) {
			return RandomSingleton.nextInt(this.numberOfActions);
		}

		// ==============================================
		// PICK THE ACTION BASED ON BOLTZMANN EXPLORATION
		// ==============================================
		if (this.USING_BOLTZMANN) {
			// actions with more utility are more probable
			final double boltzmannValues[] = new double[this.numberOfActions];

			double sum = 0;
			for (int action = 0; action < this.numberOfActions; action++) {
				final double weight = Math.exp(utilitiesArray[action] / this.TEMPERATURE);
				Preconditions.checkArgument(weight >= 0, "Boltzmann Weights must be positive");
				boltzmannValues[action] = weight;
				sum += weight;
				// TODO sum += boltzmannValues[action] = Math.exp(utilities[action] / this.temperature);
			}

			final double random = RandomSingleton.nextDouble(sum);
			
			sum = 0;
			for (int action = 0; action < this.numberOfActions; action++) {
				sum += boltzmannValues[action];

				
				
				if (random <= sum) {
					return action;
				}
			}

			return -1; // TODO just for test, this line should be never executed
		}

		// ========================
		// GREEDY POLICY - PICK MAX
		// ========================
		double maxPredictedUtility = Double.NEGATIVE_INFINITY;
		int maxIndex = -1;

		for (int action = 0; action < this.numberOfActions; action++) {
			if (utilitiesArray[action] > maxPredictedUtility) {
				maxPredictedUtility = utilitiesArray[action];
				maxIndex = action;
			}
		}

		return maxIndex;
	}

	public void setFutureDiscountRate(final double futureDiscountRate) {
		this.futureDiscountRate = futureDiscountRate;
		
		for(NeuralNetwork neuralNetwork : neuralNetworksArray){
			neuralNetwork.setDiscountRate(futureDiscountRate);
		}
	}

	public void setRandomActionsRatio(final double epsilon) {
		this.RANDOM_ACTIONS_RATIO = epsilon;
	}

	public void setTemperature(final double temperature) {
		//System.out.println(temperature);
		this.TEMPERATURE = temperature;
	}

	public void setUsingBoltzmann(final boolean usingBoltzmann) {
		if (usingBoltzmann) {
			System.out.println("Using boltzmann");
		} else {
			System.out.println("Not Using boltzmann");
		}
		this.USING_BOLTZMANN = usingBoltzmann;
	}

	// Prepare neural networks for new learning trial
	// should be called before every learning experience
	public void startNewTrial() {
		this.stepCounter = 0;
		this.lastPredictedUtility = 0;
		this.selectedActionIndex = -1;

		for (final NeuralNetwork neuralNetwork : this.neuralNetworksArray) {
			neuralNetwork.resetEligibilities();
		}
	}

	// TODO separate action selection and training so that you can just use the networks
	public void update(final double reward) {
		// ==============================================
		// FORWARD PROPAGATE INPUTS IN COMPONENT NETWORKS
		// ==============================================
		final double[] observationsArray = new double[this.sensors.size()];

		for (int index = 0; index < this.sensors.size(); index++) {
			observationsArray[index] = this.sensors.get(index).detect();
		}

		for (final NeuralNetwork neuralNetwork : this.neuralNetworksArray) {
			neuralNetwork.forwardPropagate(observationsArray);
		}

		// each neural network now contains predicted utility if the action was selected Q(state, action)
		// this can be obtained by calling method getPredictedUtility()s

		// =============
		// SELECT ACTION
		// =============

		// gather utilities into the array - this will make more sense when Selector is separate class
		final double[] utilitiesArray = new double[this.numberOfActions];
		for (int action = 0; action < this.numberOfActions; action++) {
			utilitiesArray[action] = this.neuralNetworksArray[action].getPredictedUtility();
		}

		//System.out.println(Doubles.join(",", utilitiesArray) + "\n ");
		
		// select action with the selection method based on the predicted utilities
		this.selectedActionIndex = this.selectAction(utilitiesArray);

		// ================
		// PERFORM TRAINING
		// ================

		// Q_t is the future utility of current state
		// Q_t = reward_t + discount * sum of discounted future returns
		// error = Q_t - my previous prediction of Q_t
		// error = (reward_t + discount * Q(s_t+1, a_t+1)) - Q(s_t, a_t)
		final double temporalDifferenceError = reward + this.futureDiscountRate * this.neuralNetworksArray[this.selectedActionIndex].getPredictedUtility() - this.lastPredictedUtility;
		
		for (int action = 0; action < this.numberOfActions; action++) {
			if (this.stepCounter > 0) {
				this.neuralNetworksArray[action].updateWeights(temporalDifferenceError);
			}

			this.neuralNetworksArray[action].updateEligibilities(action == this.selectedActionIndex);
		}

		this.lastPredictedUtility = this.neuralNetworksArray[this.selectedActionIndex].getPredictedUtility();
		this.stepCounter++;
	}
}
