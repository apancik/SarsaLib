package uk.ac.ox.pancik.sarsa;

import java.util.Arrays;

import uk.ac.ox.pancik.utils.RandomSingleton;

import com.google.common.base.Preconditions;

//Neural Network with eligibility traces for SARSA
public class NeuralNetwork {
	
	// symmetric sigmoid
	public static double activationFunction(final double input) {
		Preconditions.checkArgument(!Double.isNaN(input));
		Preconditions.checkArgument(!Double.isInfinite(input));

		// 2*sigmoid(x,1)-1
		return 2.0 / (1.0 + Math.exp(-input)) - 1.0;
	}

	// symmetric sigmoid function after the first derivation
	// NOTE: the input to this function is value after the sigmoid
	public static double activationFunctionDerivation(final double sigmoidOfInput) {
		Preconditions.checkArgument(!Double.isNaN(sigmoidOfInput));
		Preconditions.checkArgument(!Double.isInfinite(sigmoidOfInput));
		
		return 0.5 * (1 + sigmoidOfInput) * (1 - sigmoidOfInput);
	}

	public static void setLearningRate(final double alpha) {
		NeuralNetwork.LEARNING_RATE = alpha;
	}

	public static void setTraceDecayRate(final double traceDecayRate) {
		NeuralNetwork.TRACE_DECAY_RATE = traceDecayRate;
	}

	boolean wasSelected = false;
	// ========================
	// STRUCTURE OF THE NETWORK
	// ========================
	private final double[] sensorOutputs;
	private final int[] neuronCounts;

	private final int layersCount;
	private final double inputs[][]; // inputs[layer][input]

	private final double outputs[][]; // outputs[layer][output] - after sigmoid

	private final double weights[][][]; // weights[layer][neuron][input]

	private final double gradient[][][]; // each weight has its gradient

	private final double eligibilityTraces[][][]; // each weight has its eligibility trace

	private double discountRate;

	// =========
	// CONSTANTS
	// =========
	// learning rate ALPHA (typically 0.09)
	private static double LEARNING_RATE = 0.01;

	// trace decay rate parameter LAMBDA (should be <= gamma)
	private static double TRACE_DECAY_RATE = 0.9;

	public static void setDiscountRate(double discountRate) {
		discountRate = discountRate;
	}

	public NeuralNetwork(final int[] numberOfHiddenNeurons, final int numberOfSensors, final double discountRate) {
		this.discountRate = discountRate;

		this.layersCount = numberOfHiddenNeurons.length + 1;

		this.neuronCounts = new int[this.layersCount];

		// Copy numbers of hidden neurons
		System.arraycopy(numberOfHiddenNeurons, 0, this.neuronCounts, 0, numberOfHiddenNeurons.length);

		this.neuronCounts[this.layersCount - 1] = 1; // last layer is 1 output neuron

		// ===============
		// PREPARE SENSORS
		// ===============
		this.sensorOutputs = new double[numberOfSensors + 1]; // + 1 for constant
		this.sensorOutputs[numberOfSensors] = 1; // last one is bias

		// ====================
		// CREATE OUTPUTS TABLE
		// ====================
		this.outputs = new double[this.layersCount][];

		for (int layer = 0; layer < this.layersCount - 1; layer++) {
			// at each level, every neuron has an activation value
			this.outputs[layer] = new double[this.neuronCounts[layer] + 1]; // +1 for dummy

			// add one dummy activation for every layer
			this.outputs[layer][this.neuronCounts[layer]] = 1; // last activation of dummy is 1
		}

		// last layer has only one activation
		this.outputs[this.layersCount - 1] = new double[this.neuronCounts[this.layersCount - 1]];

		// ===================
		// CREATE LAYER INPUTS
		// ===================
		this.inputs = new double[this.layersCount][];

		this.inputs[0] = this.sensorOutputs; // first level's inputs are wired to sensors

		// level's inputs are wired to previous level's outputs
		for (int layer = 1; layer < this.layersCount; layer++) {
			this.inputs[layer] = this.outputs[layer - 1];
		}

		// ===============================
		// CREATE GRADIENTS TABLE
		// ===============================
		this.gradient = new double[this.layersCount][][];

		for (int layer = 0; layer < this.layersCount; layer++) {
			this.gradient[layer] = new double[this.neuronCounts[layer]][this.inputs[layer].length];
		}

		// ===============================
		// CREATE ELIGIBILITY TRACES TABLE
		// ===============================
		this.eligibilityTraces = new double[this.layersCount][][];

		for (int layer = 0; layer < this.layersCount; layer++) {
			this.eligibilityTraces[layer] = new double[this.neuronCounts[layer]][this.inputs[layer].length];
		}

		// ====================
		// CREATE WEIGHT TABLES
		// ====================
		this.weights = new double[this.layersCount][][];

		// each neuron has a separate weight for every input
		for (int layer = 0; layer < this.layersCount; layer++) {
			this.weights[layer] = new double[this.neuronCounts[layer]][this.inputs[layer].length];
		}

		// randomize weights
		for (int layer = 0; layer < this.layersCount; layer++) {
			for (int neuron = 0; neuron < this.neuronCounts[layer]; neuron++) {
				for (int input = 0; input < this.inputs[layer].length; input++) {
					this.weights[layer][neuron][input] = RandomSingleton.nextDouble(-1, 1); // TODO extract to constant
				}
			}
		}
	}

	// TODO Make defensive copy
	public NeuralNetwork(final NeuralNetwork original) {
		// sensorOutputs.length - 1 for the bias
		this(Arrays.copyOf(original.neuronCounts, original.neuronCounts.length - 1), original.sensorOutputs.length - 1, original.discountRate);

		for (int layer = 0; layer < original.weights.length; layer++) {
			for (int neuron = 0; neuron < original.weights[layer].length; neuron++) {
				for (int input = 0; input < original.weights[layer][neuron].length; input++) {
					this.weights[layer][neuron][input] = original.weights[layer][neuron][input];
				}
			}
		}
	}

	// Update gradients by traversing through the neural network
	private void computeGradients(final int layer, final int neuron, final double value) {
		for (int input = 0; input < this.inputs[layer].length; input++) {
			// this.activationFunctionDerivation(this.outputs[layer][neuron]) is actually equivalent to g'(input) as the
			// param is expected to be after sigmoid
			// gradient_wji += g'(in_j) * a_i
			this.gradient[layer][neuron][input] += value * NeuralNetwork.activationFunctionDerivation(this.outputs[layer][neuron]) * this.inputs[layer][input];

			// only continue traversing for non-bias nodes and before getting to inputs
			if (input < this.inputs[layer].length - 1 && layer >= 1) {
				this.computeGradients(layer - 1, input, value * NeuralNetwork.activationFunctionDerivation(this.outputs[layer][neuron]) * this.weights[layer][neuron][input]);
			}
		}
	}

	// Propagate the inputs forward to compute the outputs
	public void forwardPropagate(final double[] observationsArray) {
		// =======================================
		// UPDATE SENSOR OUTPUTS FROM OBSERVATIONS
		// =======================================
		// sensorOutputs.length - 1 because of the bias input
		for (int observation = 0; observation < this.sensorOutputs.length - 1; observation++) {
			// sensor values should be in between -1 and 1
			this.sensorOutputs[observation] = NeuralNetwork.activationFunction(observationsArray[observation]);
		}

		// =========
		// PROPAGATE
		// =========
		for (int layer = 0; layer < this.layersCount; layer++) {
			for (int neuron = 0; neuron < this.neuronCounts[layer]; neuron++) {
				// Compute aggregated neuron input
				double aggregatedInput = 0;
				for (int input = 0; input < this.inputs[layer].length; input++) {
					aggregatedInput += this.weights[layer][neuron][input] * this.inputs[layer][input];
				}

				// Compute neuron activation
				this.outputs[layer][neuron] = NeuralNetwork.activationFunction(aggregatedInput);
			}
		}
	}

	public double getPredictedUtility() {
		return this.outputs[this.outputs.length - 1][0];
	}

	// Reset age, and eligibility traces. Each trial should start with this command
	public void resetEligibilities() {
		this.wasSelected = false;

		for (int layer = 0; layer < this.layersCount; layer++) {
			for (int neuron = 0; neuron < this.neuronCounts[layer]; neuron++) {
				for (int input = 0; input < this.inputs[layer].length; input++) {
					this.eligibilityTraces[layer][neuron][input] = 0;
				}
			}
		}
	}

	public void setDiscountFactor(final double gamma) {
		this.discountRate = gamma;
	}

	public void setForgettingRate(final double lambda) {
		NeuralNetwork.TRACE_DECAY_RATE = lambda;
	}

	public void updateEligibilities(final boolean selected) {
		if (selected) {
			this.wasSelected = true;
		}

		// =========================
		// RESET GRADIENTS
		// =========================
		if (this.wasSelected) {
			for (int layer = 0; layer < this.layersCount; layer++) {
				for (int neuron = 0; neuron < this.neuronCounts[layer]; neuron++) {
					for (int input = 0; input < this.inputs[layer].length; input++) {
						this.gradient[layer][neuron][input] = 0;
					}
				}
			}
		}

		// =========================
		// UPDATE GRADIENTS
		// =========================
		// NOTE: Gradients equals zero if the network was not selected so we can safely skip this if not selected
		if (selected) {
			this.computeGradients(this.layersCount - 1, 0, 1);
		}

		// =========================
		// UPDATE ELIGIBLITIY TRACES
		// =========================
		if (this.wasSelected) {
			for (int layer = 0; layer < this.layersCount; layer++) {
				for (int neuron = 0; neuron < this.neuronCounts[layer]; neuron++) {
					for (int input = 0; input < this.inputs[layer].length; input++) {
						this.eligibilityTraces[layer][neuron][input] *= this.discountRate * NeuralNetwork.TRACE_DECAY_RATE;
						this.eligibilityTraces[layer][neuron][input] += this.gradient[layer][neuron][input];
					}
				}
			}
		}
	}

	public void updateWeights(final double error) {
		// Naive implementation of Fast online Q(Lambda) - only perform update if action was selected in this episode
		if (this.wasSelected) {
			for (int layer = 0; layer < this.layersCount; layer++) {
				for (int neuron = 0; neuron < this.neuronCounts[layer]; neuron++) {
					for (int input = 0; input < this.inputs[layer].length; input++) {
						this.weights[layer][neuron][input] += NeuralNetwork.LEARNING_RATE * error * this.eligibilityTraces[layer][neuron][input];
					}
				}
			}
		}
	}
}
