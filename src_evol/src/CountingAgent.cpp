// *******************************
// CTRNN
// *******************************

#include "CountingAgent.h"
#include "random.h"

// *******
// Control
// *******

// Init the agent
void CountingAgent::Set(int networksize, TVector<double> parameters)
{
	size = networksize;
	gain = 1.0; 
	foodsensorweights.SetBounds(1, size);
	foodsensorweights.FillContents(0.0);
	landmarksensorweights.SetBounds(1, size);
	landmarksensorweights.FillContents(0.0);
	othersensorweights.SetBounds(1, size);
	othersensorweights.FillContents(0.0);
	pos = 0.0;
	foodSensor = 0.0;
	landmarkSensor = 0.0;
	otherSensor = 0.0;

	// Instantiate the nervous systems
	NervousSystem.SetCircuitSize(size);
	int k = 1;
	// Time-constants
	for (int i = 1; i <= size; i++) {
		NervousSystem.SetNeuronTimeConstant(i,parameters(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= size; i++) {
		NervousSystem.SetNeuronBias(i,parameters(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= size; i++) {
		for (int j = 1; j <= size; j++) {
			NervousSystem.SetConnectionWeight(i,j,parameters(k));
			k++;
		}
	}
	// Food Sensor Weights
	for (int i = 1; i <= size; i++) {
		SetFoodSensorWeight(i,parameters(k));
		k++;
	}
	// Landmark Sensor Weights
	for (int i = 1; i <= size; i++) {
		SetLandmarkSensorWeight(i,parameters(k));
		k++;
	}
	// Other Sensor Weights
	for (int i = 1; i <= size; i++) {
		SetOtherSensorWeight(i,parameters(k));
		k++;
	}

	NervousSystem.RandomizeCircuitState(0.0,0.0);
}

// Reset the state of the agent
void CountingAgent::ResetPosition(double initpos)
{
	pos = initpos;
}

// Reset the state of the agent
void CountingAgent::ResetNeuralState()
{
	NervousSystem.RandomizeCircuitState(0.0,0.0);
	landmarkSensor = 0.0;
	foodSensor = 0.0;
	otherSensor = 0.0;
}

// Sense Landmarks
void CountingAgent::SenseLandmarks(double ln, TVector<double> pos_landmarks)
{
	double dist;
	for (int i = 1; i <= ln; i += 1){
		dist = fabs(pos_landmarks[i] - pos);
		if (dist < 5)
		{
			landmarkSensor = 1/(1 + exp(8 * (dist - 1)));		
		}
	}
}

// Sense 
void CountingAgent::SenseFood(double pos_food)
{
	double dist;
	dist = fabs(pos_food - pos);
	if (dist < 5)
	{
		foodSensor = 1/(1 + exp(8 * (dist - 1)));		
	}
}

// Sense 
void CountingAgent::SenseOther(double pos_other)
{
	double dist;
	dist = fabs(pos_other - pos);
	if (dist < 5)
	{
		otherSensor = 1/(1 + exp(8 * (dist - 1)));		
	}
}

// Step
void CountingAgent::Step(double StepSize)
{
	// Set sensors to external input
	for (int i = 1; i <= size; i++){
		NervousSystem.SetNeuronExternalInput(i, foodSensor*foodsensorweights[i] + landmarkSensor*landmarksensorweights[i] + otherSensor*othersensorweights[i]);
	}

	// Update the nervous system
	NervousSystem.EulerStep(StepSize);

	// Update the body position
	pos += StepSize * gain * (NervousSystem.NeuronOutput(2) - NervousSystem.NeuronOutput(1));
}
