#include <iostream>
#include "TSearch.h"
#include "CountingAgent.h"
#include "CTRNN.h"
#include "random.h"
#include <iomanip> 
#include <sstream>
#include <filesystem>
#include <fstream>
#include <assert.h>
#include <vector>

#define PRINTTOFILE

// Task params
const int LN = 3;                   // Number of landmarks in the environment
const double StepSize = 0.1;

// EA params
const int POPSIZE = 96; //96;
const int GENS = 10000; //10000;
// const int GENS=10;
const double MUTVAR = 0.01; //0.05;
const double CROSSPROB = 0.5;
const double EXPECTED = 1.1;
const double ELITISM = 0.02;

// Nervous system params
const int N = 3;
const double WR = 10.0;     
const double SR = 10.0;     
const double BR = 10.0;     
const double TMIN = 1.0;
const double TMAX = 16.0;     

// Genotype size
int VectSize = 2 * (N*N + 5*N);  // Double the amount of parameters, one for Receiver, one for Signaler

// Environment variables
// Signaller Start Zone
const int SIGNALLERSTART = -5;
const int SIGNALLEREND = -1;
// Reciver start location
const int RECEIVERSTART = 0;
// landmark zone
const int LANDMARKZONESTART = 10;
const int LANDMARKZONEEND = LN * 10 + LANDMARKZONESTART;

// experiement parameters
const double RunDuration = 300.0;
const double TransDuration = 150.0;
const double HarshDuration = 20.0;
// this was originally 50 -> length of the arena
const double ArenaLength = LANDMARKZONEEND;      
const double mindist = 3.0;     
const double hardDurationMinDist = 1;

int call_count = 0;

double SignallersLocation;

std::string dir;

// ------------------------------------
// Genotype-Phenotype Mapping Functions
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen, int k)
{
    // Time-constants
    int x = 1;

    for (int i = 1; i <= N; i++) {
        phen(x) = MapSearchParameter(gen(k), TMIN, TMAX);
        k++;
        x++;
    }
    // Bias
    for (int i = 1; i <= N; i++) {
        phen(x) = MapSearchParameter(gen(k), -BR, BR);
        k++;
        x++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            phen(x) = MapSearchParameter(gen(k), -WR, WR);
            k++;
            x++;
        }
    }
    // Food Sensor Weights
    for (int i = 1; i <= N; i++) {
        phen(x) = MapSearchParameter(gen(k), -SR, SR);
        k++;
        x++;
    }
    // Landmark Sensor Weights
    for (int i = 1; i <= N; i++) {
        phen(x) = MapSearchParameter(gen(k), -SR, SR);
        k++;
        x++;
    }
    // Other Sensor Weights
    for (int i = 1; i <= N; i++) {
        phen(x) = MapSearchParameter(gen(k), -SR, SR);
        k++;
        x++;
    }    
}

// --------------------------------------------------
// Landmark Generation
// Leap Frog (See documentation for an explanation)
// --------------------------------------------------

TVector<double> genLandmarks_LeapFrog(RandomState &rs, TVector<double> &landmarkPositions){
    // determine how many landmarks we need
    landmarkPositions.SetBounds(1, LN);

    // generate the position of the first landmark
    double lenLandmarkZone = LANDMARKZONEEND - LANDMARKZONESTART;
    double maxSpacing = lenLandmarkZone / (double)(LN+1);
    // allow there to be a 20% variation in the landmarks position -> this could be increased
    double variation = 0.0;
    // generate the spaceing using this variation
    double spacing = maxSpacing * (1.0 + rs.UniformRandom(-variation, variation));

    double total_width = (LN-1) * spacing;

    // initalise the start position
    double max_start = LANDMARKZONEEND - total_width;
    double start = rs.UniformRandom(LANDMARKZONESTART, max_start);

    for (int i = 1;i <= LN; i ++){
        landmarkPositions[i] = start + (i-1)*spacing;
    }
   
    return landmarkPositions;
}

// variance of 1 means no variance
TVector<double> genLandmarks_Simple(double start_var, double space_var, TVector<double> &landmarkPositions){
    landmarkPositions.SetBounds(1, LN);
    double start = LANDMARKZONESTART +10; // 20
    double spacing = 10; // space them 10 appart

    for (int i = 1;i <= LN; i ++){
        landmarkPositions[i] = (start+start_var) + ((i-1)*(spacing+space_var));
    }

    return landmarkPositions;
}

void SignallerBehaviourTest(TVector<double>& genotype, int current_run){
    double food_location;
    double distance_food_receiver;
    // **** create the receiver ****
    // Map genotype to phenotype

    // Map genotype to phenotype
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentSignaller(N, phenotypeSignaller);
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // save this state
    // Save state
    TVector<double> savedStateSignaller, savedStateReceiver;
    savedStateSignaller.SetBounds(1,N);
    savedStateReceiver.SetBounds(1,N);

    // **** Generate landmarks (using leapfrog method) ****

    std::vector<double> positions = {20.0,30.0,40.0};

    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);
    for (int x = 1; x<= LN; x += 1){
        landmarkPositions[x] = positions[x-1];
    }

    // **** create the files to save the behaviour/env ***
    // the stages are separated into three files
    ofstream SignallerBehaviorFile, RecieverBehaviorFile;
    SignallerBehaviorFile.open( dir + "behavior_Signaller_signallerbehaviourtest_" + std::to_string(current_run) + ".dat");
    RecieverBehaviorFile.open( dir + "behavior_Reciever_signallerbehaviourtest_" + std::to_string(current_run) + ".dat");
    ofstream Stage3Fitness, FinalFitness;
    Stage3Fitness.open( dir + "fitness_signallerbehaviourtest_" + std::to_string(current_run) +".dat");
    FinalFitness.open( dir + "final_fitness_signallerbehaviourtest_" + std::to_string(current_run) +".dat");

    // stores the location of the landmarks and the food -> this is each line 
    ofstream LandmarkFile;
    LandmarkFile.open(dir + "landmark_location_signallerbehaviourtest_"+std::to_string(current_run)+".dat");

    // files to save the neuron state
    ofstream NeuronS1, NeuronS2, NeuronS3;
    NeuronS1.open(dir + "signaller_neuron1_signallerbehaviourtest_" + std::to_string(current_run)+".dat");
    NeuronS2.open(dir + "signaller_neuron2_signallerbehaviourtest_" + std::to_string(current_run)+".dat");
    NeuronS3.open(dir + "signaller_neuron3_signallerbehaviourtest_" + std::to_string(current_run)+".dat");
    ofstream NeuronR1, NeuronR2, NeuronR3;
    NeuronR1.open(dir + "receiver_neuron1_signallerbehaviourtest_" + std::to_string(current_run)+".dat");
    NeuronR2.open(dir + "receiver_neuron2_signallerbehaviourtest_" + std::to_string(current_run)+".dat");
    NeuronR3.open(dir + "receiver_neuron3_signallerbehaviourtest_" + std::to_string(current_run)+".dat");

    for (int env =1; env<=LN;env++){
        // set the food to be at the ith landmark location
        genLandmarks_Simple(0, 0, landmarkPositions);
        food_location = landmarkPositions[env];
        // for (int x = 1; x<= LN; x += 1){
        //     LandmarkFile << landmarkPositions[x] << " ";
        // }
        // LandmarkFile << food_location << " ";
        // LandmarkFile << endl;
        
        // Initialise the agent for this trial
        AgentReceiver.SetPosition(0);
        AgentReceiver.ResetSensors();
        // Initialise the agent for this trial
        // double location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
        AgentSignaller.SetPosition(0);
        AgentSignaller.ResetSensors();
        AgentReceiver.ResetNeuralState();
        AgentSignaller.ResetNeuralState();
        // ********* PHASE 1 ************

        for (double time=0; time < RunDuration; time += StepSize){
            // only let the Signaller move
            AgentSignaller.SenseFood(food_location);
            AgentSignaller.SenseLandmarks(LN, landmarkPositions);
            AgentSignaller.SenseOther(AgentReceiver.GetPosition());
            AgentSignaller.Step(StepSize);
        }

        // Reset
        AgentReceiver.SetPosition(0);
        AgentSignaller.SetPosition(0);
        AgentReceiver.ResetSensors();
        AgentSignaller.ResetSensors();

        // ********* PHASE 2/3 ************
        for (double time=0; time < RunDuration*2; time += StepSize){
            
            AgentReceiver.SenseOther(AgentSignaller.GetPosition());
            AgentSignaller.SenseOther(AgentReceiver.GetPosition());

            AgentReceiver.SenseLandmarks(LN, landmarkPositions);

            AgentSignaller.SenseFood(food_location);

            // Move both the agents
            AgentReceiver.Step(StepSize);
            // AgentReceiver.SetPosition(0);
            AgentSignaller.Step(StepSize);

            // clam the signallers position
            double pos = AgentSignaller.GetPosition();
            
            if (pos > LANDMARKZONESTART){
                AgentSignaller.SetPosition(LANDMARKZONESTART);
            }
        }

        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            savedStateSignaller[i] = AgentSignaller.NervousSystem.NeuronState(i);
        }
        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            savedStateReceiver[i] = AgentReceiver.NervousSystem.NeuronState(i);
        }

        // ******* TESTING *********

        for (double ref_var = -2.0; ref_var <= 2.0; ref_var += 2.0){
                for (double sep_var = -2.0; sep_var <= 2.0; sep_var += 2.0){
                // Reset
                AgentReceiver.SetPosition(0);
                AgentSignaller.SetPosition(0);
                AgentReceiver.ResetSensors();
                AgentSignaller.ResetSensors();

                genLandmarks_Simple(ref_var, sep_var, landmarkPositions);
                food_location = landmarkPositions[env];

                LandmarkFile << ref_var << " ";
                for (int x = 1; x<= LN; x += 1){
                    LandmarkFile << landmarkPositions[x] << " ";
                }
                LandmarkFile << food_location << " ";

                for (int i= 1; i<=N; i++){
                    AgentReceiver.NervousSystem.SetNeuronState(i, savedStateReceiver[i]);
                }
                for (int i= 1; i<=N; i++){
                    AgentSignaller.NervousSystem.SetNeuronState(i, savedStateSignaller[i]);
                }

                // ********* PHASE 2/3 ************

                // record the initial position
                SignallerBehaviorFile << AgentSignaller.GetPosition() << " ";
                RecieverBehaviorFile << AgentReceiver.GetPosition() << " ";
                NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
                NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
                NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";

                NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
                NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
                NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";

                double scoringTime = 0.0;
                double totalScore = 0.0;

                for (double time=0; time < RunDuration*2; time += StepSize){
                    
                    AgentReceiver.SenseOther(AgentSignaller.GetPosition());
                    AgentSignaller.SenseOther(AgentReceiver.GetPosition());

                    AgentReceiver.SenseLandmarks(LN, landmarkPositions);

                    AgentSignaller.SenseFood(food_location);

                    // Move both the agents
                    // AgentReceiver.SetPosition(0);
                    AgentReceiver.Step(StepSize);
                    AgentSignaller.Step(StepSize);

                    // clam the signallers position
                    double pos = AgentSignaller.GetPosition();
                    
                    if (pos > LANDMARKZONESTART){
                        AgentSignaller.SetPosition(LANDMARKZONESTART);
                    }
                    

                    if (time > RunDuration+TransDuration){
                        distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);

                        // if the distance is within a threshold set the score to be perfect
                        if (distance_food_receiver < 1 && time > HarshDuration+TransDuration+RunDuration){
                            distance_food_receiver = 0;
                        } else if (distance_food_receiver < mindist){
                            distance_food_receiver = 0;
                        }
                        totalScore += distance_food_receiver;
                        scoringTime += 1;
                        
                        // length of the landmark zone
                        double lengthZone = landmarkPositions[LN] - landmarkPositions[1];
                        
                        if (scoringTime > 0){
                            Stage3Fitness << 1 - ((totalScore/scoringTime)/lengthZone) << " ";
                        }
                    }

                    SignallerBehaviorFile << AgentSignaller.GetPosition() << " ";
                    RecieverBehaviorFile << AgentReceiver.GetPosition() << " ";
                    
                    // record the neural state at the end of each time step 
                    NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
                    NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
                    NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";
                    NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
                    NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
                    NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";
                }

                // length of the landmark zone
                double lengthZone = landmarkPositions[LN] - landmarkPositions[1];

                // END OF TRIAL
                // score at the end of this trial
                double fitness = 0;
                if (scoringTime > 0){
                    fitness = 1 - ((totalScore/scoringTime)/lengthZone);
                    if (fitness < 0.0){
                        fitness = 0.0;
                    }
                    FinalFitness << fitness << " ";
                }

                // new line for a new trial
                Stage3Fitness << endl;
                LandmarkFile << endl;
                FinalFitness << endl;

                SignallerBehaviorFile << endl;
                NeuronS1 << endl;
                NeuronS2 << endl;
                NeuronS3 << endl;

                RecieverBehaviorFile << endl;
                NeuronR1 << endl;
                NeuronR2 << endl;
                NeuronR3 << endl;        
            }
        }
    }
    // close the files for that env
    SignallerBehaviorFile.close();
    RecieverBehaviorFile.close();
    Stage3Fitness.close();
    LandmarkFile.close();
    NeuronS1.close();
    NeuronS2.close();
    NeuronS3.close();
    NeuronR1.close();
    NeuronR2.close();
    NeuronR3.close();
    FinalFitness.close();
}

// landmark 1 and 2 fixed varying the position of landmark 2

void Test1(TVector<double>& genotype, int current_run){
    double food_location;
    double distance_food_receiver;
    // **** create the receiver ****
    // Map genotype to phenotype

    // Map genotype to phenotype
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentSignaller(N, phenotypeSignaller);
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // save this state
    // Save state
    TVector<double> savedStateSignaller, savedStateReceiver;
    savedStateSignaller.SetBounds(1,N);
    savedStateReceiver.SetBounds(1,N);

    // **** Generate landmarks (using leapfrog method) ****

    std::vector<double> positions = {20.0,30.0,40.0};

    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);
    for (int x = 1; x<= LN; x += 1){
        landmarkPositions[x] = positions[x-1];
    }

    // **** create the files to save the behaviour/env ***
    // the stages are separated into three files
    ofstream SignallerBehaviorFile, RecieverBehaviorFile;
    SignallerBehaviorFile.open( dir + "behavior_Signaller_test1_" + std::to_string(current_run) + ".dat");
    RecieverBehaviorFile.open( dir + "behavior_Reciever_test1_" + std::to_string(current_run) + ".dat");
    ofstream Stage3Fitness, FinalFitness;
    Stage3Fitness.open( dir + "fitness_test1_" + std::to_string(current_run) +".dat");
    FinalFitness.open( dir + "final_fitness_test1_" + std::to_string(current_run) +".dat");

    // stores the location of the landmarks and the food -> this is each line 
    ofstream LandmarkFile;
    LandmarkFile.open(dir + "landmark_location_test1_"+std::to_string(current_run)+".dat");

    // files to save the neuron state
    ofstream NeuronS1, NeuronS2, NeuronS3;
    NeuronS1.open(dir + "signaller_neuron1_test1_" + std::to_string(current_run)+".dat");
    NeuronS2.open(dir + "signaller_neuron2_test1_" + std::to_string(current_run)+".dat");
    NeuronS3.open(dir + "signaller_neuron3_test1_" + std::to_string(current_run)+".dat");
    ofstream NeuronR1, NeuronR2, NeuronR3;
    NeuronR1.open(dir + "receiver_neuron1_test1_" + std::to_string(current_run)+".dat");
    NeuronR2.open(dir + "receiver_neuron2_test1_" + std::to_string(current_run)+".dat");
    NeuronR3.open(dir + "receiver_neuron3_test1_" + std::to_string(current_run)+".dat");


    // set the food to be at the ith landmark location
    food_location = landmarkPositions[2];
    for (int x = 1; x<= LN; x += 1){
        LandmarkFile << landmarkPositions[x] << " ";
    }
    LandmarkFile << food_location << " ";
    
    // Initialise the agent for this trial
    AgentReceiver.SetPosition(0);
    AgentReceiver.ResetSensors();
    // Initialise the agent for this trial
    // double location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
    AgentSignaller.SetPosition(0);
    AgentSignaller.ResetSensors();
    AgentReceiver.ResetNeuralState();
    AgentSignaller.ResetNeuralState();
    // ********* PHASE 1 ************

    for (double time=0; time < RunDuration; time += StepSize){
        // only let the Signaller move
        AgentSignaller.SenseFood(food_location);
        AgentSignaller.SenseLandmarks(LN, landmarkPositions);
        AgentSignaller.SenseOther(AgentReceiver.GetPosition());
        AgentSignaller.Step(StepSize);
    }

    // Reset
    AgentReceiver.SetPosition(0);
    AgentSignaller.SetPosition(0);
    AgentReceiver.ResetSensors();
    AgentSignaller.ResetSensors();

    // ********* PHASE 2/3 ************
    for (double time=0; time < RunDuration*2; time += StepSize){
        
        AgentReceiver.SenseOther(AgentSignaller.GetPosition());
        AgentSignaller.SenseOther(AgentReceiver.GetPosition());

        AgentReceiver.SenseLandmarks(LN, landmarkPositions);

        AgentSignaller.SenseFood(food_location);

        // Move both the agents
        AgentReceiver.Step(StepSize);
        AgentSignaller.Step(StepSize);

        // clam the signallers position
        double pos = AgentSignaller.GetPosition();
        
        if (pos > LANDMARKZONESTART){
            AgentSignaller.SetPosition(LANDMARKZONESTART);
        }
    }

    // Saved each of their neural states 
    for (int i = 1; i <= N; i++)
    {
        savedStateSignaller[i] = AgentSignaller.NervousSystem.NeuronState(i);
    }
    // Saved each of their neural states 
    for (int i = 1; i <= N; i++)
    {
        savedStateReceiver[i] = AgentReceiver.NervousSystem.NeuronState(i);
    }

    // ******* TESTING *********

    for (double ref_var = 0.0; ref_var <= 20.0; ref_var += 1.0){
        // Reset
        AgentReceiver.SetPosition(0);
        AgentSignaller.SetPosition(0);
        AgentReceiver.ResetSensors();
        AgentSignaller.ResetSensors();
        
        landmarkPositions[2] = 20 + ref_var;
        food_location = landmarkPositions[2];

        LandmarkFile << ref_var << " ";
        for (int x = 1; x<= LN; x += 1){
            LandmarkFile << landmarkPositions[x] << " ";
        }
        LandmarkFile << food_location << " ";

        for (int i= 1; i<=N; i++){
            AgentReceiver.NervousSystem.SetNeuronState(i, savedStateReceiver[i]);
        }
        for (int i= 1; i<=N; i++){
            AgentSignaller.NervousSystem.SetNeuronState(i, savedStateSignaller[i]);
        }

        // ********* PHASE 2/3 ************

        // record the initial position
        SignallerBehaviorFile << AgentSignaller.GetPosition() << " ";
        RecieverBehaviorFile << AgentReceiver.GetPosition() << " ";
        NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
        NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
        NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";

        NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
        NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
        NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";

        double scoringTime = 0.0;
        double totalScore = 0.0;

        for (double time=0; time < RunDuration*2; time += StepSize){
            
            AgentReceiver.SenseOther(AgentSignaller.GetPosition());
            AgentSignaller.SenseOther(AgentReceiver.GetPosition());

            AgentReceiver.SenseLandmarks(LN, landmarkPositions);

            AgentSignaller.SenseFood(food_location);

            // Move both the agents
            AgentReceiver.Step(StepSize);
            AgentSignaller.Step(StepSize);

            // clam the signallers position
            double pos = AgentSignaller.GetPosition();
            
            if (pos > LANDMARKZONESTART){
                AgentSignaller.SetPosition(LANDMARKZONESTART);
            }
            

            if (time > RunDuration+TransDuration){
                distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);

                // if the distance is within a threshold set the score to be perfect
                if (distance_food_receiver < 1 && time > HarshDuration+TransDuration+RunDuration){
                    distance_food_receiver = 0;
                } else if (distance_food_receiver < mindist){
                    distance_food_receiver = 0;
                }
                totalScore += distance_food_receiver;
                scoringTime += 1;
                
                // length of the landmark zone
                double lengthZone = landmarkPositions[LN] - landmarkPositions[1];
                
                if (scoringTime > 0){
                    Stage3Fitness << 1 - ((totalScore/scoringTime)/lengthZone) << " ";
                }
            }

            SignallerBehaviorFile << AgentSignaller.GetPosition() << " ";
            RecieverBehaviorFile << AgentReceiver.GetPosition() << " ";
            
            // record the neural state at the end of each time step 
            NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
            NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
            NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";
            NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
            NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
            NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";
        }

        // length of the landmark zone
        double lengthZone = landmarkPositions[LN] - landmarkPositions[1];

        // END OF TRIAL
        // score at the end of this trial
        double fitness = 0;
        if (scoringTime > 0){
            fitness = 1 - ((totalScore/scoringTime)/lengthZone);
            if (fitness < 0.0){
                fitness = 0.0;
            }
            FinalFitness << fitness << " ";
        }

        // new line for a new trial
        Stage3Fitness << endl;
        LandmarkFile << endl;
        FinalFitness << endl;

        SignallerBehaviorFile << endl;
        NeuronS1 << endl;
        NeuronS2 << endl;
        NeuronS3 << endl;

        RecieverBehaviorFile << endl;
        NeuronR1 << endl;
        NeuronR2 << endl;
        NeuronR3 << endl;        
    }

    // close the files for that env
    SignallerBehaviorFile.close();
    RecieverBehaviorFile.close();
    Stage3Fitness.close();
    LandmarkFile.close();
    NeuronS1.close();
    NeuronS2.close();
    NeuronS3.close();
    NeuronR1.close();
    NeuronR2.close();
    NeuronR3.close();
    FinalFitness.close();
}

// Fixed spacing, varying the start location

void Test2(TVector<double>& genotype, int current_run){
    double food_location;
    double distance_food_receiver;
    // **** create the receiver ****
    // Map genotype to phenotype

    // Map genotype to phenotype
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentSignaller(N, phenotypeSignaller);
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // save this state
    // Save state
    TVector<double> savedStateSignaller, savedStateReceiver;
    savedStateSignaller.SetBounds(1,N);
    savedStateReceiver.SetBounds(1,N);

    // **** Generate landmarks (using leapfrog method) ****

    std::vector<double> positions = {20.0,30.0,40.0};

    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);
    for (int x = 1; x<= LN; x += 1){
        landmarkPositions[x] = positions[x-1];
    }

    // **** create the files to save the behaviour/env ***
    // the stages are separated into three files
    ofstream SignallerBehaviorFile, RecieverBehaviorFile;
    SignallerBehaviorFile.open( dir + "behavior_Signaller_test2_" + std::to_string(current_run) + ".dat");
    RecieverBehaviorFile.open( dir + "behavior_Reciever_test2_" + std::to_string(current_run) + ".dat");
    ofstream Stage3Fitness, FinalFitness;
    Stage3Fitness.open( dir + "fitness_test2_" + std::to_string(current_run) +".dat");
    FinalFitness.open( dir + "final_fitness_test2_" + std::to_string(current_run) +".dat");

    // stores the location of the landmarks and the food -> this is each line 
    ofstream LandmarkFile;
    LandmarkFile.open(dir + "landmark_location_test2_"+std::to_string(current_run)+".dat");

    // files to save the neuron state
    ofstream NeuronS1, NeuronS2, NeuronS3;
    NeuronS1.open(dir + "signaller_neuron1_test2_" + std::to_string(current_run)+".dat");
    NeuronS2.open(dir + "signaller_neuron2_test2_" + std::to_string(current_run)+".dat");
    NeuronS3.open(dir + "signaller_neuron3_test2_" + std::to_string(current_run)+".dat");
    ofstream NeuronR1, NeuronR2, NeuronR3;
    NeuronR1.open(dir + "receiver_neuron1_test2_" + std::to_string(current_run)+".dat");
    NeuronR2.open(dir + "receiver_neuron2_test2_" + std::to_string(current_run)+".dat");
    NeuronR3.open(dir + "receiver_neuron3_test2_" + std::to_string(current_run)+".dat");


    // set the food to be at the ith landmark location
    food_location = landmarkPositions[1];
    for (int x = 1; x<= LN; x += 1){
        LandmarkFile << landmarkPositions[x] << " ";
    }
    LandmarkFile << food_location << " ";
    
    // Initialise the agent for this trial
    AgentReceiver.SetPosition(0);
    AgentReceiver.ResetSensors();
    // Initialise the agent for this trial
    // double location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
    AgentSignaller.SetPosition(0);
    AgentSignaller.ResetSensors();
    AgentReceiver.ResetNeuralState();
    AgentSignaller.ResetNeuralState();
    // ********* PHASE 1 ************

    for (double time=0; time < RunDuration; time += StepSize){
        // only let the Signaller move
        AgentSignaller.SenseFood(food_location);
        AgentSignaller.SenseLandmarks(LN, landmarkPositions);
        AgentSignaller.SenseOther(AgentReceiver.GetPosition());
        AgentSignaller.Step(StepSize);
    }

    // Reset
    AgentReceiver.SetPosition(0);
    AgentSignaller.SetPosition(0);
    AgentReceiver.ResetSensors();
    AgentSignaller.ResetSensors();

    // ********* PHASE 2/3 ************
    for (double time=0; time < RunDuration*2; time += StepSize){
        
        AgentReceiver.SenseOther(AgentSignaller.GetPosition());
        AgentSignaller.SenseOther(AgentReceiver.GetPosition());

        AgentReceiver.SenseLandmarks(LN, landmarkPositions);

        AgentSignaller.SenseFood(food_location);

        // Move both the agents
        AgentReceiver.Step(StepSize);
        AgentSignaller.Step(StepSize);

        // clam the signallers position
        double pos = AgentSignaller.GetPosition();
        
        if (pos > LANDMARKZONESTART){
            AgentSignaller.SetPosition(LANDMARKZONESTART);
        }
    }

    // Saved each of their neural states 
    for (int i = 1; i <= N; i++)
    {
        savedStateSignaller[i] = AgentSignaller.NervousSystem.NeuronState(i);
    }
    // Saved each of their neural states 
    for (int i = 1; i <= N; i++)
    {
        savedStateReceiver[i] = AgentReceiver.NervousSystem.NeuronState(i);
    }

    // ******* TESTING *********

    for (double ref_var = 0.0; ref_var <= 40.0; ref_var += 1.0){
        // Reset
        AgentReceiver.SetPosition(0);
        AgentSignaller.SetPosition(0);
        AgentReceiver.ResetSensors();
        AgentSignaller.ResetSensors();

        for (int x = 1; x<= LN; x += 1){
            landmarkPositions[x] = (ref_var) + (x-1)*10;
        }
        
        
        food_location = landmarkPositions[1];

        LandmarkFile << ref_var << " ";
        for (int x = 1; x<= LN; x += 1){
            LandmarkFile << landmarkPositions[x] << " ";
        }
        LandmarkFile << food_location << " ";

        for (int i= 1; i<=N; i++){
            AgentReceiver.NervousSystem.SetNeuronState(i, savedStateReceiver[i]);
        }
        for (int i= 1; i<=N; i++){
            AgentSignaller.NervousSystem.SetNeuronState(i, savedStateSignaller[i]);
        }

        // ********* PHASE 2/3 ************

        // record the initial position
        SignallerBehaviorFile << AgentSignaller.GetPosition() << " ";
        RecieverBehaviorFile << AgentReceiver.GetPosition() << " ";
        NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
        NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
        NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";

        NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
        NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
        NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";

        double scoringTime = 0.0;
        double totalScore = 0.0;

        for (double time=0; time < RunDuration*2; time += StepSize){
            
            AgentReceiver.SenseOther(AgentSignaller.GetPosition());
            AgentSignaller.SenseOther(AgentReceiver.GetPosition());

            AgentReceiver.SenseLandmarks(LN, landmarkPositions);

            AgentSignaller.SenseFood(food_location);

            // Move both the agents
            AgentReceiver.Step(StepSize);
            AgentSignaller.Step(StepSize);

            // clam the signallers position
            double pos = AgentSignaller.GetPosition();
            
            if (pos > LANDMARKZONESTART){
                AgentSignaller.SetPosition(LANDMARKZONESTART);
            }
            

            if (time > RunDuration+TransDuration){
                distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);

                // if the distance is within a threshold set the score to be perfect
                if (distance_food_receiver < 1 && time > HarshDuration+TransDuration+RunDuration){
                    distance_food_receiver = 0;
                } else if (distance_food_receiver < mindist){
                    distance_food_receiver = 0;
                }
                totalScore += distance_food_receiver;
                scoringTime += 1;
                
                // length of the landmark zone
                double lengthZone = landmarkPositions[LN] - landmarkPositions[1];
                
                if (scoringTime > 0){
                    Stage3Fitness << 1 - ((totalScore/scoringTime)/lengthZone) << " ";
                }
            }

            SignallerBehaviorFile << AgentSignaller.GetPosition() << " ";
            RecieverBehaviorFile << AgentReceiver.GetPosition() << " ";
            
            // record the neural state at the end of each time step 
            NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
            NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
            NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";
            NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
            NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
            NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";
        }

        // length of the landmark zone
        double lengthZone = landmarkPositions[LN] - landmarkPositions[1];

        // END OF TRIAL
        // score at the end of this trial
        double fitness = 0;
        if (scoringTime > 0){
            fitness = 1 - ((totalScore/scoringTime)/lengthZone);
            if (fitness < 0.0){
                fitness = 0.0;
            }
            FinalFitness << fitness << " ";
        }

        // new line for a new trial
        Stage3Fitness << endl;
        LandmarkFile << endl;
        FinalFitness << endl;

        SignallerBehaviorFile << endl;
        NeuronS1 << endl;
        NeuronS2 << endl;
        NeuronS3 << endl;

        RecieverBehaviorFile << endl;
        NeuronR1 << endl;
        NeuronR2 << endl;
        NeuronR3 << endl;        
    }

    // close the files for that env
    SignallerBehaviorFile.close();
    RecieverBehaviorFile.close();
    Stage3Fitness.close();
    LandmarkFile.close();
    NeuronS1.close();
    NeuronS2.close();
    NeuronS3.close();
    NeuronR1.close();
    NeuronR2.close();
    NeuronR3.close();
    FinalFitness.close();
}

// 

void Test3(TVector<double>& genotype, int current_run){
    double food_location;
    double distance_food_receiver;
    // **** create the receiver ****
    // Map genotype to phenotype

    // Map genotype to phenotype
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentSignaller(N, phenotypeSignaller);
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // save this state
    // Save state
    TVector<double> savedStateSignaller, savedStateReceiver;
    savedStateSignaller.SetBounds(1,N);
    savedStateReceiver.SetBounds(1,N);

    // **** Generate landmarks (using leapfrog method) ****

    std::vector<double> positions = {20.0,30.0,40.0};

    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);
    for (int x = 1; x<= LN; x += 1){
        landmarkPositions[x] = positions[x-1];
    }

    // **** create the files to save the behaviour/env ***
    // the stages are separated into three files
    ofstream SignallerBehaviorFile, RecieverBehaviorFile;
    SignallerBehaviorFile.open( dir + "behavior_Signaller_test3_" + std::to_string(current_run) + ".dat");
    RecieverBehaviorFile.open( dir + "behavior_Reciever_test3_" + std::to_string(current_run) + ".dat");
    ofstream Stage3Fitness, FinalFitness;
    Stage3Fitness.open( dir + "fitness_test3_" + std::to_string(current_run) +".dat");
    FinalFitness.open( dir + "final_fitness_test3_" + std::to_string(current_run) +".dat");

    // stores the location of the landmarks and the food -> this is each line 
    ofstream LandmarkFile;
    LandmarkFile.open(dir + "landmark_location_test3_"+std::to_string(current_run)+".dat");

    // files to save the neuron state
    ofstream NeuronS1, NeuronS2, NeuronS3;
    NeuronS1.open(dir + "signaller_neuron1_test3_" + std::to_string(current_run)+".dat");
    NeuronS2.open(dir + "signaller_neuron2_test3_" + std::to_string(current_run)+".dat");
    NeuronS3.open(dir + "signaller_neuron3_test3_" + std::to_string(current_run)+".dat");
    ofstream NeuronR1, NeuronR2, NeuronR3;
    NeuronR1.open(dir + "receiver_neuron1_test3_" + std::to_string(current_run)+".dat");
    NeuronR2.open(dir + "receiver_neuron2_test3_" + std::to_string(current_run)+".dat");
    NeuronR3.open(dir + "receiver_neuron3_test3_" + std::to_string(current_run)+".dat");


    // set the food to be at the ith landmark location
    food_location = landmarkPositions[2];
    for (int x = 1; x<= LN; x += 1){
        LandmarkFile << landmarkPositions[x] << " ";
    }
    LandmarkFile << food_location << " ";
    
    // Initialise the agent for this trial
    AgentReceiver.SetPosition(0);
    AgentReceiver.ResetSensors();
    // Initialise the agent for this trial
    // double location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
    AgentSignaller.SetPosition(0);
    AgentSignaller.ResetSensors();
    AgentReceiver.ResetNeuralState();
    AgentSignaller.ResetNeuralState();
    // ********* PHASE 1 ************

    for (double time=0; time < RunDuration; time += StepSize){
        // only let the Signaller move
        AgentSignaller.SenseFood(food_location);
        AgentSignaller.SenseLandmarks(LN, landmarkPositions);
        AgentSignaller.SenseOther(AgentReceiver.GetPosition());
        AgentSignaller.Step(StepSize);
    }

    // Reset
    AgentReceiver.SetPosition(0);
    AgentSignaller.SetPosition(0);
    AgentReceiver.ResetSensors();
    AgentSignaller.ResetSensors();

    // ********* PHASE 2/3 ************
    for (double time=0; time < RunDuration*2; time += StepSize){
        
        AgentReceiver.SenseOther(AgentSignaller.GetPosition());
        AgentSignaller.SenseOther(AgentReceiver.GetPosition());

        AgentReceiver.SenseLandmarks(LN, landmarkPositions);

        AgentSignaller.SenseFood(food_location);

        // Move both the agents
        AgentReceiver.Step(StepSize);
        AgentSignaller.Step(StepSize);

        // clam the signallers position
        double pos = AgentSignaller.GetPosition();
        
        if (pos > LANDMARKZONESTART){
            AgentSignaller.SetPosition(LANDMARKZONESTART);
        }
    }

    // Saved each of their neural states 
    for (int i = 1; i <= N; i++)
    {
        savedStateSignaller[i] = AgentSignaller.NervousSystem.NeuronState(i);
    }
    // Saved each of their neural states 
    for (int i = 1; i <= N; i++)
    {
        savedStateReceiver[i] = AgentReceiver.NervousSystem.NeuronState(i);
    }

    // ******* TESTING *********

    for (double ref_var = 0.0; ref_var <= 20.0; ref_var += 1.0){
        for (double spacing = 1.0; spacing <= 10; spacing += 1.0){
            // Reset
            AgentReceiver.SetPosition(0);
            AgentSignaller.SetPosition(0);
            AgentReceiver.ResetSensors();
            AgentSignaller.ResetSensors();

            for (int x = 1; x<= LN; x += 1){
                landmarkPositions[x] = (10 + ref_var) + (x-1)*spacing;
            }
            
            food_location = landmarkPositions[2];

            LandmarkFile << ref_var << " " << spacing << " ";
            for (int x = 1; x<= LN; x += 1){
                LandmarkFile << landmarkPositions[x] << " ";
            }
            LandmarkFile << food_location << " ";

            for (int i= 1; i<=N; i++){
                AgentReceiver.NervousSystem.SetNeuronState(i, savedStateReceiver[i]);
            }
            for (int i= 1; i<=N; i++){
                AgentSignaller.NervousSystem.SetNeuronState(i, savedStateSignaller[i]);
            }

            // ********* PHASE 2/3 ************

            // record the initial position
            SignallerBehaviorFile << AgentSignaller.GetPosition() << " ";
            RecieverBehaviorFile << AgentReceiver.GetPosition() << " ";

            NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
            NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
            NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";

            NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
            NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
            NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";

            double scoringTime = 0.0;
            double totalScore = 0.0;

            for (double time=0; time < RunDuration*2; time += StepSize){
                
                AgentReceiver.SenseOther(AgentSignaller.GetPosition());
                AgentSignaller.SenseOther(AgentReceiver.GetPosition());

                AgentReceiver.SenseLandmarks(LN, landmarkPositions);

                AgentSignaller.SenseFood(food_location);

                // Move both the agents
                AgentReceiver.Step(StepSize);
                AgentSignaller.Step(StepSize);

                // clam the signallers position
                double pos = AgentSignaller.GetPosition();
                
                if (pos > LANDMARKZONESTART){
                    AgentSignaller.SetPosition(LANDMARKZONESTART);
                }
                

                if (time > RunDuration+TransDuration){
                    distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);

                    // if the distance is within a threshold set the score to be perfect
                    if (distance_food_receiver < 1 && time > HarshDuration+TransDuration+RunDuration){
                        distance_food_receiver = 0;
                    } else if (distance_food_receiver < mindist){
                        distance_food_receiver = 0;
                    }
                    totalScore += distance_food_receiver;
                    scoringTime += 1;
                    
                    // length of the landmark zone
                    double lengthZone = landmarkPositions[LN] - landmarkPositions[1];
                    
                    if (scoringTime > 0){
                        Stage3Fitness << 1 - ((totalScore/scoringTime)/lengthZone) << " ";
                    }
                }

                SignallerBehaviorFile << AgentSignaller.GetPosition() << " ";
                RecieverBehaviorFile << AgentReceiver.GetPosition() << " ";
                
                // record the neural state at the end of each time step 
                NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
                NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
                NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";
                NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
                NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
                NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";
            }

            // length of the landmark zone
            double lengthZone = landmarkPositions[LN] - landmarkPositions[1];

            // END OF TRIAL
            // score at the end of this trial
            double fitness = 0;
            if (scoringTime > 0){
                fitness = 1 - ((totalScore/scoringTime)/lengthZone);
                if (fitness < 0.0){
                    fitness = 0.0;
                }
                FinalFitness << fitness << " ";
            }

            // new line for a new trial
            Stage3Fitness << endl;
            LandmarkFile << endl;
            FinalFitness << endl;

            SignallerBehaviorFile << endl;
            NeuronS1 << endl;
            NeuronS2 << endl;
            NeuronS3 << endl;

            RecieverBehaviorFile << endl;
            NeuronR1 << endl;
            NeuronR2 << endl;
            NeuronR3 << endl;        
        }
    } 

    // close the files for that env
    SignallerBehaviorFile.close();
    RecieverBehaviorFile.close();
    Stage3Fitness.close();
    LandmarkFile.close();
    NeuronS1.close();
    NeuronS2.close();
    NeuronS3.close();
    NeuronR1.close();
    NeuronR2.close();
    NeuronR3.close();
    FinalFitness.close();
}

// ================================================
// C. ADDITIONAL EVOLUTIONARY FUNCTIONS
// ================================================
int TerminationFunction(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
    if (BestPerf > 0.99) {
        return 1;
    }
    else {
        return 0;
    }
}

// ------------------------------------
// Display functions
// ------------------------------------
void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
    cout << Generation << " " << BestPerf << " " << AvgPerf << endl;
}

void ResultsDisplay(TSearch &s)
{
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();

    TVector<double> bestVector;
    ofstream BestIndividualFile;
    TVector<double> phenotypeS; 
    TVector<double> phenotypeR;
    phenotypeS.SetBounds(1, (int) (VectSize/2));
    phenotypeR.SetBounds(1, (int) (VectSize/2));

    // Save the genotype of the best individual
    bestVector = s.BestIndividual();
    BestIndividualFile.open( dir + "best_gen_" + current_run + ".dat");
    BestIndividualFile << bestVector << endl;
    BestIndividualFile.close();

    GenPhenMapping(bestVector, phenotypeS, 1);
    GenPhenMapping(bestVector, phenotypeR, (int)(N*N + 5*N + 1));

    // Show the Signaler
    BestIndividualFile.open( dir + "best_ns_s_" + current_run + ".dat" );
    CountingAgent AgentSignaler( N, phenotypeS);

    // Send to file
    BestIndividualFile << "Nervous System:" << endl;
    BestIndividualFile << AgentSignaler.NervousSystem << endl;
    BestIndividualFile << "Food Sensor Weights:" << endl;
    BestIndividualFile << AgentSignaler.foodsensorweights << "\n" << endl;
    BestIndividualFile << "Landmark Sensor Weights:" << endl;
    BestIndividualFile << AgentSignaler.landmarksensorweights << "\n" << endl;
    BestIndividualFile << "Other Sensor Weights:" << endl;
    BestIndividualFile << AgentSignaler.othersensorweights << "\n" << endl;
    BestIndividualFile.close();

    // Show the Reciever
    BestIndividualFile.open(dir + "best_ns_r_" + current_run + ".dat");
    CountingAgent AgentReceiver( N, phenotypeR);

    // Send to file
    BestIndividualFile << "Nervous System:" << endl;
    BestIndividualFile << AgentReceiver.NervousSystem << endl;
    BestIndividualFile << "Food Sensor Weights:" << endl;
    BestIndividualFile << AgentReceiver.foodsensorweights << "\n" << endl;
    BestIndividualFile << "Landmark Sensor Weights:" << endl;
    BestIndividualFile << AgentReceiver.landmarksensorweights << "\n" << endl;
    BestIndividualFile << "Other Sensor Weights:" << endl;
    BestIndividualFile << AgentReceiver.othersensorweights << "\n" << endl;
    BestIndividualFile.close();

}

std::string date_as_string(){
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);

    std::ostringstream oss;
    // oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    oss << put_time(&tm, "%Y-%m-%d");
    return oss.str();
}

std::string date_time_as_string(){
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    // oss << std::put_time(&tm, "%Y-%m-%d");
    return oss.str();
}

// ------------------------------------
// The main program
// ------------------------------------
// takes 2 arguments: run number and array index
int main (int argc, const char* argv[])
{
    // ######################
    // Setup
    // ######################
    // check that argv[1] has been provided6
    if (argc < 4){
        // send an error message to the terminal
        std::cerr << "Error: missing run or array index number.\n";
        return 1;
    }
    std::string slurm_job_id = argv[3];
    std::string batch_number = argv[2];
    std::string current_run = argv[1];

    // Define output home directory
    int v = 0;
    // std::string result_dir = "/user/work/yj23812/BeeCommunication/results/"+ date_as_string() +"/" + slurm_job_id;
    std::string result_dir = "/Users/katiepambakian/Documents/BSc Computer Science/Y3/Dissertation/BeeCommunication/results/"+ date_as_string() +"/7";

    // make the genotypes into TVector
    dir = result_dir +"/batch_"+ batch_number +"/network_"+ std::to_string(1) +"/";
    std::filesystem::create_directories(dir);
    
    TVector<double> genotype_tvector1;
    genotype_tvector1.SetBounds(1, VectSize);
    genotype_tvector1.InitializeContents(
        -0.770158, -0.845142, -0.278085, 0.847128, 0.799001, 0.62655,
        0.46465, -0.700673, 0.902649, -0.51525, -0.0110606, 0.558901,
        -0.796261, -0.0405864, -0.690388, -0.730021, 0.938148, 0.617213,
        0.780996, -0.00108515, 0.0477122, -0.415357, -0.0485985, 0.029821,
        0.528821, 0.420764, -0.859646, -0.992718, -0.0594891, -0.746238,
        0.317093, 0.644294, 0.909201, 0.839186, 0.1175, -0.00669196,
        0.0419098, 0.57718, -0.0754076, -0.901299, 0.739535, 0.387623,
        0.895011, 0.334062, -0.301657, -0.314407, 1.0, -0.344882
    );

    // Test1(genotype_tvector1, 1);
    // Test2(genotype_tvector1, 1);
    // Test3(genotype_tvector1, 1);
    SignallerBehaviourTest(genotype_tvector1, 1);

    // make the genotypes into TVector
    dir = result_dir +"/batch_"+ batch_number +"/network_"+ std::to_string(2) +"/";
    std::filesystem::create_directories(dir);

    TVector<double> genotype_tvector2;
    genotype_tvector2.SetBounds(1, VectSize);
    genotype_tvector2.InitializeContents(
        0.141966, 0.22054, 0.0444776, -0.479563, 0.460137, 0.0719391,
        -0.205893, -0.741223, 0.0381378, -0.428536, -0.0252837, 0.900702,
        0.562822, -0.690796, -0.20152, 0.133782, 1.0, -0.443747,
        -0.484532, 0.0929688, -0.996927, 0.477209, -0.0379486, -0.658938,
        0.817486, 0.659407, 0.470973, -0.873631, 0.973428, 0.27275,
        0.618336, 0.733516, 0.913901, -0.0591272, -0.0482305, -0.149136,
        0.668033, -0.648329, 0.269845, 0.552258, -0.806835, 0.736388,
        0.846889, 0.579995, -0.904701, -1.0, -0.716707, -0.562514
    );

    // Test1(genotype_tvector2, 2);
    // Test2(genotype_tvector2, 2);
    // Test3(genotype_tvector2, 2);

    // make the genotypes into TVector
    dir = result_dir +"/batch_"+ batch_number +"/network_"+ std::to_string(3) +"/";
    std::filesystem::create_directories(dir);

    TVector<double> genotype_tvector3;
    genotype_tvector3.SetBounds(1, VectSize);
    genotype_tvector3.InitializeContents(
        0.481203, -0.582154, 0.352546, -0.13744, 0.641622, -0.651404,
        -0.674004, -0.637786, -0.182521, -0.37291, 0.415165, 0.770273,
        -0.24151, -0.696718, 0.279091, -0.587634, -0.751296, -0.477206,
        -0.241941, -0.602675, 0.279286, -0.62711, 0.629964, 0.431228,
        0.423108, 0.0931809, 0.112731, -0.85589, 0.622068, -0.877075,
        0.483947, -0.1239, -0.908179, 0.634542, 0.6382, 0.66854,
        -0.885871, 0.863598, -0.248425, 0.702486, -0.545109, -0.434167,
        0.302903, 0.251184, -0.58741, 0.172916, 0.678134, -0.158857
    );

    // Test1(genotype_tvector3, 3);
    // Test2(genotype_tvector3, 3);
    // Test3(genotype_tvector3, 3);

    // make the genotypes into TVector
    dir = result_dir +"/batch_"+ batch_number +"/network_"+ std::to_string(4) +"/";
    std::filesystem::create_directories(dir);

    TVector<double> genotype_tvector4;
    genotype_tvector4.SetBounds(1, VectSize);
    genotype_tvector4.InitializeContents(
        -0.42568, 0.198445, 0.189926, -0.125431, 0.963222, -0.537706,
        -0.150958, -0.970835, 0.168992, 0.478119, 0.319075, -0.18683,
        0.536899, -0.353347, 0.529823, 0.675449, 0.903951, -0.353548,
        -0.74311, 0.555753, -0.610033, -0.437473, -0.6249, -0.0536158,
        0.753158, -0.213055, 0.574463, -0.118447, -0.0589971, -0.418272,
        -0.0780912, 0.618262, -0.67389, -0.517224, 0.577556, 0.487531,
        -0.464971, -0.263196, 0.406807, 0.316884, 0.939968, -0.260726,
        0.530354, -0.937995, 0.192227, -0.6636, -0.00512722, -0.825049
    );

    // Test1(genotype_tvector4, 4);
    // Test2(genotype_tvector4, 4);
    // Test3(genotype_tvector4, 4);

    return 0;
}
