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

struct TrialData{
    std::vector<double> signaller_pos;
    std::vector<double> receiver_pos;
    std::vector<double> neuron_s1, neuron_s2, neuron_s3;
    std::vector<double> neuron_r1, neuron_r2, neuron_r3;
    std::vector<double> fitness_over_time;
    std::vector<double> landmark_positions;
    double food_location;
    double fitness;
};

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

// preparatory step -> prepare the network of the signaller to be able to find the correct food
double signaller_to_food(TVector<double> &genotype, RandomState &rs){
    // **** create the receiver ****
    // Map genotype to phenotype
    TVector<double> phenotypeSignaller, phenotypeReceiver;
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentSignaller(N, phenotypeSignaller);
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    AgentSignaller.SetPosition(0);
    AgentSignaller.ResetNeuralState();
    AgentSignaller.ResetSensors();

    AgentReceiver.SetPosition(0);
    AgentReceiver.ResetNeuralState();
    AgentReceiver.ResetSensors();

    // **** Generate landmarks (using leapfrog method) ****
    TVector<double> landmarkPositions;
    // calculating what the set other should be (value between 0.5 and 1.5)
    double step = 1.0/(LN-1);
    double start = 0.5;
    // trial variables 
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_food_receiver;

    for (int i=0; i<3; i++){
        genLandmarks_LeapFrog(rs, landmarkPositions);
        
        // set each of the landmarks to be the location of the food
        for (int env =1; env<=LN;env++){
            // set the food to be at the ith landmark location
            food_location = landmarkPositions[env];

            // Initialise the agent for this trial
            AgentSignaller.SetPosition(0);
            AgentSignaller.ResetNeuralState();
            AgentSignaller.ResetSensors();
            AgentReceiver.SetPosition(0);

            double scoringTime = 0.0;
            double totalScore = 0.0;

            // Let the Signaller Explore the enviornment (for 150) -> not scored
            for (double time=0; time < RunDuration; time += StepSize){
                AgentSignaller.SenseLandmarks(LN, landmarkPositions);
                AgentSignaller.SenseFood(food_location);
                AgentReceiver.SetPosition(0);
                AgentSignaller.SenseOther(AgentReceiver.GetPosition());
                AgentSignaller.Step(StepSize);

                if (time > TransDuration){
                    distance_food_receiver = fabs(AgentSignaller.GetPosition() - food_location);

                    // if the distance is within a threshold set the score to be perfect
                    if (distance_food_receiver < hardDurationMinDist && time > HarshDuration+TransDuration){
                        distance_food_receiver = 0;
                    } else if (distance_food_receiver < mindist){
                        distance_food_receiver = 0;
                    }
                    totalScore += distance_food_receiver;
                    scoringTime += 1;
                }
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
            }
            total_fitness += fitness;
            total_trials +=1;
        }
    }
    return (total_trials > 0) ? total_fitness / total_trials : 0.0;
}


double Fitness1(TVector<double> &genotype, RandomState &rs){
    // initalise their genotypes 
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    CountingAgent AgentSignaller(N, phenotypeSignaller);
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // save the state of the signaller and receiver
    TVector<double> savedStateSignaller, savedStateReceiver;
    savedStateSignaller.SetBounds(1,N); 
    savedStateReceiver.SetBounds(1, N);

    // Landmark Positions
    TVector<double> landmarkPositions;

    // trial variables 
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_food_receiver;

    // set each of the landmarks to be the location of the food
    for (int env =1; env<=LN;env++){

        genLandmarks_Simple(0, 0, landmarkPositions);

        // ******** SETUP **********
        // set the food to be at the ith landmark location
        food_location = landmarkPositions[env];

        // Initialise the agent for this trial
        AgentReceiver.SetPosition(0);
        AgentReceiver.ResetNeuralState();
        AgentReceiver.ResetSensors();

        // Initialise the agent for this trial
        // double location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
        AgentSignaller.SetPosition(0);
        AgentSignaller.ResetNeuralState();
        AgentSignaller.ResetSensors();

        // Phase 1 (Training Phase) -> signaller wondering

        for (double time=0; time < RunDuration; time += StepSize){
            // only let the Signaller move
            AgentSignaller.SenseFood(food_location);
            AgentSignaller.SenseLandmarks(LN, landmarkPositions);
            AgentSignaller.SenseOther(AgentReceiver.GetPosition());
            AgentSignaller.Step(StepSize);
        }
        
        // save the states

        // Reset
        AgentReceiver.SetPosition(0);
        AgentSignaller.SetPosition(0);
        AgentReceiver.ResetSensors();
        AgentSignaller.ResetSensors();

        // Phase 2/3 (Training + scoring Phase)

        for (double time=0; time < RunDuration*2; time += StepSize){
            // Receiver and Signaller only see each other 
            AgentReceiver.SenseOther(AgentSignaller.GetPosition());
            AgentSignaller.SenseOther(AgentReceiver.GetPosition());

            AgentReceiver.SenseLandmarks(LN, landmarkPositions);

            AgentSignaller.SenseFood(food_location);

            // Move both the agents
            AgentReceiver.Step(StepSize);
            AgentSignaller.Step(StepSize);

            // clamp the signallers position
            double pos = AgentSignaller.GetPosition();
            
            if (pos > LANDMARKZONESTART){
                AgentSignaller.SetPosition(LANDMARKZONESTART);
            }
        }

        for (int i = 1; i<=N;i ++){
            savedStateReceiver[i] = AgentReceiver.NervousSystem.NeuronState(i);
        }
        for (int i= 1; i<=N;i ++){
            savedStateSignaller[i] = AgentSignaller.NervousSystem.NeuronState(i);
        }

        for (double ref_var = 0.0; ref_var <= 0.0; ref_var += 1.0){
            for (double sep_var = 0.0; sep_var <= 0.0; sep_var += 1.0){

                genLandmarks_Simple(ref_var, sep_var, landmarkPositions);

                for (int i= 1; i<=N; i++){
                    AgentReceiver.NervousSystem.SetNeuronState(i, savedStateReceiver[i]);
                }
                for (int i= 1; i<=N; i++){
                    AgentSignaller.NervousSystem.SetNeuronState(i, savedStateSignaller[i]);
                }

                double scoringTime = 0.0;
                double totalScore = 0.0;
                
                for (double time=0; time < RunDuration*2; time += StepSize){
                    // Receiver and Signaller only see each other 
                    AgentReceiver.SenseOther(AgentSignaller.GetPosition());
                    AgentSignaller.SenseOther(AgentReceiver.GetPosition());

                    AgentReceiver.SenseLandmarks(LN, landmarkPositions);

                    AgentSignaller.SenseFood(food_location);

                    // Move both the agents
                    AgentReceiver.Step(StepSize);
                    AgentSignaller.Step(StepSize);

                    // clamp the signallers position
                    double pos = AgentSignaller.GetPosition();
                    
                    if (pos > LANDMARKZONESTART){
                        AgentSignaller.SetPosition(LANDMARKZONESTART);
                    }

                    if (time > RunDuration+TransDuration){
                        distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);

                        // if the distance is within a threshold set the score to be perfect
                        if (distance_food_receiver < hardDurationMinDist && time > HarshDuration+TransDuration+RunDuration){
                            distance_food_receiver = 0;
                        } else if (distance_food_receiver < mindist){
                            distance_food_receiver = 0;
                        }
                        totalScore += distance_food_receiver;
                        scoringTime += 1;
                    }
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
                }
                total_fitness += fitness;
                total_trials +=1;
            }
        }
    }
    return (total_trials > 0) ? (total_fitness / total_trials) : 0.0;
}

// ---------------------------------------------------------
// stage 1: Evolution of receiver
// only looking at the distance between the receiver and the 
// flower/food
// no signaller in the task
// the sensor of the receiver is set to correspond to which
// of the landmarks the receiver should go to
// ---------------------------------------------------------

double Fitness2(TVector<double> &genotype, RandomState &rs){
    // initalise their genotypes 
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    CountingAgent AgentSignaller(N, phenotypeSignaller);
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // save the state of the signaller and receiver
    TVector<double> savedStateSignaller, savedStateReceiver;
    savedStateSignaller.SetBounds(1,N); 
    savedStateReceiver.SetBounds(1, N);

    // Landmark Positions
    TVector<double> landmarkPositions;

    // trial variables 
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_food_receiver;

    // set each of the landmarks to be the location of the food
    for (int env =1; env<=LN;env++){

        // ******** SETUP **********
        genLandmarks_Simple(0, 0, landmarkPositions);

        // set the food to be at the ith landmark location
        food_location = landmarkPositions[env];

        // Initialise the agent for this trial
        AgentReceiver.SetPosition(0);
        AgentReceiver.ResetNeuralState();
        AgentReceiver.ResetSensors();

        // Initialise the agent for this trial
        // double location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
        AgentSignaller.SetPosition(0);
        AgentSignaller.ResetNeuralState();
        AgentSignaller.ResetSensors();

        // Phase 1 (Training Phase) -> signaller wondering

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

        // Phase 2/3 (Training + scoring Phase)
        
        for (double time=0; time < RunDuration*2; time += StepSize){
            // Receiver and Signaller only see each other 
            AgentReceiver.SenseOther(AgentSignaller.GetPosition());
            AgentSignaller.SenseOther(AgentReceiver.GetPosition());

            AgentReceiver.SenseLandmarks(LN, landmarkPositions);

            AgentSignaller.SenseFood(food_location);

            // Move both the agents
            AgentReceiver.Step(StepSize);
            AgentSignaller.Step(StepSize);

            // clamp the signallers position
            double pos = AgentSignaller.GetPosition();
            
            if (pos > LANDMARKZONESTART){
                AgentSignaller.SetPosition(LANDMARKZONESTART);
            }
        }

        for (int i= 1; i<=N;i ++){
            savedStateReceiver[i] = AgentReceiver.NervousSystem.NeuronState(i);
        }
        for (int i= 1; i<=N;i ++){
            savedStateSignaller[i] = AgentSignaller.NervousSystem.NeuronState(i);
        }

        for (double ref_var = -1.0; ref_var <= 1.0; ref_var += 1.0){
            for (double sep_var = -1.0; sep_var <= 1.0; sep_var += 1.0){

                genLandmarks_Simple(ref_var, sep_var, landmarkPositions);

                for (int i= 1; i<=N; i++){
                    AgentReceiver.NervousSystem.SetNeuronState(i, savedStateReceiver[i]);
                }
                for (int i= 1; i<=N; i++){
                    AgentSignaller.NervousSystem.SetNeuronState(i, savedStateSignaller[i]);
                }

                // Phase 2/3 (Training + scoring Phase)

                double scoringTime = 0.0;
                double totalScore = 0.0;
                
                for (double time=0; time < RunDuration*2; time += StepSize){
                    // Receiver and Signaller only see each other 
                    AgentReceiver.SenseOther(AgentSignaller.GetPosition());
                    AgentSignaller.SenseOther(AgentReceiver.GetPosition());

                    AgentReceiver.SenseLandmarks(LN, landmarkPositions);

                    AgentSignaller.SenseFood(food_location);

                    // Move both the agents
                    AgentReceiver.Step(StepSize);
                    AgentSignaller.Step(StepSize);

                    // clamp the signallers position
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
                    }
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
                }
                total_fitness += fitness;
                total_trials +=1;

            }
        }
    }
    return (total_trials > 0) ? total_fitness / total_trials : 0.0;
}


double Fitness3(TVector<double> &genotype, RandomState &rs){
    call_count ++;

    // initalise their genotypes 
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    CountingAgent AgentSignaller(N, phenotypeSignaller);
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // save the state of the signaller and receiver
    TVector<double> savedStateSignaller, savedStateReceiver;
    savedStateSignaller.SetBounds(1,N); 
    savedStateReceiver.SetBounds(1, N);

    // Landmark Positions
    TVector<double> landmarkPositions;

    // trial variables 
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_food_receiver;

    double env_fitness[LN+1] = {0};  // store fitness per env
    double env_receiver_pos[LN+1] = {0};
    double env_food_loc[LN+1] = {0};


    for (int env =1; env<=LN;env++){
        // ******** SETUP **********
        genLandmarks_Simple(0, 0, landmarkPositions);

        // set the food to be at the ith landmark location
        food_location = landmarkPositions[env];

        // Initialise the agent for this trial
        AgentReceiver.SetPosition(0);
        AgentReceiver.ResetNeuralState();
        AgentReceiver.ResetSensors();

        // Initialise the agent for this trial
        // double location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
        AgentSignaller.SetPosition(0);
        AgentSignaller.ResetNeuralState();
        AgentSignaller.ResetSensors();

        // Phase 1 (Training Phase) -> signaller wondering

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

        // Phase 2/3 (Training + scoring Phase)
        
        for (double time=0; time < RunDuration*2; time += StepSize){
            // Receiver and Signaller only see each other 
            AgentReceiver.SenseOther(AgentSignaller.GetPosition());
            AgentSignaller.SenseOther(AgentReceiver.GetPosition());

            AgentReceiver.SenseLandmarks(LN, landmarkPositions);

            AgentSignaller.SenseFood(food_location);

            // Move both the agents
            AgentReceiver.Step(StepSize);
            AgentSignaller.Step(StepSize);

            // clamp the signallers position
            double pos = AgentSignaller.GetPosition();
            
            if (pos > LANDMARKZONESTART){
                AgentSignaller.SetPosition(LANDMARKZONESTART);
            }
        }

        for (int i= 1; i<=N;i ++){
            savedStateReceiver[i] = AgentReceiver.NervousSystem.NeuronState(i);
        }
        for (int i= 1; i<=N;i ++){
            savedStateSignaller[i] = AgentSignaller.NervousSystem.NeuronState(i);
        }

        for (double ref_var = -2.0; ref_var <= 2.0; ref_var += 2.0){
            for (double sep_var = -2.0; sep_var <= 2.0; sep_var += 2.0){
                AgentReceiver.SetPosition(0);
                AgentSignaller.SetPosition(0);
                AgentReceiver.ResetSensors();
                AgentSignaller.ResetSensors();

                genLandmarks_Simple(ref_var, sep_var, landmarkPositions);
                food_location = landmarkPositions[env];

                for (int i= 1; i<=N; i++){
                    AgentReceiver.NervousSystem.SetNeuronState(i, savedStateReceiver[i]);
                }
                for (int i= 1; i<=N; i++){
                    AgentSignaller.NervousSystem.SetNeuronState(i, savedStateSignaller[i]);
                }

                // Phase 2/3 (Training + scoring Phase)

                double scoringTime = 0.0;
                double totalScore = 0.0;
                
                for (double time=0; time < RunDuration*2; time += StepSize){
                    // Receiver and Signaller only see each other 
                    AgentReceiver.SenseOther(AgentSignaller.GetPosition());
                    AgentSignaller.SenseOther(AgentReceiver.GetPosition());

                    AgentReceiver.SenseLandmarks(LN, landmarkPositions);

                    AgentSignaller.SenseFood(food_location);

                    // Move both the agents
                    AgentReceiver.Step(StepSize);
                    AgentSignaller.Step(StepSize);

                    // clamp the signallers position
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
                    }
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
                }
                total_fitness += fitness;
                total_trials +=1;

                // for debug
                env_fitness[env] += fitness;
                env_receiver_pos[env] = AgentReceiver.GetPosition();
                env_food_loc[env] = food_location;
            }
        }
    }

    double final_fitness = (total_trials > 0) ? total_fitness / total_trials : 0.0;
    
    // // Write debug at end of this call
    // ofstream debugfile;
    // std::string file_name = dir + "fitness3_debug.dat";
    // debugfile.open(file_name, std::ios::app); // append to the end of the file
    // debugfile << "call=" << call_count << " total_fitness=" << final_fitness;
    // for (int i = 1; i <= LN; i++){
    //     debugfile << " env" << i << "_fitness=" << env_fitness[i]/9.0  // 9 trials per env
    //               << " env" << i << "_receiver=" << env_receiver_pos[i]
    //               << " env" << i << "_food=" << env_food_loc[i];
    // }
    // debugfile << endl;
    // debugfile.close();

    // if (final_fitness > 0.99){
    //     ofstream genFile;
    //     std::string file_name = dir + "fitness3_good_generations.dat";
    //     genFile.open(file_name, std::ios::app);
    //     genFile << "call=" << call_count
    //             << " fitness=" << final_fitness
    //             << " genotype=" << genotype
    //             << endl;
    //     genFile.close();
    // }

    return final_fitness;
}


double Fitness3_withRecord(TVector<double> &genotype, RandomState &rs){
    call_count ++;

    // initalise their genotypes 
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    CountingAgent AgentSignaller(N, phenotypeSignaller);
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // save the state of the signaller and receiver
    TVector<double> savedStateSignaller, savedStateReceiver;
    savedStateSignaller.SetBounds(1,N); 
    savedStateReceiver.SetBounds(1, N);

    // Landmark Positions
    TVector<double> landmarkPositions;

    // trial variables 
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_food_receiver;

    double env_fitness[LN+1] = {0};  // store fitness per env
    double env_receiver_pos[LN+1] = {0};
    double env_food_loc[LN+1] = {0};

    // storage for testing phase
    std::vector<std::vector<TrialData>> all_trials(LN+1, std::vector<TrialData>(9)); // [env][trial_num]

    // storage for initial phase
    std::vector<std::vector<double>> phase1_signaller(LN+1); // one row for each env
    std::vector<std::vector<double>> phase1_receiver(LN+1);
    std::vector<std::vector<double>> phase1_ns1(LN+1), phase1_ns2(LN+1), phase1_ns3(LN+1);
    std::vector<std::vector<double>> phase1_nr1(LN+1), phase1_nr2(LN+1), phase1_nr3(LN+1);
    std::vector<double> phase1_food(LN+1);
    std::vector<std::vector<double>> phase1_landmark(LN+1);

    for (int env =1; env<=LN;env++){
        // ******** SETUP **********
        genLandmarks_Simple(0, 0, landmarkPositions);

        // set the food to be at the ith landmark location
        food_location = landmarkPositions[env];

        phase1_food[env] = food_location;
        phase1_landmark[env].clear();
        for (int l = 1; l <= LN; l++){
            phase1_landmark[env].push_back(landmarkPositions[l]);
        }

        // Initialise the agent for this trial
        AgentReceiver.SetPosition(0);
        AgentReceiver.ResetNeuralState();
        AgentReceiver.ResetSensors();

        // Initialise the agent for this trial
        // double location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
        AgentSignaller.SetPosition(0);
        AgentSignaller.ResetNeuralState();
        AgentSignaller.ResetSensors();

        // Phase 1 (Training Phase) -> signaller wondering

        for (double time=0; time < RunDuration; time += StepSize){
            // only let the Signaller move
            AgentSignaller.SenseFood(food_location);
            AgentSignaller.SenseLandmarks(LN, landmarkPositions);
            AgentSignaller.SenseOther(AgentReceiver.GetPosition());
            AgentSignaller.Step(StepSize);

            // record that timestep
            phase1_signaller[env].push_back(AgentSignaller.GetPosition());
            phase1_receiver[env].push_back(AgentReceiver.GetPosition());
            phase1_ns1[env].push_back(AgentSignaller.NervousSystem.NeuronState(1));
            phase1_ns2[env].push_back(AgentSignaller.NervousSystem.NeuronState(2));
            phase1_ns3[env].push_back(AgentSignaller.NervousSystem.NeuronState(3));
            phase1_nr1[env].push_back(AgentReceiver.NervousSystem.NeuronState(1));
            phase1_nr2[env].push_back(AgentReceiver.NervousSystem.NeuronState(2));
            phase1_nr3[env].push_back(AgentReceiver.NervousSystem.NeuronState(3));
        }
        
        // Reset
        AgentReceiver.SetPosition(0);
        AgentSignaller.SetPosition(0);
        AgentReceiver.ResetSensors();
        AgentSignaller.ResetSensors();

        // Phase 2/3 (Training + scoring Phase)
        
        for (double time=0; time < RunDuration*2; time += StepSize){
            // Receiver and Signaller only see each other 
            AgentReceiver.SenseOther(AgentSignaller.GetPosition());
            AgentSignaller.SenseOther(AgentReceiver.GetPosition());

            AgentReceiver.SenseLandmarks(LN, landmarkPositions);

            AgentSignaller.SenseFood(food_location);

            // Move both the agents
            AgentReceiver.Step(StepSize);
            AgentSignaller.Step(StepSize);

            // clamp the signallers position
            double pos = AgentSignaller.GetPosition();
            
            if (pos > LANDMARKZONESTART){
                AgentSignaller.SetPosition(LANDMARKZONESTART);
            }

            // record the timestep
            phase1_signaller[env].push_back(AgentSignaller.GetPosition());
            phase1_receiver[env].push_back(AgentReceiver.GetPosition());
            phase1_ns1[env].push_back(AgentSignaller.NervousSystem.NeuronState(1));
            phase1_ns2[env].push_back(AgentSignaller.NervousSystem.NeuronState(2));
            phase1_ns3[env].push_back(AgentSignaller.NervousSystem.NeuronState(3));
            phase1_nr1[env].push_back(AgentReceiver.NervousSystem.NeuronState(1));
            phase1_nr2[env].push_back(AgentReceiver.NervousSystem.NeuronState(2));
            phase1_nr3[env].push_back(AgentReceiver.NervousSystem.NeuronState(3));
        }

        for (int i= 1; i<=N;i ++){
            savedStateReceiver[i] = AgentReceiver.NervousSystem.NeuronState(i);
        }
        for (int i= 1; i<=N;i ++){
            savedStateSignaller[i] = AgentSignaller.NervousSystem.NeuronState(i);
        }

        int trial_num = 0;

        for (double ref_var = -2.0; ref_var <= 2.0; ref_var += 2.0){
            for (double sep_var = -2.0; sep_var <= 2.0; sep_var += 2.0){
                AgentReceiver.SetPosition(0);
                AgentSignaller.SetPosition(0);
                AgentReceiver.ResetSensors();
                AgentSignaller.ResetSensors();

                genLandmarks_Simple(ref_var, sep_var, landmarkPositions);
                food_location = landmarkPositions[env];
                double lengthZone = landmarkPositions[LN] - landmarkPositions[1];

                for (int i= 1; i<=N; i++){
                    AgentReceiver.NervousSystem.SetNeuronState(i, savedStateReceiver[i]);
                }
                for (int i= 1; i<=N; i++){
                    AgentSignaller.NervousSystem.SetNeuronState(i, savedStateSignaller[i]);
                }

                // Phase 2/3 (Training + scoring Phase)

                double scoringTime = 0.0;
                double totalScore = 0.0;

                TrialData &td = all_trials[env][trial_num];

                td.food_location = food_location;
                td.landmark_positions.clear();
                for (int l = 1; l <= LN; l++) {
                    td.landmark_positions.push_back(landmarkPositions[l]);
                }

                // record initial position
                td.signaller_pos.push_back(AgentSignaller.GetPosition());
                td.receiver_pos.push_back(AgentReceiver.GetPosition());
                td.neuron_s1.push_back(AgentSignaller.NervousSystem.NeuronState(1));
                td.neuron_s2.push_back(AgentSignaller.NervousSystem.NeuronState(2));
                td.neuron_s3.push_back(AgentSignaller.NervousSystem.NeuronState(3));
                td.neuron_r1.push_back(AgentReceiver.NervousSystem.NeuronState(1));
                td.neuron_r2.push_back(AgentReceiver.NervousSystem.NeuronState(2));
                td.neuron_r3.push_back(AgentReceiver.NervousSystem.NeuronState(3));
                
                for (double time=0; time < RunDuration*2; time += StepSize){
                    // Receiver and Signaller only see each other 
                    AgentReceiver.SenseOther(AgentSignaller.GetPosition());
                    AgentSignaller.SenseOther(AgentReceiver.GetPosition());

                    AgentReceiver.SenseLandmarks(LN, landmarkPositions);

                    AgentSignaller.SenseFood(food_location);

                    // Move both the agents
                    AgentReceiver.Step(StepSize);
                    AgentSignaller.Step(StepSize);

                    // clamp the signallers position
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

                        // record fitness over time
                        if (scoringTime > 0){
                            td.fitness_over_time.push_back(1 - ((totalScore/scoringTime)/lengthZone));
                        }
                    }

                    td.signaller_pos.push_back(AgentSignaller.GetPosition());
                    td.receiver_pos.push_back(AgentReceiver.GetPosition());
                    td.neuron_s1.push_back(AgentSignaller.NervousSystem.NeuronState(1));
                    td.neuron_s2.push_back(AgentSignaller.NervousSystem.NeuronState(2));
                    td.neuron_s3.push_back(AgentSignaller.NervousSystem.NeuronState(3));
                    td.neuron_r1.push_back(AgentReceiver.NervousSystem.NeuronState(1));
                    td.neuron_r2.push_back(AgentReceiver.NervousSystem.NeuronState(2));
                    td.neuron_r3.push_back(AgentReceiver.NervousSystem.NeuronState(3));

                }
                // length of the landmark zone

                // END OF TRIAL
                // score at the end of this trial
                double fitness = 0;
                if (scoringTime > 0){
                    fitness = 1 - ((totalScore/scoringTime)/lengthZone);
                    if (fitness < 0.0){
                        fitness = 0.0;
                    }
                }
                total_fitness += fitness;
                total_trials +=1;

                // for debug
                env_fitness[env] += fitness;
                env_receiver_pos[env] = AgentReceiver.GetPosition();
                env_food_loc[env] = food_location;

                td.fitness = fitness;
                trial_num++;
            }
        }
    }

    double final_fitness = (total_trials > 0) ? total_fitness / total_trials : 0.0;
    
    // Write debug at end of this call
    ofstream debugfile;
    std::string file_name = dir + "fitness3_debug.dat";
    debugfile.open(file_name, std::ios::app); // append to the end of the file
    debugfile << "call=" << call_count << " total_fitness=" << final_fitness;
    for (int i = 1; i <= LN; i++){
        debugfile << " env" << i << "_fitness=" << env_fitness[i]/9.0  // 9 trials per env
                  << " env" << i << "_receiver=" << env_receiver_pos[i]
                  << " env" << i << "_food=" << env_food_loc[i];
    }
    debugfile << endl;
    debugfile.close();


    // If the behaviour was what we wanted record it

    if (final_fitness > 0.99){
        std::string run_id = std::to_string(call_count);

        ofstream TotalFitnessFile;
        TotalFitnessFile.open(dir + "total_fitness_" + run_id + ".dat");

        double running_total = 0.0;
        int running_trials = 0;

        for (int env = 1; env <= LN; env++){
            std::string s_env = std::to_string(env);

            // open all the same files as RecordBehavior
            ofstream SignallerBehaviorFile, ReceiverBehaviorFile;
            SignallerBehaviorFile.open(dir + "behavior_Signaller_training_env_" + s_env + "_" + run_id + ".dat");
            ReceiverBehaviorFile.open(dir + "behavior_Receiver_training_env_" + s_env + "_" + run_id + ".dat");

            ofstream SignallerBehaviorFile2, ReceiverBehaviorFile2;
            SignallerBehaviorFile2.open(dir + "behavior_Signaller_testing_env_" + s_env + "_" + run_id + ".dat");
            ReceiverBehaviorFile2.open(dir + "behavior_Receiver_testing_env_" + s_env + "_" + run_id + ".dat");

            ofstream FitnessFile, LandmarkFile, LandmarkFile2;
            FitnessFile.open(dir + "fitness_env_" + s_env + "_" + run_id + ".dat");
            LandmarkFile.open(dir + "landmark_location_training_env_" + s_env + "_" + run_id + ".dat");
            LandmarkFile2.open(dir + "landmark_location_testing_env_" + s_env + "_" + run_id + ".dat");

            ofstream NeuronS1, NeuronS2, NeuronS3, NeuronR1, NeuronR2, NeuronR3;
            NeuronS1.open(dir + "signaller_neuron1_training_env_" + s_env + "_" + run_id + ".dat");
            NeuronS2.open(dir + "signaller_neuron2_training_env_" + s_env + "_" + run_id + ".dat");
            NeuronS3.open(dir + "signaller_neuron3_training_env_" + s_env + "_" + run_id + ".dat");
            NeuronR1.open(dir + "receiver_neuron1_training_env_" + s_env + "_" + run_id + ".dat");
            NeuronR2.open(dir + "receiver_neuron2_training_env_" + s_env + "_" + run_id + ".dat");
            NeuronR3.open(dir + "receiver_neuron3_training_env_" + s_env + "_" + run_id + ".dat");

            ofstream NeuronS12, NeuronS22, NeuronS32, NeuronR12, NeuronR22, NeuronR32;
            NeuronS12.open(dir + "signaller_neuron1_testing_env_" + s_env + "_" + run_id + ".dat");
            NeuronS22.open(dir + "signaller_neuron2_testing_env_" + s_env + "_" + run_id + ".dat");
            NeuronS32.open(dir + "signaller_neuron3_testing_env_" + s_env + "_" + run_id + ".dat");
            NeuronR12.open(dir + "receiver_neuron1_testing_env_" + s_env + "_" + run_id + ".dat");
            NeuronR22.open(dir + "receiver_neuron2_testing_env_" + s_env + "_" + run_id + ".dat");
            NeuronR32.open(dir + "receiver_neuron3_testing_env_" + s_env + "_" + run_id + ".dat");

            // write phase 1
            for (double p : phase1_signaller[env]) SignallerBehaviorFile << p << " ";
            for (double p : phase1_receiver[env])  ReceiverBehaviorFile  << p << " ";
            for (double p : phase1_ns1[env]) NeuronS1 << p << " ";
            for (double p : phase1_ns2[env]) NeuronS2 << p << " ";
            for (double p : phase1_ns3[env]) NeuronS3 << p << " ";
            for (double p : phase1_nr1[env]) NeuronR1 << p << " ";
            for (double p : phase1_nr2[env]) NeuronR2 << p << " ";
            for (double p : phase1_nr3[env]) NeuronR3 << p << " ";
            for (double lp : phase1_landmark[env]) LandmarkFile << lp << " ";
            LandmarkFile << phase1_food[env] << " ";

            // write each trial (testing phase)
            for (int t = 0; t < 9; t++){
                TrialData &td = all_trials[env][t];

                // landmark file
                for (double lp : td.landmark_positions) LandmarkFile2 << lp << " ";
                LandmarkFile2 << td.food_location << " ";
                LandmarkFile2 << endl;

                // positions
                for (double p : td.signaller_pos) SignallerBehaviorFile2 << p << " ";
                for (double p : td.receiver_pos)  ReceiverBehaviorFile2  << p << " ";
                SignallerBehaviorFile2 << endl;
                ReceiverBehaviorFile2  << endl;

                // neurons
                for (double p : td.neuron_s1) NeuronS12 << p << " ";
                for (double p : td.neuron_s2) NeuronS22 << p << " ";
                for (double p : td.neuron_s3) NeuronS32 << p << " ";
                for (double p : td.neuron_r1) NeuronR12 << p << " ";
                for (double p : td.neuron_r2) NeuronR22 << p << " ";
                for (double p : td.neuron_r3) NeuronR32 << p << " ";
                NeuronS12 << endl; NeuronS22 << endl; NeuronS32 << endl;
                NeuronR12 << endl; NeuronR22 << endl; NeuronR32 << endl;

                // fitness over time
                for (double f : td.fitness_over_time) FitnessFile << f << " ";
                FitnessFile << endl;

                // total fitness running average
                running_total += td.fitness;
                running_trials++;
                TotalFitnessFile << running_total / running_trials << " ";
                TotalFitnessFile << endl;
            }

            SignallerBehaviorFile.close();  ReceiverBehaviorFile.close();
            SignallerBehaviorFile2.close(); ReceiverBehaviorFile2.close();
            FitnessFile.close();      LandmarkFile.close();
            NeuronS1.close(); NeuronS2.close(); NeuronS3.close();
            NeuronR1.close(); NeuronR2.close(); NeuronR3.close();
            NeuronS12.close(); NeuronS22.close(); NeuronS32.close();
            NeuronR12.close(); NeuronR22.close(); NeuronR32.close();
        }
        TotalFitnessFile.close();
    }

    return final_fitness;
}

double Fitness4(TVector<double> &genotype, RandomState &rs){
    call_count ++;

    // initalise their genotypes 
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    CountingAgent AgentSignaller(N, phenotypeSignaller);
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // save the state of the signaller and receiver
    TVector<double> savedStateSignaller, savedStateReceiver;
    savedStateSignaller.SetBounds(1,N); 
    savedStateReceiver.SetBounds(1, N);

    std::vector<std::vector<double>> positions = {
        {15, 25, 35},
        {15, 30, 45},
        {20, 25, 30}
        {20, 30, 40},
        {20, 35, 50},
    };

    // Landmark Positions (last column removed)
    std::vector<TVector<double>> landmarkPositions(positions.size());
    for (int x = 0; x < positions.size(); x ++){
        std::vector<double> temp = positions[x];
        landmarkPositions[x].SetBounds(1,3);
        for (int i = 1; i<=3;i++){
            landmarkPositions[x][i] = positions[x][i-1];
        }
    }

    // trial variables 
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_food_receiver;

    double env_fitness[LN+1] = {0};  // store fitness per env
    double env_receiver_pos[LN+1] = {0};
    double env_food_loc[LN+1] = {0};


    for (int env =1; env<=LN;env++){
        // ******** SETUP **********
        // set the food to be at the ith landmark location
        food_location = landmarkPositions[4][env];

        // Initialise the agent for this trial
        AgentReceiver.SetPosition(0);
        AgentReceiver.ResetNeuralState();
        AgentReceiver.ResetSensors();

        // Initialise the agent for this trial
        // double location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
        AgentSignaller.SetPosition(0);
        AgentSignaller.ResetNeuralState();
        AgentSignaller.ResetSensors();

        // Phase 1 (Training Phase) -> signaller wondering

        for (double time=0; time < RunDuration; time += StepSize){
            // only let the Signaller move
            AgentSignaller.SenseFood(food_location);
            AgentSignaller.SenseLandmarks(LN, landmarkPositions[4]);
            AgentSignaller.SenseOther(AgentReceiver.GetPosition());
            AgentSignaller.Step(StepSize);
        }
        
        // Reset
        AgentReceiver.SetPosition(0);
        AgentSignaller.SetPosition(0);
        AgentReceiver.ResetSensors();
        AgentSignaller.ResetSensors();

        // Phase 2/3 (Training + scoring Phase)
        
        for (double time=0; time < RunDuration*2; time += StepSize){
            // Receiver and Signaller only see each other 
            AgentReceiver.SenseOther(AgentSignaller.GetPosition());
            AgentSignaller.SenseOther(AgentReceiver.GetPosition());

            AgentReceiver.SenseLandmarks(LN, landmarkPositions[4]);

            AgentSignaller.SenseFood(food_location);

            // Move both the agents
            AgentReceiver.Step(StepSize);
            AgentSignaller.Step(StepSize);

            // clamp the signallers position
            double pos = AgentSignaller.GetPosition();
            
            if (pos > LANDMARKZONESTART){
                AgentSignaller.SetPosition(LANDMARKZONESTART);
            }
        }

        for (int i= 1; i<=N;i ++){
            savedStateReceiver[i] = AgentReceiver.NervousSystem.NeuronState(i);
        }
        for (int i= 1; i<=N;i ++){
            savedStateSignaller[i] = AgentSignaller.NervousSystem.NeuronState(i);
        }

        for (double trial = 0; trial < positions.size(); trial += 1){
            AgentReceiver.SetPosition(0);
            AgentSignaller.SetPosition(0);
            AgentReceiver.ResetSensors();
            AgentSignaller.ResetSensors();

            food_location = landmarkPositions[trial][env];

            for (int i= 1; i<=N; i++){
                AgentReceiver.NervousSystem.SetNeuronState(i, savedStateReceiver[i]);
            }
            for (int i= 1; i<=N; i++){
                AgentSignaller.NervousSystem.SetNeuronState(i, savedStateSignaller[i]);
            }

            // Phase 2/3 (Training + scoring Phase)

            double scoringTime = 0.0;
            double totalScore = 0.0;
            
            for (double time=0; time < RunDuration*2; time += StepSize){
                // Receiver and Signaller only see each other 
                AgentReceiver.SenseOther(AgentSignaller.GetPosition());
                AgentSignaller.SenseOther(AgentReceiver.GetPosition());

                AgentReceiver.SenseLandmarks(LN, landmarkPositions[trial]);

                AgentSignaller.SenseFood(food_location);

                // Move both the agents
                AgentReceiver.Step(StepSize);
                AgentSignaller.Step(StepSize);

                // clamp the signallers position
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
                }
            }
            // length of the landmark zone
            double lengthZone = landmarkPositions[trial][LN] - landmarkPositions[trial][1];
            // END OF TRIAL
            // score at the end of this trial
            double fitness = 0;
            if (scoringTime > 0){
                fitness = 1 - ((totalScore/scoringTime)/lengthZone);
                if (fitness < 0.0){
                    fitness = 0.0;
                }
            }
            total_fitness += fitness;
            total_trials +=1;

            // for debug
            env_fitness[env] += fitness;
            env_receiver_pos[env] = AgentReceiver.GetPosition();
            env_food_loc[env] = food_location;
        
        }
    }

    double final_fitness = (total_trials > 0) ? total_fitness / total_trials : 0.0;
    

    if (final_fitness > 0.99){
        ofstream genFile;
        std::string file_name = dir + "fitness4_good_generations.dat";
        genFile.open(file_name, std::ios::app);
        genFile << "call=" << call_count
                << " fitness=" << final_fitness
                << " genotype=" << genotype
                << endl;
        genFile.close();
    }

    return final_fitness;
}

double RecordBehavior(TSearch &s, RandomState &rs){
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();

    // trial variables 
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_food_receiver;
    double distance_signaller_receiver;

    // **** create the receiver ****
    // Map genotype to phenotype
    TVector<double> genotype;
    genotype = s.BestIndividual();

    ofstream genFile;
    std::string file_name = dir + "record_genotype.dat";
    genFile.open(file_name, std::ios::app);
    genFile << genotype
            << endl;
    genFile.close();

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

    TVector<double> savedStateSignallerTester, savedStateReceiverTester;
    savedStateSignallerTester.SetBounds(1,N);
    savedStateReceiverTester.SetBounds(1,N);

    // **** Generate landmarks (using leapfrog method) ****
    TVector<double> landmarkPositions;
    ofstream TotalFitness;
    TotalFitness.open( dir + "total_fitness_" + current_run +".dat");

    for (int env =1; env<=LN;env++){
        std::string s_env = std::to_string(env);
        // **** create the files to save the behaviour/env ***
        // storing them all in the same file, each gets a new line (so 9 trials per file)
        // the stages are separated into three files
        ofstream SignallerBehaviorFile, RecieverBehaviorFile;
        SignallerBehaviorFile.open( dir + "behavior_Signaller_stage1_env_"+ s_env +"_" + current_run + ".dat");
        RecieverBehaviorFile.open( dir + "behavior_Reciever_stage1_env_"+ s_env +"_" + current_run + ".dat");
        ofstream SignallerBehaviorFile2, RecieverBehaviorFile2;
        SignallerBehaviorFile2.open( dir + "behavior_Signaller_stage2_env_"+ s_env +"_" + current_run + ".dat");
        RecieverBehaviorFile2.open( dir + "behavior_Reciever_stage2_env_"+ s_env +"_" + current_run + ".dat");
        ofstream Stage3Fitness;
        Stage3Fitness.open( dir + "fitness_env_"+ s_env +"_" + current_run +".dat");

        // stores the location of the landmarks and the food -> this is each line 
        ofstream LandmarkFile;
        LandmarkFile.open(dir + "landmark_location_stage3_env_"+ s_env +"_"+current_run+".dat");

        // files to save the neuron state
        ofstream NeuronS1, NeuronS2, NeuronS3;
        NeuronS1.open(dir + "signaller_neuron1_stage3_env_"+ s_env +"_" +current_run+".dat");
        NeuronS2.open(dir + "signaller_neuron2_stage3_env_"+ s_env +"_" +current_run+".dat");
        NeuronS3.open(dir + "signaller_neuron3_stage3_env_"+ s_env +"_" +current_run+".dat");
        ofstream NeuronR1, NeuronR2, NeuronR3;
        NeuronR1.open(dir + "receiver_neuron1_stage3_env_"+ s_env +"_" +current_run+".dat");
        NeuronR2.open(dir + "receiver_neuron2_stage3_env_"+ s_env +"_" +current_run+".dat");
        NeuronR3.open(dir + "receiver_neuron3_stage3_env_"+ s_env +"_" +current_run+".dat");


        genLandmarks_Simple(0, 0, landmarkPositions);
        // set the food to be at the ith landmark location
        food_location = landmarkPositions[env];
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

        // record the initial position
        SignallerBehaviorFile << AgentSignaller.GetPosition() << " ";
        RecieverBehaviorFile << AgentReceiver.GetPosition() << " ";

        NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
        NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
        NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";

        NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
        NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
        NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";

        for (double time=0; time < RunDuration; time += StepSize){
            // only let the Signaller move
            AgentSignaller.SenseFood(food_location);
            AgentSignaller.SenseLandmarks(LN, landmarkPositions);
            AgentSignaller.SenseOther(AgentReceiver.GetPosition());
            AgentSignaller.Step(StepSize);

            SignallerBehaviorFile << AgentSignaller.GetPosition() << " ";
            RecieverBehaviorFile << AgentReceiver.GetPosition() << " ";
        
            NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
            NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
            NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";

            NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
            NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
            NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";
        }

        // Reset
        AgentReceiver.SetPosition(0);
        AgentSignaller.SetPosition(0);
        AgentReceiver.ResetSensors();
        AgentSignaller.ResetSensors();

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

        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            savedStateSignallerTester[i] = AgentSignaller.NervousSystem.NeuronState(i);
        }
        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            savedStateReceiverTester[i] = AgentReceiver.NervousSystem.NeuronState(i);
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
                for (int x = 1; x<= LN; x += 1){
                    LandmarkFile << landmarkPositions[x] << " ";
                }
                LandmarkFile << food_location << " ";

                for (int i= 1; i<=N; i++){
                    AgentReceiver.NervousSystem.SetNeuronState(i, savedStateReceiverTester[i]);
                }
                for (int i= 1; i<=N; i++){
                    AgentSignaller.NervousSystem.SetNeuronState(i, savedStateSignallerTester[i]);
                }

                // ********* PHASE 2/3 ************

                // record the initial position
                SignallerBehaviorFile2 << AgentSignaller.GetPosition() << " ";
                RecieverBehaviorFile2 << AgentReceiver.GetPosition() << " ";
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

                    SignallerBehaviorFile2 << AgentSignaller.GetPosition() << " ";
                    RecieverBehaviorFile2 << AgentReceiver.GetPosition() << " ";
                    
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
                }
                total_fitness += fitness;
                total_trials +=1;

                if (total_trials > 0){
                    TotalFitness << total_fitness / total_trials << " ";
                } else{
                    TotalFitness << 0 << " ";
                }

                // new line for a new trial
                Stage3Fitness << endl;
                LandmarkFile << endl;

                SignallerBehaviorFile2 << endl;
                NeuronS1 << endl;
                NeuronS2 << endl;
                NeuronS3 << endl;

                RecieverBehaviorFile2 << endl;
                NeuronR1 << endl;
                NeuronR2 << endl;
                NeuronR3 << endl;

                TotalFitness << endl;
            }
        }
        // close the files for that env
        SignallerBehaviorFile.close();
        RecieverBehaviorFile.close();
        SignallerBehaviorFile2.close();
        RecieverBehaviorFile2.close();
        Stage3Fitness.close();
        LandmarkFile.close();
        NeuronS1.close();
        NeuronS2.close();
        NeuronS3.close();
        NeuronR1.close();
        NeuronR2.close();
        NeuronR3.close();
    }
    TotalFitness.close();
    return (total_trials > 0) ? total_fitness / total_trials : 0.0;
}


double RecordBehavior2(TSearch &s, RandomState &rs){
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();

    // trial variables 
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_food_receiver;
    double distance_signaller_receiver;

    // **** create the receiver ****
    // Map genotype to phenotype
    TVector<double> genotype;
    genotype = s.BestIndividual();

    ofstream genFile;
    std::string file_name = dir + "record_genotype.dat";
    genFile.open(file_name, std::ios::app);
    genFile << genotype
            << endl;
    genFile.close();

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

    TVector<double> savedStateSignallerTester, savedStateReceiverTester;
    savedStateSignallerTester.SetBounds(1,N);
    savedStateReceiverTester.SetBounds(1,N);

    // **** Generate landmarks (using leapfrog method) ****
    std::vector<std::vector<double>> positions = {
        {15, 25, 35},
        {15, 30, 45},
        {20, 25, 30}
        {20, 30, 40},
        {20, 35, 50},
    };

    // Landmark Positions (last column removed)
    std::vector<TVector<double>> landmarkPositions(positions.size());
    for (int x = 0; x < positions.size(); x ++){
        std::vector<double> temp = positions[x];
        landmarkPositions[x].SetBounds(1,3);
        for (int i = 1; i<=3;i++){
            landmarkPositions[x][i] = positions[x][i-1];
        }
    }
    ofstream TotalFitness;
    TotalFitness.open( dir + "total_fitness_" + current_run +".dat");

    for (int env =1; env<=LN;env++){
        std::string s_env = std::to_string(env);
        // **** create the files to save the behaviour/env ***
        // storing them all in the same file, each gets a new line (so 9 trials per file)
        // the stages are separated into three files
        ofstream SignallerBehaviorFile, RecieverBehaviorFile;
        SignallerBehaviorFile.open( dir + "behavior_Signaller_stage1_env_"+ s_env +"_" + current_run + ".dat");
        RecieverBehaviorFile.open( dir + "behavior_Reciever_stage1_env_"+ s_env +"_" + current_run + ".dat");
        ofstream SignallerBehaviorFile2, RecieverBehaviorFile2;
        SignallerBehaviorFile2.open( dir + "behavior_Signaller_stage2_env_"+ s_env +"_" + current_run + ".dat");
        RecieverBehaviorFile2.open( dir + "behavior_Reciever_stage2_env_"+ s_env +"_" + current_run + ".dat");
        ofstream Stage3Fitness;
        Stage3Fitness.open( dir + "fitness_env_"+ s_env +"_" + current_run +".dat");

        // stores the location of the landmarks and the food -> this is each line 
        ofstream LandmarkFile;
        LandmarkFile.open(dir + "landmark_location_stage3_env_"+ s_env +"_"+current_run+".dat");

        // files to save the neuron state
        ofstream NeuronS1, NeuronS2, NeuronS3;
        NeuronS1.open(dir + "signaller_neuron1_stage3_env_"+ s_env +"_" +current_run+".dat");
        NeuronS2.open(dir + "signaller_neuron2_stage3_env_"+ s_env +"_" +current_run+".dat");
        NeuronS3.open(dir + "signaller_neuron3_stage3_env_"+ s_env +"_" +current_run+".dat");
        ofstream NeuronR1, NeuronR2, NeuronR3;
        NeuronR1.open(dir + "receiver_neuron1_stage3_env_"+ s_env +"_" +current_run+".dat");
        NeuronR2.open(dir + "receiver_neuron2_stage3_env_"+ s_env +"_" +current_run+".dat");
        NeuronR3.open(dir + "receiver_neuron3_stage3_env_"+ s_env +"_" +current_run+".dat");


        // set the food to be at the ith landmark location
        food_location = landmarkPositions[4][env];
        for (int x = 1; x<= LN; x += 1){
            LandmarkFile << landmarkPositions[4][x] << " ";
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

        // record the initial position
        SignallerBehaviorFile << AgentSignaller.GetPosition() << " ";
        RecieverBehaviorFile << AgentReceiver.GetPosition() << " ";

        NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
        NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
        NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";

        NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
        NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
        NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";

        for (double time=0; time < RunDuration; time += StepSize){
            // only let the Signaller move
            AgentSignaller.SenseFood(food_location);
            AgentSignaller.SenseLandmarks(LN, landmarkPositions[4]);
            AgentSignaller.SenseOther(AgentReceiver.GetPosition());
            AgentSignaller.Step(StepSize);

            SignallerBehaviorFile << AgentSignaller.GetPosition() << " ";
            RecieverBehaviorFile << AgentReceiver.GetPosition() << " ";
        
            NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
            NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
            NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";

            NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
            NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
            NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";
        }

        // Reset
        AgentReceiver.SetPosition(0);
        AgentSignaller.SetPosition(0);
        AgentReceiver.ResetSensors();
        AgentSignaller.ResetSensors();

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

        for (double time=0; time < RunDuration*2; time += StepSize){
            
            AgentReceiver.SenseOther(AgentSignaller.GetPosition());
            AgentSignaller.SenseOther(AgentReceiver.GetPosition());

            AgentReceiver.SenseLandmarks(LN, landmarkPositions[4]);

            AgentSignaller.SenseFood(food_location);

            // Move both the agents
            AgentReceiver.Step(StepSize);
            AgentSignaller.Step(StepSize);

            // clam the signallers position
            double pos = AgentSignaller.GetPosition();
            
            if (pos > LANDMARKZONESTART){
                AgentSignaller.SetPosition(LANDMARKZONESTART);
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

        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            savedStateSignallerTester[i] = AgentSignaller.NervousSystem.NeuronState(i);
        }
        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            savedStateReceiverTester[i] = AgentReceiver.NervousSystem.NeuronState(i);
        }

        // ******* TESTING *********

        for (int trial = 0; trial < positions.size(); trial ++){
            // Reset
            AgentReceiver.SetPosition(0);
            AgentSignaller.SetPosition(0);
            AgentReceiver.ResetSensors();
            AgentSignaller.ResetSensors();

            food_location = landmarkPositions[trial][env];
            for (int x = 1; x<= LN; x += 1){
                LandmarkFile << landmarkPositions[trial][x] << " ";
            }
            LandmarkFile << food_location << " ";

            for (int i= 1; i<=N; i++){
                AgentReceiver.NervousSystem.SetNeuronState(i, savedStateReceiverTester[i]);
            }
            for (int i= 1; i<=N; i++){
                AgentSignaller.NervousSystem.SetNeuronState(i, savedStateSignallerTester[i]);
            }

            // ********* PHASE 2/3 ************

            // record the initial position
            SignallerBehaviorFile2 << AgentSignaller.GetPosition() << " ";
            RecieverBehaviorFile2 << AgentReceiver.GetPosition() << " ";
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

                AgentReceiver.SenseLandmarks(LN, landmarkPositions[trial]);

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
                    double lengthZone = landmarkPositions[trial][LN] - landmarkPositions[trial][1];
                    
                    if (scoringTime > 0){
                        Stage3Fitness << 1 - ((totalScore/scoringTime)/lengthZone) << " ";
                    }
                }

                SignallerBehaviorFile2 << AgentSignaller.GetPosition() << " ";
                RecieverBehaviorFile2 << AgentReceiver.GetPosition() << " ";
                
                // record the neural state at the end of each time step 
                NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
                NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
                NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";
                NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
                NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
                NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";
            }

            // length of the landmark zone
            double lengthZone = landmarkPositions[trial][LN] - landmarkPositions[trial][1];

            // END OF TRIAL
            // score at the end of this trial
            double fitness = 0;
            if (scoringTime > 0){
                fitness = 1 - ((totalScore/scoringTime)/lengthZone);
                if (fitness < 0.0){
                    fitness = 0.0;
                }
            }
            total_fitness += fitness;
            total_trials +=1;

            if (total_trials > 0){
                TotalFitness << total_fitness / total_trials << " ";
            } else{
                TotalFitness << 0 << " ";
            }

            // new line for a new trial
            Stage3Fitness << endl;
            LandmarkFile << endl;

            SignallerBehaviorFile2 << endl;
            NeuronS1 << endl;
            NeuronS2 << endl;
            NeuronS3 << endl;

            RecieverBehaviorFile2 << endl;
            NeuronR1 << endl;
            NeuronR2 << endl;
            NeuronR3 << endl;

            TotalFitness << endl;
            
        }
        // close the files for that env
        SignallerBehaviorFile.close();
        RecieverBehaviorFile.close();
        SignallerBehaviorFile2.close();
        RecieverBehaviorFile2.close();
        Stage3Fitness.close();
        LandmarkFile.close();
        NeuronS1.close();
        NeuronS2.close();
        NeuronS3.close();
        NeuronR1.close();
        NeuronR2.close();
        NeuronR3.close();
    }
    TotalFitness.close();
    return (total_trials > 0) ? total_fitness / total_trials : 0.0;
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

void output_config(std::string dir, std::string run, std::string batch){
    // open the file
    ofstream configfile;
    configfile.open (dir + "config_" + run + ".dat");

    // print the run information
    configfile << "*********CONFIG FILE *********" << "\n" << "\n"
            << "*********Housekeeping Info*********" << "\n" << "\n"
            << "Date of experiement: " << date_time_as_string() << "\n" << "\n"
            << "Run number " << run << " as part of slurm batch "+ batch << "\n" << "\n"
            << "*********Parameter Info*********" << "\n" << "\n"
            << "LN: " << LN << "\n"
            << "StepSize: " << StepSize << "\n"
            << "RunDuration: " <<  RunDuration << "\n"
            << "TransDuration: " << TransDuration << "\n"
            << "Min Length: " << ArenaLength << "\n"
            << "mindist: " << mindist << "\n"
            << "\n"
            << "Population Size: " << POPSIZE << "\n"
            << "Num Generations: " << GENS << "\n"
            << "Mutation rate: " << MUTVAR << "\n"
            << "Crossover Probability: " << CROSSPROB << "\n"
            << "Expected: " << EXPECTED << "\n"
            << "Elitism: " << ELITISM << "\n"
            << "\n*********Nervious System Parameters*********\n"
            << "N: " << N << "\n"
            << "WR: " << WR << "\n"
            << "SR: " << SR << "\n"
            << "BR: " << BR << "\n"
            << "TMIN: " << TMIN << "\n"
            << "TMAX: " << TMAX << "\n"
            << "\n"
            ;
    configfile.close();
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
    // check that argv[1] has been provided
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
    std::string result_dir = "/user/work/yj23812/BeeCommunication/results/"+ date_as_string() +"/" + slurm_job_id;
    // std::string result_dir = "/Users/katiepambakian/Documents/BSc Computer Science/Y3/Dissertation/BeeCommunication/results/"+ date_as_string() +"/v";

    dir = result_dir +"/batch_"+ batch_number +"/run_"+ current_run +"/";
    // there is not acutally an error here
    std::filesystem::create_directories(dir);

    // print to the config file
    output_config(dir, current_run, batch_number);
    
    // Random seed -> unique to each run
    long randomseed = static_cast<long>(time(NULL));
    randomseed += atoi(argv[1]);

    // save the seed to a file
    ofstream seedfile;
    seedfile.open (dir + "seed_" + current_run + ".dat");
    seedfile << randomseed << endl;
    seedfile.close();

    // if logging is enabled
    #ifdef PRINTTOFILE
        // redirect all output to that file
        ofstream file;
        file.open  (dir + "evol_" + current_run + ".dat");
        cout.rdbuf(file.rdbuf());
    #endif

    // Create the search Object with length VectSize
    TSearch search(VectSize);
    
    // Configure the search
    search.SetRandomSeed(randomseed);
    search.SetDir(dir);
    search.SetCurrentRun(current_run);
    search.SetSearchResultsDisplayFunction(ResultsDisplay);
    search.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
    search.SetSelectionMode(RANK_BASED);
    search.SetReproductionMode(GENETIC_ALGORITHM);
    search.SetPopulationSize(POPSIZE);
    search.SetMaxGenerations(GENS);
    search.SetCrossoverProbability(CROSSPROB);
    search.SetCrossoverMode(UNIFORM);
    search.SetMutationVariance(MUTVAR);
    search.SetMaxExpectedOffspring(EXPECTED);
    search.SetElitistFraction(ELITISM);
    search.SetSearchConstraint(1);

    /* Initialize and seed the search */
    // get a random population
    search.InitializeSearch();
    
    /* Evolve */

    // Stage 0: Evolve the Signaller
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(signaller_to_food);
    search.ExecuteSearch();

    // Stage 1: A Slightly Easier task (simpler landmark generation)
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(Fitness1);
    search.ExecuteSearch();

    // Stage 2: Full Task
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(Fitness2);
    search.ExecuteSearch();

    // Stage 2: Full Task
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(Fitness3);
    search.ExecuteSearch();

    // Stage 2: Full Task
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(Fitness4);
    search.ExecuteSearch();

    if (search.BestPerformance() > 0.99) {
        RecordBehavior2(search, search.getRandomState());
    }

    #ifdef PRINTTOFILE
        file.close();
    #endif

    return 0;
}
