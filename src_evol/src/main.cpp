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
const double HarshDuration = 30.0;
// this was originally 50 -> length of the arena
const double ArenaLength = LANDMARKZONEEND - SIGNALLERSTART;      
const double mindist = 3.0;     

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

// ------------------------------------
// scoring functions around an object
// returns a double of the score between 0 and 1
// like a discrete normal distribution based on distance
// distance of 0 = perfect score
// the larger the distance the worse the score
// ------------------------------------

// EASY MODE 
double easy_score(double distance){
    
    // calculate the scoring zone
    double landmark_zone_length = LANDMARKZONEEND - LANDMARKZONESTART;
    double score_zone_length = landmark_zone_length / N;
    double one_side = score_zone_length / 2;
    
    double one_segment = one_side / 3;

    if (distance <= one_segment){
        return 1;
    } 
    else if (distance <= one_segment*2){
        return 0.7;
    }
    else if (distance <= one_segment*3){
        return 0.3;
    }
    return 0;
}

double normal_distribution(double sigma, double x){
    return std::exp(-(x * x) / (2 * sigma * sigma));
}

// HARD MODE 
double hard_score(double distance){
    // calculate the scoring zone
    double landmark_zone_length = LANDMARKZONEEND - LANDMARKZONESTART;
    double score_zone_length = landmark_zone_length / N;
    double one_side = score_zone_length / 2;
    double one_segment = one_side / 4;

    if (distance <= one_segment){
        return 1;
    } 
    else if (distance <= one_segment*2){
        return 0.75;
    }
    else if (distance <= one_segment*3){
        return 0.5;
    }
    else if (distance <= one_segment*4){
        return 0.25;
    }
    return 0;
}

double score(double position, double distance){
    // check to see if the agent is out of bounds (return 0)

    if (position > 15 || position < -5){
        return 0;
    } else{
        double landmark_zone_length = LANDMARKZONEEND - LANDMARKZONESTART;
        double score_zone_length = landmark_zone_length / N;
        
        double sigma = score_zone_length;
        return normal_distribution(sigma, distance);
    }
}

// ---------------------------------------------------------
// stage 1: Evolution of receiver
// only looking at the distance between the receiver and the 
// flower/food
// no signaller in the task
// the sensor of the receiver is set to correspond to which
// of the landmarks the receiver should go to
// ---------------------------------------------------------

double stage1(TVector<double> &genotype, RandomState &rs){


    // **** create the receiver ****
    // Map genotype to phenotype
    TVector<double> phenotypeReceiver;
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    // N*N +5N is the size of one agents parameter set
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentReceiver(N, phenotypeReceiver);
    AgentReceiver.SetPosition(0);
    AgentReceiver.ResetNeuralState();
    
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
            AgentReceiver.SetPosition(0);
            AgentReceiver.ResetNeuralState();
            AgentReceiver.SetOther(start+step*(env-1));

            double scoringTime = 0.0;
            double totalScore = 0.0;

            // Let the Receiver and Signaller Explore the enviornment (for 150) -> not scored
            for (double time=0; time < RunDuration; time += StepSize){
                AgentReceiver.SenseLandmarks(LN, landmarkPositions);
                AgentReceiver.SetOther(start+step*env);
                AgentReceiver.Step(StepSize);

                if (time > TransDuration){
                    distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);

                    // if the distance is within a threshold set the score to be perfect
                    if (distance_food_receiver < 1 && time > HarshDuration){
                        distance_food_receiver = 0;
                    } else if (distance_food_receiver < mindist){
                        distance_food_receiver = 0;
                    }
                    totalScore += distance_food_receiver;
                    scoringTime += 1;
                }
            }
            // END OF TRIAL
            // score at the end of this trial
            double fitness = 1 - ((totalScore/scoringTime)/ArenaLength);
            if (fitness < 0.0){
                fitness = 0.0;
            }
            total_fitness += fitness;
            total_trials +=1;
        }
    }
    return total_fitness / total_trials;
}


double task_signaller_food_to_home(TVector<double> &genotype, RandomState &rs){
    
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    CountingAgent AgentSignaler(N, phenotypeSignaller);

    // **** Generate landmarks (using leapfrog method) ****
    TVector<double> landmarkPositions;

    // trial variables 
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_receiver_signaller;

    for (int i=0; i<3; i++){
        genLandmarks_LeapFrog(rs, landmarkPositions);
        
        // set each of the landmarks to be the location of the food
        for (int env =1; env<=LN;env++){
            // set the food to be at the ith landmark location
            food_location = landmarkPositions[env];

            // Initialise the agent for this trial
            AgentSignaler.SetPosition(food_location);
            AgentSignaler.ResetNeuralState();
            AgentSignaler.ResetSensors();

            double scoringTime = 0.0;
            double totalScore = 0.0;

            for (double time=0; time < RunDuration; time += StepSize){
                // only let the Signaller move
                AgentSignaler.SenseFood(food_location);
                AgentSignaler.SenseLandmarks(LN, landmarkPositions);
                AgentSignaler.SenseOther(0);
                AgentSignaler.Step(StepSize);

                if (time > TransDuration){
                    distance_receiver_signaller = fabs(0 - AgentSignaler.GetPosition());

                    // if the distance is within a threshold set the score to be perfect
                    if (distance_receiver_signaller < 1 && time > HarshDuration){
                        distance_receiver_signaller = 0;
                    } else if (distance_receiver_signaller < mindist){
                        distance_receiver_signaller = 0;
                    }
                    totalScore += distance_receiver_signaller;
                    scoringTime += 1;
                }
            }
            // END OF TRIAL
            // score at the end of this trial
            double fitness = 1 - ((totalScore/scoringTime)/ArenaLength);
            if (fitness < 0.0){
                fitness = 0.0;
            }
            total_fitness += fitness;
            total_trials +=1;
        }
    }
    return total_fitness / total_trials;
}

// ---------------------------------------------------------
// stage 2: Evolution of signaller
// only looking a the distance between the Signaller and the
// receiver (if that is equal to the distance of the landmark)
// ---------------------------------------------------------
double stage2(TVector<double> &genotype, RandomState &rs){
    // trial variables 
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_receiver_signaller;

    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    CountingAgent AgentSignaler(N, phenotypeSignaller);
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // save the state of the receiver
    // save the state of the signaller and receiver
    TVector<double> savedStateReceiver;
    savedStateReceiver.SetBounds(1, N);
    for (int i = 1; i <= N; i++)
    {
        savedStateReceiver[i] = AgentReceiver.NervousSystem.NeuronState(i);
    }

    // **** Generate landmarks (using leapfrog method) ****
    TVector<double> landmarkPositions;

    for (int i=0; i<3; i++){
        genLandmarks_LeapFrog(rs, landmarkPositions);
        
        // set each of the landmarks to be the location of the food
        for (int env =1; env<=LN;env++){
            // set the food to be at the ith landmark location
            food_location = landmarkPositions[env];

            // Initialise the agent for this trial
            int location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
            AgentSignaler.SetPosition(location);
            AgentSignaler.ResetNeuralState();
            AgentSignaler.ResetSensors();

            // PHASE 1
            double scoringTime_phase1 = 0.0;
            double totalScore_phase1 = 0.0;

            for (double time=0; time < RunDuration; time += StepSize){
                // only let the Signaller move
                AgentSignaler.SenseFood(food_location);
                AgentSignaler.SenseLandmarks(LN, landmarkPositions);
                AgentSignaler.SenseOther(0);
                AgentSignaler.Step(StepSize);

                if (time >= TransDuration){
                    distance_receiver_signaller = fabs(food_location - AgentSignaler.GetPosition());

                    // if the distance is within a threshold set the score to be perfect
                    if (distance_receiver_signaller < 1 && time > HarshDuration){
                        distance_receiver_signaller = 0;
                    } else if (distance_receiver_signaller < mindist){
                        distance_receiver_signaller = 0;
                    }
                    totalScore_phase1 += distance_receiver_signaller;
                    scoringTime_phase1 += 1;
                }
            }

            // PHASE 2
            AgentSignaler.SetPosition(food_location);
            double scoringTime_phase2 = 0.0;
            double totalScore_phase2 = 0.0;
            AgentSignaler.ResetSensors();

            for (double time=0; time < RunDuration; time += StepSize){
                // only let the Signaller move
                AgentSignaler.SenseFood(food_location);
                AgentSignaler.SenseLandmarks(LN, landmarkPositions);
                AgentSignaler.SenseOther(0);
                AgentSignaler.Step(StepSize);

                if (time >= TransDuration){
                    distance_receiver_signaller = fabs(0 - AgentSignaler.GetPosition());

                    // if the distance is within a threshold set the score to be perfect
                    if (distance_receiver_signaller < 1 && time > HarshDuration){
                        distance_receiver_signaller = 0;
                    } else if (distance_receiver_signaller < mindist){
                        distance_receiver_signaller = 0;
                    }
                    totalScore_phase2 += distance_receiver_signaller;
                    scoringTime_phase2 += 1;
                }
            }

            // END OF TRIAL
            // score at the end of this trial
            // averaged score
            double avg1 = 0.0;
            double avg2 = 0.0;

            if (scoringTime_phase1 >0){
                avg1 = ((totalScore_phase1/scoringTime_phase1)/ArenaLength);
            }

            if (scoringTime_phase2 > 0){
                avg2 =((totalScore_phase2/scoringTime_phase2)/ArenaLength);
            }

            double average_both_trials = (avg1 + avg2)/2;

            double fitness = 1 - average_both_trials;
            if (fitness < 0.0){
                fitness = 0.0;
            }
            total_fitness += fitness;
            total_trials +=1;
        }
    }

    // restore the state of the receiver
    for (int i = 1; i <= N; i++)
    {
        AgentReceiver.NervousSystem.SetNeuronState(i, savedStateReceiver[i]);
    
    }

    return total_fitness / total_trials;
}

double stage3(TVector<double> &genotype, RandomState &rs){
    // initalise their genotypes 
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    CountingAgent AgentSignaller(N, phenotypeSignaller);
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // save the state of the signaller and receiver
    TVector<double> savedstateSignaller, savedstateReceiver;
    savedstateSignaller.SetBounds(1,N); 
    savedstateReceiver.SetBounds(1, N);
    for (int i = 1; i <= N; i++)
    {
        savedstateSignaller[i] = AgentSignaller.NervousSystem.NeuronState(i);
    }
    for (int i = 1; i <= N; i++)
    {
        savedstateReceiver[i] = AgentReceiver.NervousSystem.NeuronState(i);
    }

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
    double distance_signaller_receiver;

    for (int i=0; i<3; i++){
        genLandmarks_LeapFrog(rs, landmarkPositions);
        // set each of the landmarks to be the location of the food
        for (int env =1; env<=LN;env++){
            // set the food to be at the ith landmark location
            food_location = landmarkPositions[env];

            // ********** PHASE 1 **********
            // ********** PHASE 2 **********
            // setup 
            AgentReceiver.SetPosition(0);
            AgentReceiver.ResetSensors();
            int location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
            AgentSignaller.SetPosition(location);
            // restore the signaller agents state
            for (int i = 1; i <= N; i++)
            {
                AgentSignaller.NervousSystem.SetNeuronState(i, savedstateSignaller[i]);
            }

            double scoringTime_phase2 = 0.0;
            double totalScore_phase2 = 0.0;

            for (double time=0; time < RunDuration; time += StepSize){
                // Receiver see environemnt 
                AgentReceiver.SenseLandmarks(LN, landmarkPositions);
                AgentReceiver.SenseOther(AgentSignaller.GetPosition());
                // Signaller see environemnt 
                AgentSignaller.SenseFood(food_location);
                AgentSignaller.SenseLandmarks(LN, landmarkPositions);
                AgentSignaller.SenseOther(AgentReceiver.GetPosition());

                // Move both the agents
                AgentReceiver.Step(StepSize);
                AgentSignaller.Step(StepSize);

                if (time > TransDuration){

                    // distance between the receiver and the food
                    distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);
                    
                    if (distance_food_receiver < 1 && time > HarshDuration){
                        distance_food_receiver = 0;
                    } else if (distance_food_receiver < mindist){
                        distance_food_receiver = 0;
                    }

                    // distance between the signaller and receiver

                    distance_signaller_receiver = fabs(AgentReceiver.GetPosition() - AgentSignaller.GetPosition());

                    // it has to be one away 
                    if (distance_signaller_receiver < 1.5 && distance_signaller_receiver >= 0.5){
                        distance_signaller_receiver = 0;
                    }

                    // average distance
                    totalScore_phase2 += (distance_food_receiver + distance_signaller_receiver) /2;
                    scoringTime_phase2 += 1;
                }
            }

            double fitness = 1 - ((totalScore_phase2/scoringTime_phase2)/ArenaLength);
            if (fitness < 0.0){
                fitness = 0.0;
            }
            total_fitness += fitness;
            total_trials +=1;
        }
    }
    return total_fitness / total_trials;
}

double RecordBehaviorStage1(TSearch &s, RandomState &rs){
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();

    // **** create the files to save the behaviour/env ***
    // storing them all in the same file, each gets a new line (so 9 trials per file)
    // the stages are separated into three files
    ofstream ReceiverBehaviorFile1;
    // SignalerBehaviorFile1.open( dir + "behavior_Signaler_stage1_" + current_run + ".dat");
    ReceiverBehaviorFile1.open( dir + "behavior_Receiver_stage1_" + current_run + ".dat");

    ofstream Stage1Fitness;
    Stage1Fitness.open( dir + "fitness_stage1_" + current_run +".dat");
    
    // stores the location of the landmarks and the food -> this is each line 
    ofstream LandmarkFile;
    LandmarkFile.open(dir + "landmark_location_stage1_"+current_run+".dat");

    // files to save the neuron state
    ofstream Neuron1, Neuron2, Neuron3;
    Neuron1.open(dir + "neuron1_stage1_"+current_run+".dat");
    Neuron2.open(dir + "neuron2_stage1_"+current_run+".dat");
    Neuron3.open(dir + "neuron3_stage1_"+current_run+".dat");

    // trial variables 
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_food_receiver;

    // **** create the receiver ****
    // Map genotype to phenotype
    TVector<double> genotype;
    genotype = s.BestIndividual();

    // Map genotype to phenotype
    TVector<double> phenotypeReceiver;
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    // N*N +5N is the size of one agents parameter set
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // save this state
    // Save state
    TVector<double> savedstateReceiver;
    savedstateReceiver.SetBounds(1,N);

    // Saved each of their neural states 
    for (int i = 1; i <= N; i++)
    {
        savedstateReceiver[i] = AgentReceiver.NervousSystem.NeuronState(i);
    }

    // **** Generate landmarks (using leapfrog method) ****
    TVector<double> landmarkPositions;

    // calculating what the set other should be (value between 0.5 and 1.5)
    double step = 1.0/(LN-1);
    double start = 0.5;

    for (int i=0; i<3; i++){
        genLandmarks_LeapFrog(rs, landmarkPositions);
        
        // set each of the landmarks to be the location of the food
        for (int env =1; env<=LN;env++){
            AgentReceiver.ResetSensors();
            // Reset neural state
            for (int i = 1; i <= N; i++)
            {
                AgentReceiver.NervousSystem.SetNeuronState(i, savedstateReceiver[i]);
            
            }
            // set the food to be at the ith landmark location
            food_location = landmarkPositions[env];
            for (int i = 1; i <= LN; i += 1){
                LandmarkFile << landmarkPositions[i] << " ";
            }
            LandmarkFile << food_location << " ";

            // Initialise the agent for this trial
            AgentReceiver.SetPosition(0);
            AgentReceiver.ResetNeuralState();
            AgentReceiver.SetOther(start+step*env);

            double scoringTime = 0.0;
            double totalScore = 0.0;

            // record the initial position
            ReceiverBehaviorFile1 << AgentReceiver.GetPosition() << " ";

            // Let the Receiver and Signaller Explore the enviornment (for 150) -> not scored
            for (double time=0; time < RunDuration; time += StepSize){
                AgentReceiver.SenseLandmarks(LN, landmarkPositions);
                AgentReceiver.SetOther(start+step*env);
                AgentReceiver.Step(StepSize);

                if (time >= TransDuration){
                    distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);

                    // if the distance is within a threshold set the score to be perfect
                    if (distance_food_receiver < mindist){
                        distance_food_receiver = 0;
                    }

                    totalScore += distance_food_receiver;
                    scoringTime += 1;
                }
                
                ReceiverBehaviorFile1 << AgentReceiver.GetPosition() << " ";
                if (scoringTime > 0){
                    Stage1Fitness << 1 - ((totalScore/scoringTime)/ArenaLength) << " ";
                }
                // record the neural state at the end of each time step 
                Neuron1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
                Neuron2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
                Neuron3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";
            }

            // END OF TRIAL
            // score at the end of this trial
            double fitness = 1 - ((totalScore/scoringTime)/ArenaLength);
            if (fitness < 0.0){
                fitness = 0.0;
            }
            total_fitness += fitness;
            total_trials +=1;

            // new line for a new trial
            ReceiverBehaviorFile1 << endl;
            Stage1Fitness << endl;
            LandmarkFile << endl;
            Neuron1 << endl;
            Neuron2 << endl;
            Neuron3 << endl;
        }
    }
        // close the files for that env
    ReceiverBehaviorFile1.close();
    Stage1Fitness.close();
    LandmarkFile.close();
    Neuron1.close();
    Neuron2.close();
    Neuron3.close();
    return total_fitness / total_trials;
}

double RecordBehaviorStage1Half(TSearch &s, RandomState &rs){
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();

    // **** create the files to save the behaviour/env ***
    // storing them all in the same file, each gets a new line (so 9 trials per file)
    // the stages are separated into three files
    ofstream SignallerBehaviorFile1;
    SignallerBehaviorFile1.open( dir + "behavior_Signaller_stage1half_" + current_run + ".dat");

    ofstream Stage2Fitness;
    Stage2Fitness.open( dir + "fitness_stage1half_" + current_run +".dat");
    
    // stores the location of the landmarks and the food -> this is each line 
    ofstream LandmarkFile;
    LandmarkFile.open(dir + "landmark_location_stage1half_"+current_run+".dat");

    // files to save the neuron state
    ofstream Neuron1, Neuron2, Neuron3;
    Neuron1.open(dir + "signaller_neuron1_stage1half_"+current_run+".dat");
    Neuron2.open(dir + "signaller_neuron2_stage1half_"+current_run+".dat");
    Neuron3.open(dir + "signaller_neuron3_stage1half_"+current_run+".dat");

    // trial variables 
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_receiver_signaller;

    // **** create the receiver ****
    // Map genotype to phenotype
    TVector<double> genotype;
    genotype = s.BestIndividual();

    // Map genotype to phenotype
    TVector<double> phenotypeSignaller;
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    CountingAgent AgentSignaller(N, phenotypeSignaller);

    // save this state
    // Save state
    TVector<double> savedStateSignaller;
    savedStateSignaller.SetBounds(1,N);

    // Saved each of their neural states 
    for (int i = 1; i <= N; i++)
    {
        savedStateSignaller[i] = AgentSignaller.NervousSystem.NeuronState(i);
    }

    // **** Generate landmarks (using leapfrog method) ****
    TVector<double> landmarkPositions;

    TVector<double> phenotypeReceiver;
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentReceiver(N, phenotypeReceiver);


    // calculating what the set other should be (value between 0.5 and 1.5)
    double step = 1.0/(LN-1);
    double start = 0.5;

    for (int i=0; i<3; i++){
        genLandmarks_LeapFrog(rs, landmarkPositions);

        // set each of the landmarks to be the location of the food
        for (int env =1; env<=LN;env++){
            // set the food to be at the ith landmark location
            food_location = landmarkPositions[env];
            for (int i = 1; i <= LN; i += 1){
                LandmarkFile << landmarkPositions[i] << " ";
            }
            LandmarkFile << food_location << " ";

            AgentSignaller.ResetSensors();
            // Reset neural state
            for (int i = 1; i <= N; i++)
            {
                AgentSignaller.NervousSystem.SetNeuronState(i, savedStateSignaller[i]);
            
            }
            // Initialise the agent for this trial
            AgentSignaller.SetPosition(food_location);

            double scoringTime = 0.0;
            double totalScore = 0.0;

            // record the initial position
            SignallerBehaviorFile1 << AgentSignaller.GetPosition() << " ";

            // Let the Receiver and Signaller Explore the enviornment (for 150) -> not scored
            for (double time=0; time < RunDuration; time += StepSize){
                AgentSignaller.SenseFood(food_location);
                AgentSignaller.SenseLandmarks(LN, landmarkPositions);
                AgentSignaller.SenseOther(0);
                AgentSignaller.Step(StepSize);

                if (time >= TransDuration){
                    distance_receiver_signaller = fabs(0 - AgentSignaller.GetPosition());

                    // if the distance is within a threshold set the score to be perfect
                    if (distance_receiver_signaller < 1 && time > HarshDuration){
                        distance_receiver_signaller = 0;
                    } else if (distance_receiver_signaller < mindist){
                        distance_receiver_signaller = 0;
                    }
                    totalScore += distance_receiver_signaller;
                    scoringTime += 1;
                }
                
                SignallerBehaviorFile1 << AgentSignaller.GetPosition() << " ";
                
                if (scoringTime > 0){
                    Stage2Fitness << 1 - ((totalScore/scoringTime)/ArenaLength) << " ";
                }

                // record the neural state at the end of each time step 
                
                Neuron1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
                Neuron2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
                Neuron3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";

            }

            // END OF TRIAL
            // score at the end of this trial
            double fitness = 1 - ((totalScore/scoringTime)/ArenaLength);
            if (fitness < 0.0){
                fitness = 0.0;
            }
            total_fitness += fitness;
            total_trials +=1;

            // new line for a new trial
            SignallerBehaviorFile1 << endl;
            Stage2Fitness << endl;
            LandmarkFile << endl;
            Neuron1 << endl;
            Neuron2 << endl;
            Neuron3 << endl;
        }
    }
        // close the files for that env
    SignallerBehaviorFile1.close();
    Stage2Fitness.close();
    LandmarkFile.close();
    Neuron1.close();
    Neuron2.close();
    Neuron3.close();
    return total_fitness / total_trials;
}

double RecordBehaviorStage2(TSearch &s, RandomState &rs){
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();

    // **** create the files to save the behaviour/env ***
    // storing them all in the same file, each gets a new line (so 9 trials per file)
    // the stages are separated into three files
    ofstream SignallerBehaviorFile1;
    SignallerBehaviorFile1.open( dir + "behavior_Signaller_stage2_" + current_run + ".dat");

    ofstream Stage2Fitness;
    Stage2Fitness.open( dir + "fitness_stage2_" + current_run +".dat");
    
    // stores the location of the landmarks and the food -> this is each line 
    ofstream LandmarkFile;
    LandmarkFile.open(dir + "landmark_location_stage2_"+current_run+".dat");

    // files to save the neuron state
    ofstream Neuron1, Neuron2, Neuron3;
    Neuron1.open(dir + "signaller_neuron1_stage2_"+current_run+".dat");
    Neuron2.open(dir + "signaller_neuron2_stage2_"+current_run+".dat");
    Neuron3.open(dir + "signaller_neuron3_stage2_"+current_run+".dat");

    // trial variables 
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_receiver_signaller;

    // **** create the receiver ****
    // Map genotype to phenotype
    TVector<double> genotype;
    genotype = s.BestIndividual();

    // Map genotype to phenotype
    TVector<double> phenotypeSignaller;
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    CountingAgent AgentSignaller(N, phenotypeSignaller);

    // save this state
    // Save state
    TVector<double> savedStateSignaller;
    savedStateSignaller.SetBounds(1,N);

    // Saved each of their neural states 
    for (int i = 1; i <= N; i++)
    {
        savedStateSignaller[i] = AgentSignaller.NervousSystem.NeuronState(i);
    }

    // **** Generate landmarks (using leapfrog method) ****
    TVector<double> landmarkPositions;

    TVector<double> phenotypeReceiver;
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentReceiver(N, phenotypeReceiver);


    // calculating what the set other should be (value between 0.5 and 1.5)
    double step = 1.0/(LN-1);
    double start = 0.5;

    for (int i=0; i<3; i++){
        genLandmarks_LeapFrog(rs, landmarkPositions);

        // set each of the landmarks to be the location of the food
        for (int env =1; env<=LN;env++){
            // set the food to be at the ith landmark location
            food_location = landmarkPositions[env];
            for (int i = 1; i <= LN; i += 1){
                LandmarkFile << landmarkPositions[i] << " ";
            }
            LandmarkFile << food_location << " ";

            AgentSignaller.ResetSensors();
            // Reset neural state
            for (int i = 1; i <= N; i++)
            {
                AgentSignaller.NervousSystem.SetNeuronState(i, savedStateSignaller[i]);
            
            }
            // Initialise the agent for this trial
            int location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
            AgentSignaller.SetPosition(location);

            // record the initial position
            SignallerBehaviorFile1 << AgentSignaller.GetPosition() << " ";

            // PHASE 1
            double scoringTime_phase1 = 0.0;
            double totalScore_phase1 = 0.0;

            for (double time=0; time < RunDuration; time += StepSize){
                // only let the Signaller move
                AgentSignaller.SenseFood(food_location);
                AgentSignaller.SenseLandmarks(LN, landmarkPositions);
                AgentSignaller.SenseOther(0);
                AgentSignaller.Step(StepSize);

                if (time > TransDuration){
                    distance_receiver_signaller = fabs(food_location- AgentSignaller.GetPosition());

                    // if the distance is within a threshold set the score to be perfect
                    if (distance_receiver_signaller < 1 && time > HarshDuration){
                        distance_receiver_signaller = 0;
                    } else if (distance_receiver_signaller < mindist){
                        distance_receiver_signaller = 0;
                    }
                    totalScore_phase1 += distance_receiver_signaller;
                    scoringTime_phase1 += 1;
                }

                SignallerBehaviorFile1 << AgentSignaller.GetPosition() << " ";
                
                if (scoringTime_phase1 > 0){
                    Stage2Fitness << 1 - ((totalScore_phase1/scoringTime_phase1)/ArenaLength) << " ";
                }

                // record the neural state at the end of each time step 
                
                Neuron1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
                Neuron2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
                Neuron3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";
            }

            // PHASE 2
            double scoringTime_phase2 = 0.0;
            double totalScore_phase2 = 0.0;
            AgentSignaller.SetPosition(food_location);
            AgentSignaller.ResetSensors();

            // Let the Receiver and Signaller Explore the enviornment (for 150) -> not scored
            for (double time=0; time < RunDuration; time += StepSize){
                AgentSignaller.SenseFood(food_location);
                AgentSignaller.SenseLandmarks(LN, landmarkPositions);
                AgentSignaller.SenseOther(0);
                AgentSignaller.Step(StepSize);

                if (time >= TransDuration){
                    distance_receiver_signaller = fabs(0 - AgentSignaller.GetPosition());

                    // if the distance is within a threshold set the score to be perfect
                    if (distance_receiver_signaller < 1 && time > HarshDuration){
                        distance_receiver_signaller = 0;
                    } else if (distance_receiver_signaller < mindist){
                        distance_receiver_signaller = 0;
                    }
                    totalScore_phase2 += distance_receiver_signaller;
                    scoringTime_phase2 += 1;
                }
                
                SignallerBehaviorFile1 << AgentSignaller.GetPosition() << " ";
                
                if (scoringTime_phase2 > 0){
                    Stage2Fitness << 1 - ((totalScore_phase2/scoringTime_phase2)/ArenaLength) << " ";
                }

                // record the neural state at the end of each time step 
                
                Neuron1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
                Neuron2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
                Neuron3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";

            }

            // END OF TRIAL
            // score at the end of this trial
            double avg1 = 0.0;
            double avg2 = 0.0;

            if (scoringTime_phase1 >0){
                avg1 = ((totalScore_phase1/scoringTime_phase1)/ArenaLength);
            }

            if (scoringTime_phase2 > 0){
                avg2 =((totalScore_phase2/scoringTime_phase2)/ArenaLength);
            }

            double average_both_trials = (avg1 + avg2)/2;

            double fitness = 1 - average_both_trials;
            if (fitness < 0.0){
                fitness = 0.0;
            }
            total_fitness += fitness;
            total_trials +=1;

            // new line for a new trial
            SignallerBehaviorFile1 << endl;
            Stage2Fitness << endl;
            LandmarkFile << endl;
            Neuron1 << endl;
            Neuron2 << endl;
            Neuron3 << endl;
        }
    }
    
    // close the files for that env
    SignallerBehaviorFile1.close();
    Stage2Fitness.close();
    LandmarkFile.close();
    Neuron1.close();
    Neuron2.close();
    Neuron3.close();
    return total_fitness / total_trials;
}

double RecordBehaviorStage3(TSearch &s, RandomState &rs){
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();

    // **** create the files to save the behaviour/env ***
    // storing them all in the same file, each gets a new line (so 9 trials per file)
    // the stages are separated into three files
    ofstream SignallerBehaviorFile, RecieverBehaviorFile;
    SignallerBehaviorFile.open( dir + "behavior_Signaller_stage3_" + current_run + ".dat");
    RecieverBehaviorFile.open( dir + "behavior_Reciever_stage3_" + current_run + ".dat");
    ofstream Stage3Fitness;
    Stage3Fitness.open( dir + "fitness_stage3_" + current_run +".dat");
    
    // stores the location of the landmarks and the food -> this is each line 
    ofstream LandmarkFile;
    LandmarkFile.open(dir + "landmark_location_stage3_"+current_run+".dat");

    // files to save the neuron state
    ofstream NeuronS1, NeuronS2, NeuronS3;
    NeuronS1.open(dir + "signaller_neuron1_stage3_"+current_run+".dat");
    NeuronS2.open(dir + "signaller_neuron2_stage3_"+current_run+".dat");
    NeuronS3.open(dir + "signaller_neuron3_stage3_"+current_run+".dat");
    ofstream NeuronR1, NeuronR2, NeuronR3;
    NeuronR1.open(dir + "receiver_neuron1_stage3_"+current_run+".dat");
    NeuronR2.open(dir + "receiver_neuron2_stage3_"+current_run+".dat");
    NeuronR3.open(dir + "receiver_neuron3_stage3_"+current_run+".dat");

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

    // **** Generate landmarks (using leapfrog method) ****
    TVector<double> landmarkPositions;

    // calculating what the set other should be (value between 0.5 and 1.5)
    for (int i=0; i<3; i++){
        genLandmarks_LeapFrog(rs, landmarkPositions);
        // set each of the landmarks to be the location of the food
        for (int env =1; env<=LN;env++){
            // set the food to be at the ith landmark location
            food_location = landmarkPositions[env];
            for (int i = 1; i <= LN; i += 1){
                LandmarkFile << landmarkPositions[i] << " ";
            }
            LandmarkFile << food_location << " ";

            // Setup
            int location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
            AgentReceiver.SetPosition(0);
            AgentReceiver.ResetSensors();
            AgentSignaller.SetPosition(location);
            AgentSignaller.ResetSensors();

            // Reset neural state
            for (int i = 1; i <= N; i++)
            {
                AgentSignaller.NervousSystem.SetNeuronState(i, savedStateSignaller[i]);
            
            }
            for (int i = 1; i <= N; i++)
            {
                AgentReceiver.NervousSystem.SetNeuronState(i, savedStateReceiver[i]);
            
            }

            // record the initial position
            SignallerBehaviorFile << AgentSignaller.GetPosition() << " ";
            RecieverBehaviorFile << AgentReceiver.GetPosition() << " ";

            // PHASE 2
            double scoringTime_phase2 = 0.0;
            double totalScore_phase2 = 0.0;

            for (double time=0; time < RunDuration; time += StepSize){
                // only let the Signaller move
                AgentReceiver.SenseLandmarks(LN, landmarkPositions);
                AgentReceiver.SenseOther(AgentSignaller.GetPosition());
                // Signaller see environemnt 
                AgentSignaller.SenseFood(food_location);
                AgentSignaller.SenseLandmarks(LN, landmarkPositions);
                AgentSignaller.SenseOther(AgentReceiver.GetPosition());

                // Move both the agents
                AgentReceiver.Step(StepSize);
                AgentSignaller.Step(StepSize);

                if (time > TransDuration){
                    // distance between the receiver and the food
                    distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);
                    
                    // if the distance is within a threshold set the score to be perfect
                    if (distance_food_receiver < 1 && time > HarshDuration){
                        distance_food_receiver = 0;
                    } else if (distance_food_receiver < mindist){
                        distance_food_receiver = 0;
                    }

                    distance_signaller_receiver = fabs(AgentReceiver.GetPosition() - AgentSignaller.GetPosition());

                    // it has to be one away 
                    if (distance_signaller_receiver < 1.5 && distance_signaller_receiver >= 0.5){
                        distance_signaller_receiver = 0;
                    }

                    // average distance
                    totalScore_phase2 += (distance_food_receiver + distance_signaller_receiver) /2;
                    scoringTime_phase2 += 1;
                }

                SignallerBehaviorFile << AgentSignaller.GetPosition() << " ";
                RecieverBehaviorFile << AgentReceiver.GetPosition() << " ";
                
                if (scoringTime_phase2 > 0){
                    Stage3Fitness << 1 - ((totalScore_phase2/scoringTime_phase2)/ArenaLength) << " ";
                }

                // record the neural state at the end of each time step 
                
                NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
                NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
                NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";

                NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
                NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
                NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";
            }

            // END OF TRIAL
            // score at the end of this trial
            double fitness = 1 - ((totalScore_phase2/scoringTime_phase2)/ArenaLength);
            if (fitness < 0.0){
                fitness = 0.0;
            }
            total_fitness += fitness;
            total_trials +=1;

            // new line for a new trial
            Stage3Fitness << endl;
            LandmarkFile << endl;

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
    return total_fitness / total_trials;
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
    if (argc < 3){
        // send an error message to the terminal
        std::cerr << "Error: missing run or array index number.\n";
        return 1;
    }
    std::string slurm_job_id = argv[3];
    std::string batch_number = argv[2];
    std::string current_run = argv[1];

    // Define output home directory
    int v = 0;
    // std::string result_dir = "/user/home/yj23812/BeeCommunication/results/"+ date_as_string() +"/" + slurm_job_id;
    std::string result_dir = "/Users/katiepambakian/Documents/BSc Computer Science/Y3/Dissertation/BeeCommunication/results/"+ date_as_string() +"/v";

    std::string dir = result_dir +"/batch_"+ batch_number +"/run_"+ current_run +"/";
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

    // Stage 1: Evolution of Reciver
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(stage1);
    search.ExecuteSearch();
    
    // // Stage 1.5: Evolution of Signaller (food to home)
    // search.SetSearchTerminationFunction(TerminationFunction);
    // search.SetEvaluationFunction(task_signaller_food_to_home);
    // search.ExecuteSearch();

    // Stage 2: Evolution of Signaller (home to food to home)
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(stage2);
    search.ExecuteSearch();

    // Stage 3: Join task
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(stage3);
    search.ExecuteSearch();

    // TODO: Stage 4: Testing


    if (search.BestPerformance() > 0.99) {
        RecordBehaviorStage3(search, search.getRandomState());
    }

    #ifdef PRINTTOFILE
        file.close();
    #endif

    return 0;
}
