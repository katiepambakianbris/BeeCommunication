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

// Flag
int record = 0;

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

// Repeatable components

// building agents

void CreatePhenotypes(TVector<double>& genotype, TVector<double>& phenotypeSignaller, TVector<double>& phenotypeReceiver){

    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
}


void ResetAgentsStart(CountingAgent& signaller, CountingAgent& receiver){
    signaller.SetPosition(0);
    receiver.SetPosition(0);

    signaller.ResetNeuralState();
    receiver.ResetNeuralState();

    signaller.ResetSensors();
    receiver.ResetSensors();
}

void ResetAgentsMidTrial(CountingAgent& signaller, CountingAgent& receiver){
    signaller.SetPosition(0);
    receiver.SetPosition(0);

    signaller.ResetSensors();
    receiver.ResetSensors();
}

void SaveNeuralState(CountingAgent& agent, TVector<double>& savedState){
    for (int i=1;i<=N;i++) savedState[i] = agent.NervousSystem.NeuronState(i);
}

void RestoreNeuralState(CountingAgent& agent, TVector<double>& savedState){
    for (int i=1;i<=N;i++) agent.NervousSystem.SetNeuronState(i, savedState[i]);
}

void Phase1(CountingAgent& AgentSignaller, CountingAgent& AgentReceiver, double food_location, TVector<double>& landmarkPositions){
    // only let the Signaller move
    AgentSignaller.SenseFood(food_location);
    AgentSignaller.SenseLandmarks(LN, landmarkPositions);
    AgentSignaller.SenseOther(AgentReceiver.GetPosition());

    AgentSignaller.Step(StepSize);
}

void RunPhase1(CountingAgent& AgentSignaller, CountingAgent& AgentReceiver, double food_location, TVector<double>& landmarkPositions){
    for (double time=0; time < RunDuration; time += StepSize){
        Phase1(AgentSignaller, AgentReceiver, food_location, landmarkPositions);
    }
}

void Phase2(CountingAgent& AgentSignaller, CountingAgent& AgentReceiver, double food_location, TVector<double>& landmarkPositions){
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

void RunPhase2(CountingAgent& AgentSignaller, CountingAgent& AgentReceiver, double food_location, TVector<double>& landmarkPositions){
    for (double time=0; time < RunDuration*2; time += StepSize)
    {
        Phase2(AgentSignaller, AgentReceiver, food_location, landmarkPositions);
    }
}

double Score(CountingAgent& AgentReceiver, double food_location, double time){
    double distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);

    // if the distance is within a threshold set the score to be perfect
    if (distance_food_receiver < hardDurationMinDist && time > HarshDuration+TransDuration+RunDuration){
        distance_food_receiver = 0;
    } else if (distance_food_receiver < mindist){
        distance_food_receiver = 0;
    }

    return distance_food_receiver;
}

void OpenFiles(int env, int gen,
    ofstream& SignallerBehaviorFile, ofstream& RecieverBehaviorFile,  
    ofstream& SignallerBehaviorFile2, ofstream& RecieverBehaviorFile2,
    ofstream& Fitness, ofstream& LandmarkFile,
    ofstream& NeuronS1, ofstream& NeuronS2, ofstream& NeuronS3,
    ofstream& NeuronR1, ofstream& NeuronR2, ofstream& NeuronR3
    ){

    std::string s_env = std::to_string(env);
    std::string s_gen = std::to_string(gen);
    
    SignallerBehaviorFile.open( dir + "behavior_Signaller_training_env_"+ s_env +"_" + s_gen + ".dat");
    RecieverBehaviorFile.open( dir + "behavior_Reciever_training_env_"+ s_env +"_" + s_gen + ".dat");
    SignallerBehaviorFile.open( dir + "behavior_Signaller_training_env_"+ s_env +"_" + s_gen + ".dat");
    RecieverBehaviorFile.open( dir + "behavior_Reciever_training_env_"+ s_env +"_" + s_gen + ".dat");
    
    SignallerBehaviorFile2.open( dir + "behavior_Signaller_testing_env_"+ s_env +"_" + s_gen + ".dat");
    RecieverBehaviorFile2.open( dir + "behavior_Reciever_testing_env_"+ s_env +"_" + s_gen + ".dat");
    SignallerBehaviorFile2.open( dir + "behavior_Signaller_testing_env_"+ s_env +"_" + s_gen + ".dat");
    RecieverBehaviorFile2.open( dir + "behavior_Reciever_testing_env_"+ s_env +"_" + s_gen + ".dat");

    Fitness.open( dir + "fitness_env_"+ s_env +"_" + s_gen +".dat");
    LandmarkFile.open(dir + "landmark_location_env_"+ s_env +"_"+s_gen+".dat");

    // files to save the neuron state
    
    NeuronS1.open(dir + "signaller_neuron1_training_env_"+ s_env +"_" +s_gen+".dat");
    NeuronS2.open(dir + "signaller_neuron2_training_env_"+ s_env +"_" +s_gen+".dat");
    NeuronS3.open(dir + "signaller_neuron3_training_env_"+ s_env +"_" +s_gen+".dat");
    
    NeuronR1.open(dir + "receiver_neuron1_training_env_"+ s_env +"_" +s_gen+".dat");
    NeuronR2.open(dir + "receiver_neuron2_training_env_"+ s_env +"_" +s_gen+".dat");
    NeuronR3.open(dir + "receiver_neuron3_training_env_"+ s_env +"_" +s_gen+".dat");
}


void NewLine(
    ofstream& SignallerBehaviorFile, ofstream& RecieverBehaviorFile,  
    ofstream& SignallerBehaviorFile2, ofstream& RecieverBehaviorFile2,
    ofstream& Fitness, ofstream& LandmarkFile,
    ofstream& NeuronS1, ofstream& NeuronS2, ofstream& NeuronS3,
    ofstream& NeuronR1, ofstream& NeuronR2, ofstream& NeuronR3
    ){
    Fitness << endl;
    LandmarkFile << endl;

    SignallerBehaviorFile2 << endl;
    NeuronS1 << endl;
    NeuronS2 << endl;
    NeuronS3 << endl;

    RecieverBehaviorFile2 << endl;
    NeuronR1 << endl;
    NeuronR2 << endl;
    NeuronR3 << endl; 
}

void CloseFiles(ofstream& SignallerBehaviorFile, ofstream& RecieverBehaviorFile,  
    ofstream& SignallerBehaviorFile2, ofstream& RecieverBehaviorFile2,
    ofstream& Fitness, ofstream& LandmarkFile,
    ofstream& NeuronS1, ofstream& NeuronS2, ofstream& NeuronS3,
    ofstream& NeuronR1, ofstream& NeuronR2, ofstream& NeuronR3){

    SignallerBehaviorFile.close();
    RecieverBehaviorFile.close();
    SignallerBehaviorFile2.close();
    RecieverBehaviorFile2.close();
    Fitness.close();
    LandmarkFile.close();
    NeuronS1.close();
    NeuronS2.close();
    NeuronS3.close();
    NeuronR1.close();
    NeuronR2.close();
    NeuronR3.close();

}

void RecordLandmarks(ofstream& LandmarkFile, TVector<double>& landmarkPositions, double food_location){
    for (int x = 1; x<= LN; x += 1){
        LandmarkFile << landmarkPositions[x] << " ";
    }
    LandmarkFile << food_location << " ";
}

void RecordTimeStep(
    CountingAgent& AgentSignaller,CountingAgent& AgentReceiver,
    ofstream& SignallerBehaviorFile, ofstream& RecieverBehaviorFile,  
    ofstream& NeuronS1, ofstream& NeuronS2, ofstream& NeuronS3,
    ofstream& NeuronR1, ofstream& NeuronR2, ofstream& NeuronR3
){
    // record the initial position
    SignallerBehaviorFile << AgentSignaller.GetPosition() << " ";
    RecieverBehaviorFile << AgentReceiver.GetPosition() << " ";

    NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
    NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
    NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";

    NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
    NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
    NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";
}

double FitnessFunction(TVector<double> &genotype, RandomState &rs, double start, double end, double step){
    // initalise agents
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    CreatePhenotypes(genotype,phenotypeSignaller,phenotypeReceiver);
    CountingAgent AgentSignaller(N, phenotypeSignaller);
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // trial variables 
    TVector<double> savedStateSignaller, savedStateReceiver;
    savedStateSignaller.SetBounds(1,N); 
    savedStateReceiver.SetBounds(1, N);
    TVector<double> landmarkPositions;
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_food_receiver;

    // set each of the landmarks to be the location of the food
    for (int env =1; env<=LN;env++){
        // ******** SETUP **********
        genLandmarks_Simple(0, 0, landmarkPositions);
        food_location = landmarkPositions[env];
        ResetAgentsStart(AgentSignaller, AgentReceiver);

        // Phase 1 (Training Phase) -> signaller wondering
        RunPhase1(AgentSignaller, AgentReceiver, food_location, landmarkPositions);
        
        // Reset
        ResetAgentsMidTrial(AgentSignaller, AgentReceiver);

        // Phase 2/3 (Training + scoring Phase)
        RunPhase2(AgentSignaller, AgentReceiver, food_location, landmarkPositions);

        // Testing
        SaveNeuralState(AgentReceiver, savedStateReceiver);
        SaveNeuralState(AgentSignaller, savedStateSignaller);

        for (double ref_var = 0.0; ref_var <= 0.0; ref_var += 1.0){
            for (double sep_var = 0.0; sep_var <= 0.0; sep_var += 1.0){
                genLandmarks_Simple(ref_var, sep_var, landmarkPositions);
                food_location = landmarkPositions[env];

                ResetAgentsMidTrial(AgentSignaller, AgentReceiver);
                RestoreNeuralState(AgentReceiver, savedStateReceiver);
                RestoreNeuralState(AgentSignaller, savedStateSignaller);

                double scoringTime = 0.0;
                double totalScore = 0.0;
                
                for (double time=0; time < RunDuration*2; time += StepSize){
                    Phase2(AgentSignaller, AgentReceiver, food_location, landmarkPositions);

                    if (time > RunDuration+TransDuration){
                        distance_food_receiver = Score(AgentReceiver, food_location,time);
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


double Fitness1(TVector<double> &genotype, RandomState &rs){
    double start = 0.0;
    double end = 0.0;
    double step = 0.0;
    return FitnessFunction(genotype, rs, start, end, step);
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
    double start = -1.0;
    double end = 1.0;
    double step = 1.0;
    return FitnessFunction(genotype, rs, start, end, step);
}


double Fitness3(TVector<double> &genotype, RandomState &rs){
    double start = -2.0;
    double end = 2.0;
    double step = 2.0;
    return FitnessFunction(genotype, rs, start, end, step);

}

void RecordResults(
    double call_count, vector<std::vector<TrialData>>& all_trials, vector<std::vector<double>>& phase1_signaller, 
    vector<std::vector<double>>& phase1_reciever, vector<std::vector<double>>& phase1_ns1, vector<std::vector<double>>& phase1_ns2,
    vector<std::vector<double>>& phase1_ns3, vector<std::vector<double>>& phase1_nr1, vector<std::vector<double>>& phase1_nr2, 
    vector<std::vector<double>>& phase1_nr3, vector<double>& phase1_food, vector<std::vector<double>>& phase1_landmark

){
    std::string run_id = std::to_string(call_count);

    ofstream TotalFitnessFile;
    TotalFitnessFile.open(dir + "total_fitness_" + run_id + ".dat");

    double running_total = 0.0;
    int running_trials = 0;

    for (int env = 1; env <= LN; env++){
        std::string s_env = std::to_string(env);

        ofstream SignallerBehaviorFile, RecieverBehaviorFile;
        ofstream SignallerBehaviorFile2, RecieverBehaviorFile2;
        ofstream FitnessFile;
        ofstream LandmarkFile, LandmarkFile2;
        ofstream NeuronS1, NeuronS2, NeuronS3;
        ofstream NeuronR1, NeuronR2, NeuronR3;
        OpenFiles(env, call_count, SignallerBehaviorFile, RecieverBehaviorFile, SignallerBehaviorFile2, RecieverBehaviorFile2,
        FitnessFile, LandmarkFile, NeuronS1, NeuronS2, NeuronS3, NeuronR1, NeuronR2, NeuronR3);

        ofstream NeuronS12, NeuronS22, NeuronS32, NeuronR12, NeuronR22, NeuronR32;
        NeuronS12.open(dir + "signaller_neuron1_testing_env_" + s_env + "_" + run_id + ".dat");
        NeuronS22.open(dir + "signaller_neuron2_testing_env_" + s_env + "_" + run_id + ".dat");
        NeuronS32.open(dir + "signaller_neuron3_testing_env_" + s_env + "_" + run_id + ".dat");
        NeuronR12.open(dir + "receiver_neuron1_testing_env_" + s_env + "_" + run_id + ".dat");
        NeuronR22.open(dir + "receiver_neuron2_testing_env_" + s_env + "_" + run_id + ".dat");
        NeuronR32.open(dir + "receiver_neuron3_testing_env_" + s_env + "_" + run_id + ".dat");
        LandmarkFile2.open(dir + "landmark_location_testing_env_" + s_env + "_" + run_id + ".dat");

        // write phase 1
        for (double p : phase1_signaller[env]) SignallerBehaviorFile << p << " ";
        for (double p : phase1_reciever[env])  RecieverBehaviorFile  << p << " ";
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
            for (double p : td.receiver_pos)  RecieverBehaviorFile2  << p << " ";
            SignallerBehaviorFile2 << endl;
            RecieverBehaviorFile2  << endl;

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

        SignallerBehaviorFile.close();  RecieverBehaviorFile.close();
        SignallerBehaviorFile2.close(); RecieverBehaviorFile2.close();
        FitnessFile.close();      LandmarkFile.close();
        NeuronS1.close(); NeuronS2.close(); NeuronS3.close();
        NeuronR1.close(); NeuronR2.close(); NeuronR3.close();
        NeuronS12.close(); NeuronS22.close(); NeuronS32.close();
        NeuronR12.close(); NeuronR22.close(); NeuronR32.close();
    }
    TotalFitnessFile.close();
}

double Fitness3_withRecord(TVector<double> &genotype, RandomState &rs){
    call_count ++;

    // initalise their genotypes 
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    CreatePhenotypes(genotype,phenotypeSignaller,phenotypeReceiver);
    CountingAgent AgentSignaller(N, phenotypeSignaller);
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

    // For debug
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
        food_location = landmarkPositions[env];
        ResetAgentsStart(AgentSignaller, AgentReceiver);

        phase1_food[env] = food_location;
        phase1_landmark[env].clear();
        for (int l = 1; l <= LN; l++){
            phase1_landmark[env].push_back(landmarkPositions[l]);
        }

        // Phase 1 (Training Phase) -> signaller wondering

        for (double time=0; time < RunDuration; time += StepSize){
            // only let the Signaller move
            Phase1(AgentSignaller, AgentReceiver, food_location, landmarkPositions);

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
        ResetAgentsMidTrial(AgentSignaller, AgentReceiver);

        // Phase 2/3 (Training + scoring Phase)
        for (double time=0; time < RunDuration*2; time += StepSize){
            // Receiver and Signaller only see each other 
            Phase2(AgentSignaller, AgentReceiver, food_location, landmarkPositions);

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

        SaveNeuralState(AgentReceiver, savedStateReceiver);
        SaveNeuralState(AgentSignaller, savedStateSignaller);

        int trial_num = 0;

        for (double ref_var = -2.0; ref_var <= 2.0; ref_var += 2.0){
            for (double sep_var = -2.0; sep_var <= 2.0; sep_var += 2.0){

                genLandmarks_Simple(ref_var, sep_var, landmarkPositions);
                food_location = landmarkPositions[env];

                ResetAgentsMidTrial(AgentSignaller, AgentReceiver);
                RestoreNeuralState(AgentReceiver, savedStateReceiver);
                RestoreNeuralState(AgentSignaller, savedStateSignaller);
                double lengthZone = landmarkPositions[LN] - landmarkPositions[1];

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
                    Phase2(AgentSignaller, AgentReceiver, food_location, landmarkPositions);

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
    
    // Write debug
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
        RecordResults(
            call_count, all_trials, phase1_signaller, phase1_receiver, phase1_ns1, phase1_ns2, phase1_ns3, phase1_nr1,
            phase1_nr2, phase1_nr3, phase1_food, phase1_landmark
        );
    }

    return final_fitness;
}

double TestingStage(TVector<double> &genotype, RandomState &rs){
    call_count ++;

    // initalise their genotypes 
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    CreatePhenotypes(genotype,phenotypeSignaller,phenotypeReceiver);
    CountingAgent AgentSignaller(N, phenotypeSignaller);
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

    // For debug
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
        genLandmarks_LeapFrog(rs, landmarkPositions);
        food_location = landmarkPositions[env];
        ResetAgentsStart(AgentSignaller, AgentReceiver);

        phase1_food[env] = food_location;
        phase1_landmark[env].clear();
        for (int l = 1; l <= LN; l++){
            phase1_landmark[env].push_back(landmarkPositions[l]);
        }

        // Phase 1 (Training Phase) -> signaller wondering

        for (double time=0; time < RunDuration; time += StepSize){
            // only let the Signaller move
            Phase1(AgentSignaller, AgentReceiver, food_location, landmarkPositions);

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
        ResetAgentsMidTrial(AgentSignaller, AgentReceiver);

        // Phase 2/3 (Training + scoring Phase)
        for (double time=0; time < RunDuration*2; time += StepSize){
            // Receiver and Signaller only see each other 
            Phase2(AgentSignaller, AgentReceiver, food_location, landmarkPositions);

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

        SaveNeuralState(AgentReceiver, savedStateReceiver);
        SaveNeuralState(AgentSignaller, savedStateSignaller);

        int trial_num = 0;

        for (double ref_var = -2.0; ref_var <= 2.0; ref_var += 2.0){
            for (double sep_var = -2.0; sep_var <= 2.0; sep_var += 2.0){

                genLandmarks_Simple(ref_var, sep_var, landmarkPositions);
                food_location = landmarkPositions[env];

                ResetAgentsMidTrial(AgentSignaller, AgentReceiver);
                RestoreNeuralState(AgentReceiver, savedStateReceiver);
                RestoreNeuralState(AgentSignaller, savedStateSignaller);
                double lengthZone = landmarkPositions[LN] - landmarkPositions[1];

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
                    Phase2(AgentSignaller, AgentReceiver, food_location, landmarkPositions);

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
    
    // Write debug
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
        RecordResults(
            call_count, all_trials, phase1_signaller, phase1_receiver, phase1_ns1, phase1_ns2, phase1_ns3, phase1_nr1,
            phase1_nr2, phase1_nr3, phase1_food, phase1_landmark
        );
    }

    return final_fitness;
}

void RecordBehavior(TSearch &s, RandomState &rs){
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();

    TVector<double> genotype;
    genotype = s.BestIndividual();

    // Map genotype to phenotype
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    CreatePhenotypes(genotype,phenotypeSignaller,phenotypeReceiver);
    CountingAgent AgentSignaller(N, phenotypeSignaller);
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // save this state
    // Save state
    TVector<double> savedStateSignaller, savedStateReceiver;
    savedStateSignaller.SetBounds(1,N); 
    savedStateReceiver.SetBounds(1, N);
    TVector<double> landmarkPositions;
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_food_receiver;

    ofstream TotalFitness;
    TotalFitness.open( dir + "total_fitness_" + current_run +".dat");

    for (int env =1; env<=LN;env++){
        std::string s_env = std::to_string(env);
        ofstream SignallerBehaviorFile, RecieverBehaviorFile;
        ofstream SignallerBehaviorFile2, RecieverBehaviorFile2;
        ofstream Fitness;
        ofstream LandmarkFile;
        ofstream NeuronS1, NeuronS2, NeuronS3;
        ofstream NeuronR1, NeuronR2, NeuronR3;
        OpenFiles(env, 0, SignallerBehaviorFile, RecieverBehaviorFile, SignallerBehaviorFile2, RecieverBehaviorFile2,
        Fitness, LandmarkFile, NeuronS1, NeuronS2, NeuronS3, NeuronR1, NeuronR2, NeuronR3);

        genLandmarks_Simple(0, 0, landmarkPositions);
        food_location = landmarkPositions[env];
        RecordLandmarks(LandmarkFile, landmarkPositions, food_location);
        ResetAgentsStart(AgentSignaller, AgentReceiver);
        
        // ********* PHASE 1 ************
        RecordTimeStep(AgentSignaller, AgentReceiver, SignallerBehaviorFile, RecieverBehaviorFile, NeuronS1, NeuronS2, NeuronS3, NeuronR1, NeuronR2, NeuronR3);
        for (double time=0; time < RunDuration; time += StepSize){
            Phase1(AgentSignaller, AgentReceiver, food_location, landmarkPositions);
            RecordTimeStep(AgentSignaller, AgentReceiver, SignallerBehaviorFile, RecieverBehaviorFile, NeuronS1, NeuronS2, NeuronS3, NeuronR1, NeuronR2, NeuronR3);
        }

        // Reset
        ResetAgentsMidTrial(AgentSignaller, AgentReceiver);

        // ********* PHASE 2/3 ************
        // record the initial position
        RecordTimeStep(AgentSignaller, AgentReceiver, SignallerBehaviorFile, RecieverBehaviorFile, NeuronS1, NeuronS2, NeuronS3, NeuronR1, NeuronR2, NeuronR3);
        for (double time=0; time < RunDuration*2; time += StepSize){
            Phase2(AgentSignaller, AgentReceiver, food_location, landmarkPositions);
            RecordTimeStep(AgentSignaller, AgentReceiver, SignallerBehaviorFile, RecieverBehaviorFile, NeuronS1, NeuronS2, NeuronS3, NeuronR1, NeuronR2, NeuronR3);
        }
        SaveNeuralState(AgentReceiver, savedStateReceiver);
        SaveNeuralState(AgentSignaller, savedStateSignaller);

        // ******* TESTING *********
        for (double ref_var = -2.0; ref_var <= 2.0; ref_var += 2.0){
            for (double sep_var = -2.0; sep_var <= 2.0; sep_var += 2.0){
                genLandmarks_Simple(ref_var, sep_var, landmarkPositions);
                food_location = landmarkPositions[env];
                RecordLandmarks(LandmarkFile, landmarkPositions, food_location);

                ResetAgentsMidTrial(AgentSignaller, AgentReceiver);
                RestoreNeuralState(AgentReceiver, savedStateReceiver);
                RestoreNeuralState(AgentSignaller, savedStateSignaller);

                // ********* PHASE 2/3 ************
                // record the initial position
                RecordTimeStep(AgentSignaller, AgentReceiver, SignallerBehaviorFile2, RecieverBehaviorFile2, NeuronS1, NeuronS2, NeuronS3, NeuronR1, NeuronR2, NeuronR3);

                double scoringTime = 0.0;
                double totalScore = 0.0;

                for (double time=0; time < RunDuration*2; time += StepSize){
                    
                    Phase2(AgentSignaller, AgentReceiver, food_location, landmarkPositions);

                    if (time > RunDuration+TransDuration){
                        distance_food_receiver = Score(AgentReceiver, food_location,time);
                        totalScore += distance_food_receiver;
                        scoringTime += 1;
                        
                        // length of the landmark zone
                        double lengthZone = landmarkPositions[LN] - landmarkPositions[1];
                        if (scoringTime > 0){
                            Fitness << 1 - ((totalScore/scoringTime)/lengthZone) << " ";
                        }
                    }
                    RecordTimeStep(AgentSignaller, AgentReceiver, SignallerBehaviorFile2, RecieverBehaviorFile2, NeuronS1, NeuronS2, NeuronS3, NeuronR1, NeuronR2, NeuronR3);
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
                NewLine(SignallerBehaviorFile, RecieverBehaviorFile, SignallerBehaviorFile2, RecieverBehaviorFile2,
                    Fitness, LandmarkFile, NeuronS1, NeuronS2, NeuronS3, NeuronR1, NeuronR2, NeuronR3);

                TotalFitness << endl;  
            }
        }
        // close the files for that env
        CloseFiles(SignallerBehaviorFile, RecieverBehaviorFile, SignallerBehaviorFile2, RecieverBehaviorFile2,
                    Fitness, LandmarkFile, NeuronS1, NeuronS2, NeuronS3, NeuronR1, NeuronR2, NeuronR3);

    }
    TotalFitness.close();
    // return (total_trials > 0) ? total_fitness / total_trials : 0.0;
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

    record = 0;

    // Stage 0: Evolve the Signaller
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(signaller_to_food);
    search.ExecuteSearch();

    // Stage 1: A Slightly Easier task (simpler landmark generation)
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(Fitness1);
    search.ExecuteSearch();

    // Stage 2: Full Task - Medium
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(Fitness2);
    search.ExecuteSearch();

    // Stage 3: Full Task - Hard
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(Fitness3);
    search.ExecuteSearch();

    // Stage 3: Full Task - with lots of crossover
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(TestingStage);
    search.ExecuteSearch();

    #ifdef PRINTTOFILE
        file.close();
    #endif

    return 0;
}
