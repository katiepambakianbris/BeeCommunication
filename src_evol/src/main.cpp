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
// 4 N as we only have 2 sensors for each neuron now
int VectSize = 2 * (N*N + 4*N);  // Double the amount of parameters, one for Receiver, one for Signaler
double REF = 15;
double SEP = 15;


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
const double MinLength = 50;
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
    // // Food Sensor Weights
    // for (int i = 1; i <= N; i++) {
    //     phen(x) = MapSearchParameter(gen(k), -SR, SR);
    //     k++;
    //     x++;
    // }
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

// ---------------------------------------------------------
// stage 1: Evolution of receiver
// only looking at the distance between the receiver and the 
// flower/food
// no signaller in the task
// the sensor of the receiver is set to correspond to which
// of the landmarks the receiver should go to
// ---------------------------------------------------------

double stage1(TVector<double> &genotype, RandomState &rs){
    // initalise their genotypes 
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);

    CountingAgent AgentSignaller(N, phenotypeSignaller);
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 4*N +1));
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // save the state of the signaller and receiver
    TVector<double> savedstateSignaller, savedstateReceiver;
    savedstateSignaller.SetBounds(1,N); 
    savedstateReceiver.SetBounds(1, N);

    // Landmark Positions
    TVector<double> landmarkPositions;

    // trial variables 
    double food_location;
    double total_trials = 0;
    double total_fitness =0;
    double distance_food_receiver;

    double input[N] = {0.5,1.0,1.5};


    for (int i=0; i<3; i++){
        genLandmarks_LeapFrog(rs, landmarkPositions);
        // set each of the landmarks to be the location of the food
        for (int env =1; env<=LN;env++){

            // ******** SETUP **********

            // set the food to be at the ith landmark location
            food_location = landmarkPositions[env];

            // Initialise the agent for this trial
            AgentReceiver.SetPosition(0);
            AgentReceiver.ResetNeuralState();
            AgentReceiver.ResetSensors();

            // Initialise the agent for this trial
            double location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
            AgentSignaller.SetPosition(location);
            AgentSignaller.ResetNeuralState();
            AgentSignaller.ResetSensors();

            // Phase 1 (Training Phase) -> signaller wondering
            for (double time=0; time < RunDuration; time += StepSize){
                // only let the Signaller move
                AgentSignaller.SenseTarget(input[env-1]);
                AgentSignaller.SenseOther(AgentReceiver.Position());
                AgentSignaller.Step(StepSize);
            }
            
            // Reset
            AgentReceiver.SetPosition(0);
            AgentSignaller.SetPosition(location);
            AgentReceiver.ResetSensors();
            AgentSignaller.ResetSensors();

            // Phase 2/3 (Training + scoring Phase)

            double scoringTime = 0.0;
            double totalScore = 0.0;
            
            for (double time=0; time < RunDuration*2; time += StepSize){
                // Receiver and Signaller only see each other 
                AgentReceiver.SenseOther(AgentSignaller.Position());
                AgentSignaller.SenseOther(AgentReceiver.Position());

                AgentReceiver.SenseLandmarks(LN, landmarkPositions);

                // AgentSignaller.SenseTarget(input[env-1]);

                // Move both the agents
                AgentReceiver.Step(StepSize);
                AgentSignaller.Step(StepSize);

                // clam the signallers position
                double pos = AgentSignaller.Position();
                
                if (pos > LANDMARKZONESTART){
                    AgentSignaller.SetPosition(LANDMARKZONESTART);
                }

                if (time > TransDuration*2){
                    distance_food_receiver = fabs(AgentReceiver.Position() - food_location);

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

double RecordBehavior(TSearch &s, RandomState &rs){
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();

    // **** create the files to save the behaviour/env ***
    // storing them all in the same file, each gets a new line (so 9 trials per file)
    // the stages are separated into three files
    ofstream SignallerBehaviorFile, RecieverBehaviorFile;
    SignallerBehaviorFile.open( dir + "behavior_Signaller_stage1_" + current_run + ".dat");
    RecieverBehaviorFile.open( dir + "behavior_Reciever_stage1_" + current_run + ".dat");
    ofstream Stage3Fitness;
    Stage3Fitness.open( dir + "fitness_" + current_run +".dat");
    
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
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 4*N +1));
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

    double input[3] = {0.5,1.0,1.5};

    // calculating what the set other should be (value between 0.5 and 1.5)
    for (int i=0; i<3; i++){
        genLandmarks_LeapFrog(rs, landmarkPositions);
        // set each of the landmarks to be the location of the food
        for (int env =1; env<=LN;env++){
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
            double location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
            AgentSignaller.SetPosition(location);
            AgentSignaller.ResetSensors();

            // Reset neural state
            for (int x = 1; x <= N; x++)
            {
                AgentSignaller.NervousSystem.SetNeuronState(x, savedStateSignaller[x]);
            
            }
            for (int x = 1; x <= N; x++)
            {
                AgentReceiver.NervousSystem.SetNeuronState(x, savedStateReceiver[x]);
            
            }
            // ********* PHASE 1 ************

            // record the initial position
            SignallerBehaviorFile << AgentSignaller.Position() << " ";
            RecieverBehaviorFile << AgentReceiver.Position() << " ";

            NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
            NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
            NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";

            NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
            NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
            NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";

            for (double time=0; time < RunDuration; time += StepSize){
                // only let the Signaller move
                AgentSignaller.SenseTarget(input[env-1]);
                AgentSignaller.SenseOther(AgentReceiver.Position());
                AgentSignaller.Step(StepSize);

                SignallerBehaviorFile << AgentSignaller.Position() << " ";
                RecieverBehaviorFile << AgentReceiver.Position() << " ";
            
                NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
                NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
                NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";

                NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
                NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
                NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";
            }

            // Reset
            AgentReceiver.SetPosition(0);
            AgentSignaller.SetPosition(location);
            AgentReceiver.ResetSensors();
            AgentSignaller.ResetSensors();

            // ********* PHASE 2/3 ************

            // record the initial position
            SignallerBehaviorFile << AgentSignaller.Position() << " ";
            RecieverBehaviorFile << AgentReceiver.Position() << " ";
            NeuronS1 << AgentSignaller.NervousSystem.NeuronState(1) << " ";
            NeuronS2 << AgentSignaller.NervousSystem.NeuronState(2) << " ";
            NeuronS3 << AgentSignaller.NervousSystem.NeuronState(3) << " ";

            NeuronR1 << AgentReceiver.NervousSystem.NeuronState(1) << " ";
            NeuronR2 << AgentReceiver.NervousSystem.NeuronState(2) << " ";
            NeuronR3 << AgentReceiver.NervousSystem.NeuronState(3) << " ";

            double scoringTime = 0.0;
            double totalScore = 0.0;

            for (double time=0; time < RunDuration*2; time += StepSize){
                
                AgentReceiver.SenseOther(AgentSignaller.Position());
                AgentSignaller.SenseOther(AgentReceiver.Position());

                AgentReceiver.SenseLandmarks(LN, landmarkPositions);

                // AgentSignaller.SenseTarget(input[env-1]);

                // Move both the agents
                AgentReceiver.Step(StepSize);
                AgentSignaller.Step(StepSize);

                // clam the signallers position
                double pos = AgentSignaller.Position();
                
                if (pos > LANDMARKZONESTART){
                    AgentSignaller.SetPosition(LANDMARKZONESTART);
                }
                

                if (time > TransDuration*2){
                    distance_food_receiver = fabs(AgentReceiver.Position() - food_location);

                    // if the distance is within a threshold set the score to be perfect
                    if (distance_food_receiver < 1 && time > HarshDuration){
                        distance_food_receiver = 0;
                    } else if (distance_food_receiver < mindist){
                        distance_food_receiver = 0;
                    }
                    totalScore += distance_food_receiver;
                    scoringTime += 1;

                    if (scoringTime > 0){
                        Stage3Fitness << 1 - ((totalScore/scoringTime)/ArenaLength) << " ";
                    }
                }

                SignallerBehaviorFile << AgentSignaller.Position() << " ";
                RecieverBehaviorFile << AgentReceiver.Position() << " ";
                
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
            double fitness = 1 - ((totalScore/scoringTime)/ArenaLength);
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
    GenPhenMapping(bestVector, phenotypeR, (int)(N*N + 4*N + 1));

    // Show the Signaler
    BestIndividualFile.open( dir + "best_ns_s_" + current_run + ".dat" );
    CountingAgent AgentSignaler( N, phenotypeS);

    // Send to file
    BestIndividualFile << "Nervous System:" << endl;
    BestIndividualFile << AgentSignaler.NervousSystem << endl;
    // BestIndividualFile << "Food Sensor Weights:" << endl;
    // BestIndividualFile << AgentSignaler.foodsensorweights << "\n" << endl;
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
    // BestIndividualFile << "Food Sensor Weights:" << endl;
    // BestIndividualFile << AgentReceiver.foodsensorweights << "\n" << endl;
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
    std::string result_dir = "/user/work/yj23812/BeeCommunication/results/"+ date_as_string() +"/" + slurm_job_id;
    // std::string result_dir = "/Users/katiepambakian/Documents/BSc Computer Science/Y3/Dissertation/BeeCommunication/results/"+ date_as_string() +"/v";

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

    if (search.BestPerformance() > 0.99) {
        RecordBehavior(search, search.getRandomState());
    }

    #ifdef PRINTTOFILE
        file.close();
    #endif

    return 0;
}
