#include <iostream>
#include "TSearch.h"
#include "CountingAgent.h"
#include "CTRNN.h"
#include "random.h"
#include <iomanip> 
#include <sstream>
#include <filesystem>
#include <fstream>

#define PRINTTOFILE

// Task params
const int LN = 3;                   // Number of landmarks in the environment
const double StepSize = 0.1;
const double RunDuration = 300.0;
const double TransDuration = 150.0;
const double MinLength = 50.0;      
const double mindist = 5.0;         

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
int VectSize = (N*N + 5*N);

// landmark parameters 
double REF = 15;
double SEP = 15;

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


// ------------------------------------
// Stages of each of the tasks
// -----------------------------------

// ************************
// No change
// ************************
double FitnessFunction1(TVector<double> &genotype, RandomState &rs)
{
    // Map genotype to phenotype
    TVector<double> phenotypeS;
    phenotypeS.SetBounds(1, (int)(VectSize));
    GenPhenMapping(genotype, phenotypeS, 1);

    // construct the signaller agent
    CountingAgent Agent(N, phenotypeS);

    // Save state
    // Use this to save the neural state during learning
    TVector<double> saved_state;
    saved_state.SetBounds(1,N);

    // Keep track of performance (fitness across landmarks/env)
    double totaltrials = 0;
    double totaltime;
    double dist;
    double totaldist;
    double totalfit = 0.0;
    double food_loc, food_loc_mod;
    double fit;

    // ************************
    // Set up Landmarks
    // ************************

    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1)
    {
        landmarkPositions[i] = REF + (i * SEP);
    }

    TVector<double> landmarkPositionTest;
    landmarkPositionTest.SetBounds(1,LN);  

    // loop through different enviornment 
    // for each enviornment i, landmark i has the food 
    for (int env = 1; env <= LN; env += 1)
    {
        // Step 1: Setup
        // Establish food location
        food_loc = landmarkPositions[env];
    
        // reset position + neural state of the signaller
        Agent.ResetPosition(0); 
        Agent.ResetNeuralState();

        // Step 2: Training Phase
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            Agent.SenseFood(food_loc);
            Agent.SenseLandmarks(LN,landmarkPositions);
            Agent.Step(StepSize);
        }

        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            saved_state[i] = Agent.NervousSystem.NeuronState(i);
        }
    
        // 2. Testing Phase
        for (double ref_var = 0.0; ref_var <= 0.0; ref_var += 1.0) // OFFSET = 15
        {
            for (double sep_var = 0.0; sep_var <= 0.0; sep_var += 1.0) // SEP = 15
            {
                // arrange the landmarks in position  
                for (int i = 1; i <= LN; i += 1)
                {
                    landmarkPositionTest[i] = (REF + ref_var) + (i * (SEP + sep_var));
                }
                food_loc_mod = landmarkPositionTest[env];

                // Setup
                Agent.ResetPosition(0);
                Agent.ResetSensors();
                for (int i = 1; i <= N; i++)
                {
                    Agent.NervousSystem.SetNeuronState(i, saved_state[i]);
                }
                totaldist = 0.0;
                totaltime = 0.0;
        
                // run the trial
                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    Agent.SenseLandmarks(LN,landmarkPositionTest);
                    Agent.Step(StepSize);

                    // Measure distance between them (after transients)
                    if (time > TransDuration)
                    {
                        dist = fabs(Agent.pos - food_loc_mod);
                        
                        if (dist < mindist){
                            dist = 0.0;
                        }
                        totaldist += dist;

                        totaltime += 1;
                    }
                }

                fit = 1 - ((totaldist / totaltime)/MinLength);
                if (fit < 0.0){
                    fit = 0.0;
                }
                totalfit += fit;
                totaltrials += 1;
            }
        }
    }
    return totalfit / totaltrials;
}

// ************************
// Small change (-1 -> 1)
// ************************
double FitnessFunction2(TVector<double> &genotype, RandomState &rs)
{
    // Map genotype to phenotype
    TVector<double> phenotypeS;
    phenotypeS.SetBounds(1, (int)(VectSize));
    GenPhenMapping(genotype, phenotypeS, 1);

    // construct the signaller agent
    CountingAgent Agent(N, phenotypeS);

    // Save state
    TVector<double> saved_state;
    saved_state.SetBounds(1,N);

    // Keep track of performance (fitness across landmarks/env)
    double totaltrials = 0;
    double totaltime;
    double dist;
    double totaldist;
    double totalfit = 0.0;
    double food_loc, food_loc_mod;
    double fit;

    // ************************
    // Set up Landmarks
    // ************************

    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1)
    {
        landmarkPositions[i] = REF + (i * SEP);
    }

    TVector<double> landmarkPositionTest;
    landmarkPositionTest.SetBounds(1,LN);  
    
    // Use this to save the neural state during learning

    // loop through different enviornment 
    // for each enviornment i, landmark i has the food 
    for (int env = 1; env <= LN; env += 1)
    {
        // Step 1: Setup

        // Establish food location
        food_loc = landmarkPositions[env];
    
        // reset position + neural state of the signaller
        Agent.ResetPosition(0); 
        Agent.ResetNeuralState();

        // Step 2: Training Phase
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            Agent.SenseFood(food_loc);
            Agent.SenseLandmarks(LN,landmarkPositions);
            Agent.Step(StepSize);
        }

        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            saved_state[i] = Agent.NervousSystem.NeuronState(i);
        }
    
        // 2. Testing Phase
        for (double ref_var = -1.0; ref_var <= 1.0; ref_var += 1.0)
        {
            for (double sep_var = -1.0; sep_var <= 1.0; sep_var += 1.0)
            {
                // arrange the landmarks in position  
                for (int i = 1; i <= LN; i += 1)
                {
                    landmarkPositionTest[i] = (REF + ref_var) + (i * (SEP + sep_var));
                }
                food_loc_mod = landmarkPositionTest[env];

                // Setup
                Agent.ResetPosition(0);
                Agent.ResetSensors();
                for (int i = 1; i <= N; i++)
                {
                    Agent.NervousSystem.SetNeuronState(i, saved_state[i]);
                }
                totaldist = 0.0;
                totaltime = 0.0;
        
                // run the trial
                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    Agent.SenseLandmarks(LN,landmarkPositionTest);
                    Agent.Step(StepSize);

                    // Measure distance between them (after transients)
                    if (time > TransDuration)
                    {
                        dist = fabs(Agent.pos - food_loc_mod);
                        
                        if (dist < mindist){
                            dist = 0.0;
                        }
                        totaldist += dist;

                        totaltime += 1;
                    }
                }

                fit = 1 - ((totaldist / totaltime)/MinLength);
                if (fit < 0.0){
                    fit = 0.0;
                }
                totalfit += fit;

                totaltrials += 1;
            }
        }
    }
    return totalfit / totaltrials;
}

// ************************
// Larger change (-2 -> 2)
// ************************
double FitnessFunction3(TVector<double> &genotype, RandomState &rs)
{
    // Map genotype to phenotype
    TVector<double> phenotypeS;
    phenotypeS.SetBounds(1, (int)(VectSize));
    GenPhenMapping(genotype, phenotypeS, 1);

    // construct the signaller agent
    CountingAgent Agent(N, phenotypeS);

    // Save state
    TVector<double> saved_state;
    saved_state.SetBounds(1,N);

    // Keep track of performance (fitness across landmarks/env)
    double totaltrials = 0;
    double totaltime;
    double dist;
    double totaldist;
    double totalfit = 0.0;
    double food_loc, food_loc_mod;
    double fit;

    // ************************
    // Set up Landmarks
    // ************************

    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1)
    {
        landmarkPositions[i] = REF + (i * SEP);
    }

    TVector<double> landmarkPositionTest;
    landmarkPositionTest.SetBounds(1,LN);  
    
    // Use this to save the neural state during learning

    // loop through different enviornment 
    // for each enviornment i, landmark i has the food 
    for (int env = 1; env <= LN; env += 1)
    {
        // Step 1: Setup

        // Establish food location
        food_loc = landmarkPositions[env];
    
        // reset position + neural state of the signaller
        Agent.ResetPosition(0); 
        Agent.ResetNeuralState();

        // Step 2: Training Phase
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            Agent.SenseFood(food_loc);
            Agent.SenseLandmarks(LN,landmarkPositions);
            Agent.Step(StepSize);
        }

        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            saved_state[i] = Agent.NervousSystem.NeuronState(i);
        }
    
        // 2. Testing Phase
        for (double ref_var = -2.0; ref_var <= 2.0; ref_var += 2.0)
        {
            for (double sep_var = -2.0; sep_var <= 2.0; sep_var += 2.0)
            {
                // arrange the landmarks in position  
                for (int i = 1; i <= LN; i += 1)
                {
                    landmarkPositionTest[i] = (REF + ref_var) + (i * (SEP + sep_var));
                }
                food_loc_mod = landmarkPositionTest[env];

                // Setup
                Agent.ResetPosition(0);
                Agent.ResetSensors();
                for (int i = 1; i <= N; i++)
                {
                    Agent.NervousSystem.SetNeuronState(i, saved_state[i]);
                }
                totaldist = 0.0;
                totaltime = 0.0;
        
                // run the trial
                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    Agent.SenseLandmarks(LN,landmarkPositionTest);
                    Agent.Step(StepSize);

                    // Measure distance between them (after transients)
                    if (time > TransDuration)
                    {
                        dist = fabs(Agent.pos - food_loc_mod);
                        
                        if (dist < mindist){
                            dist = 0.0;
                        }
                        totaldist += dist;

                        totaltime += 1;
                    }
                }

                fit = 1 - ((totaldist / totaltime)/MinLength);
                if (fit < 0.0){
                    fit = 0.0;
                }
                totalfit += fit;

                totaltrials += 1;
            }
        }
    }
    return totalfit / totaltrials;
}

// *****************************
// Time Delay
// between testing and training
// *****************************
double FitnessFunction4(TVector<double> &genotype, RandomState &rs)
{
    // Map genotype to phenotype
    TVector<double> phenotypeS;
    phenotypeS.SetBounds(1, (int)(VectSize));
    GenPhenMapping(genotype, phenotypeS, 1);

    // construct the signaller agent
    CountingAgent Agent(N, phenotypeS);

    // Save state
    TVector<double> saved_state;
    saved_state.SetBounds(1,N);

    // Keep track of performance (fitness across landmarks/env)
    double totaltrials = 0;
    double totaltime;
    double dist;
    double totaldist;
    double totalfit = 0.0;
    double food_loc, food_loc_mod;
    double fit;

    // ************************
    // Set up Landmarks
    // ************************

    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1)
    {
        landmarkPositions[i] = REF + (i * SEP);
    }

    TVector<double> landmarkPositionTest;
    landmarkPositionTest.SetBounds(1,LN);  


    // loop through different enviornment 
    // for each enviornment i, landmark i has the food 
    for (int env = 1; env <= LN; env += 1)
    {
        for (int delay=0; delay<=10; delay +=5){
            // Step 1: Setup

            // Establish food location
            food_loc = landmarkPositions[env];
        
            // reset position + neural state of the signaller
            Agent.ResetPosition(0);
            Agent.ResetSensors();
            Agent.ResetNeuralState();

            // Step 2: Training Phase
            for (double time = 0; time < RunDuration; time += StepSize)
            {
                Agent.SenseFood(food_loc);
                Agent.SenseLandmarks(LN,landmarkPositions);
                Agent.Step(StepSize);
            }

            // 2. delay
            Agent.ResetPosition(0); 
            Agent.ResetSensors();

            for (double time=0; time < delay; time += StepSize){
                // not sure if it should step or not
                Agent.Step(StepSize);
            }

            // Saved each of their neural states 
            for (int i = 1; i <= N; i++)
            {
                saved_state[i] = Agent.NervousSystem.NeuronState(i);
            }
            
            // 3. Testing Phase
            for (double ref_var = -2.0; ref_var <= 2.0; ref_var += 2.0)
            {
                for (double sep_var = -2.0; sep_var <= 2.0; sep_var += 2.0)
                {
                    // arrange the landmarks in position  
                    for (int i = 1; i <= LN; i += 1)
                    {
                        landmarkPositionTest[i] = (REF + ref_var) + (i * (SEP + sep_var));
                    }
                    food_loc_mod = landmarkPositionTest[env];

                    // Setup
                    Agent.ResetPosition(0);
                    Agent.ResetSensors();
                    for (int i = 1; i <= N; i++)
                    {
                        Agent.NervousSystem.SetNeuronState(i, saved_state[i]);
                    }
                    totaldist = 0.0;
                    totaltime = 0.0;
            
                    // run the trial
                    for (double time = 0; time < RunDuration; time += StepSize)
                    {
                        Agent.SenseLandmarks(LN,landmarkPositionTest);
                        Agent.Step(StepSize);

                        // Measure distance between them (after transients)
                        if (time > TransDuration)
                        {
                            dist = fabs(Agent.pos - food_loc_mod);
                            
                            if (dist < mindist){
                                dist = 0.0;
                            }
                            totaldist += dist;

                            totaltime += 1;
                        }
                    }

                    fit = 1 - ((totaldist / totaltime)/MinLength);
                    if (fit < 0.0){
                        fit = 0.0;
                    }
                    totalfit += fit;

                    totaltrials += 1;
                }
            }
        }
    }
    return totalfit / totaltrials;
}

double RecordBehavior1(TSearch &s) {
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();
    
    // Map genotype to phenotype
    TVector<double> genotype;
    genotype = s.BestIndividual();

    TVector<double> phenotype;
    phenotype.SetBounds(1, (int)(VectSize));
    GenPhenMapping(genotype, phenotype, 1);

    CountingAgent Agent( N, phenotype);

    // Bookeeping variables
    double totaltrials = 0;
    double totaltime;
    double dist;
    double totaldist;
    double totalfit = 0.0;
    double food_loc, food_loc_mod;
    double fit;

    // Landmarks base position and variations
    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1){
        landmarkPositions[i] = REF + (i * SEP);
    }

    TVector<double> landmarkPositionTest;
    landmarkPositionTest.SetBounds(1,LN);

    // save the state
    TVector<double> savedStateStart, savedStateHalfWay;
    savedStateStart.SetBounds(1,N);
    savedStateHalfWay.SetBounds(1,N);

    // Saved each of their neural states 
    for (int i = 1; i <= N; i++)
    {
        savedStateStart[i] = Agent.NervousSystem.NeuronState(i);
    }

    // Phase
    for (int env = 1; env <= LN; env += 1){
        std::string s_env = std::to_string(env);
        ofstream BehaviorFile1;
        BehaviorFile1.open( dir + "behavior_" + current_run + "_Env" + s_env + "_stage1" + "_Phase1.dat");
        ofstream BehaviorFile2;
        BehaviorFile2.open( dir + "behavior_" + current_run + "_Env" + s_env + "_stage1" + "_Phase2.dat");

        // stores the location of the landmarks and the food
        ofstream LandmarkFile1, LandmarkFile2;
        LandmarkFile1.open(dir + "landmark_location_"+current_run+"_env"+s_env+ "_stage1" + "_phase1.dat");
        LandmarkFile2.open(dir + "landmark_location_"+current_run+"_env"+s_env+ "_stage1" +"_phase2.dat");

        ofstream FitnessFile;
        FitnessFile.open(dir + "fitness_"+current_run+"_env"+s_env+ "_stage1" + ".dat");

        // PHASE 1: Setup

        // Establish landmark and food location
        food_loc = landmarkPositions[env];
        for (int i = 1; i <= LN; i += 1){
            LandmarkFile1 << landmarkPositions[i] << " ";
        }
        LandmarkFile1 << food_loc << " ";

        //  setup the agents position + state
        Agent.ResetPosition(0);
        Agent.ResetSensors();
        for (int i = 1; i <= N; i++)
        {
            Agent.NervousSystem.SetNeuronState(i, savedStateStart[i]);
        }


        BehaviorFile1 << Agent.Position() << " ";

        // Step 2: Training Phase
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            Agent.SenseFood(food_loc);
            Agent.SenseLandmarks(LN,landmarkPositions);
            Agent.Step(StepSize);
            BehaviorFile1 << Agent.Position() << " ";
        }

        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            savedStateHalfWay[i] = Agent.NervousSystem.NeuronState(i);
        }
    
        // Testing Phase
        for (double ref_var = 0.0; ref_var <= 0.0; ref_var += 1.0) // OFFSET = 15
        {
            for (double sep_var = 0.0; sep_var <= 0.0; sep_var += 1.0) // SEP = 15
            {
                // arrange the landmarks in position  
                for (int i = 1; i <= LN; i += 1)
                {
                    landmarkPositionTest[i] = (REF + ref_var) + (i * (SEP + sep_var));
                }
                food_loc_mod = landmarkPositionTest[env];
                // write the landmarks to the output
                for (int i = 1; i <= LN; i += 1){
                    LandmarkFile2 << landmarkPositionTest[i] << " ";
                }
                LandmarkFile2 << food_loc_mod << " ";

                // setup
                Agent.ResetSensors();
                Agent.ResetPosition(0);
                for (int i = 1; i <= N; i++)
                {
                    Agent.NervousSystem.SetNeuronState(i, savedStateHalfWay[i]);
                }
                totaldist = 0.0;
                totaltime = 0.0;

                BehaviorFile2 << Agent.Position() << " ";

                // run the trial
                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    Agent.SenseLandmarks(LN,landmarkPositionTest);
                    Agent.Step(StepSize);
                    BehaviorFile2 << Agent.Position() << " ";

                    // Measure distance between them (after transients)
                    if (time > TransDuration)
                    {
                        dist = fabs(Agent.pos - food_loc_mod);
                        
                        if (dist < mindist){
                            dist = 0.0;
                        }

                        totaldist += dist;

                        totaltime += 1;
                        FitnessFile << 1 - ((totaldist / totaltime)/MinLength) << " ";
                    }
                }

                fit = 1 - ((totaldist / totaltime)/MinLength);
                if (fit < 0.0){
                    fit = 0.0;
                }
                totalfit += fit;

                totaltrials += 1;
            
                BehaviorFile2 << endl;
                LandmarkFile2 << endl;
                FitnessFile << endl;
            }
        }
        BehaviorFile1.close();
        BehaviorFile2.close();
        LandmarkFile1.close();
        LandmarkFile2.close();
        FitnessFile.close();
    }
    
    return (totalfit) / (totaltrials);
}

double RecordBehavior2(TSearch &s) {
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();
    
    // Map genotype to phenotype
    TVector<double> genotype;
    genotype = s.BestIndividual();

    TVector<double> phenotype;
    phenotype.SetBounds(1, (int)(VectSize));
    GenPhenMapping(genotype, phenotype, 1);

    CountingAgent Agent( N, phenotype);

    // Bookeeping variables
    double totaltrials = 0;
    double totaltime;
    double dist;
    double totaldist;
    double totalfit = 0.0;
    double food_loc, food_loc_mod;
    double fit;

    // Landmarks base position and variations
    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1){
        landmarkPositions[i] = REF + (i * SEP);
    }

    TVector<double> landmarkPositionTest;
    landmarkPositionTest.SetBounds(1,LN);

    // save the state
    TVector<double> savedStateStart, savedStateHalfWay;
    savedStateStart.SetBounds(1,N);
    savedStateHalfWay.SetBounds(1,N);

    // Saved each of their neural states 
    for (int i = 1; i <= N; i++)
    {
        savedStateStart[i] = Agent.NervousSystem.NeuronState(i);
    }

    // Phase
    for (int env = 1; env <= LN; env += 1){
        std::string s_env = std::to_string(env);
        ofstream BehaviorFile1;
        BehaviorFile1.open( dir + "behavior_" + current_run + "_Env" + s_env + "_stage2" + "_Phase1.dat");
        ofstream BehaviorFile2;
        BehaviorFile2.open( dir + "behavior_" + current_run + "_Env" + s_env + "_stage2" + "_Phase2.dat");

        // stores the location of the landmarks and the food
        ofstream LandmarkFile1, LandmarkFile2;
        LandmarkFile1.open(dir + "landmark_location_"+current_run+"_env"+s_env+ "_stage2" + "_phase1.dat");
        LandmarkFile2.open(dir + "landmark_location_"+current_run+"_env"+s_env+ "_stage2" +"_phase2.dat");

        ofstream FitnessFile;
        FitnessFile.open(dir + "fitness_"+current_run+"_env"+s_env+ "_stage2" + ".dat");

        // PHASE 1: Setup

        // Establish landmark and food location
        food_loc = landmarkPositions[env];
        for (int i = 1; i <= LN; i += 1){
            LandmarkFile1 << landmarkPositions[i] << " ";
        }
        LandmarkFile1 << food_loc << " ";

        //  setup the agents position + state
        Agent.ResetPosition(0);
        Agent.ResetSensors();
        for (int i = 1; i <= N; i++)
        {
            Agent.NervousSystem.SetNeuronState(i, savedStateStart[i]);
        }


        BehaviorFile1 << Agent.Position() << " ";

        // Step 2: Training Phase
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            Agent.SenseFood(food_loc);
            Agent.SenseLandmarks(LN,landmarkPositions);
            Agent.Step(StepSize);
            BehaviorFile1 << Agent.Position() << " ";
        }

        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            savedStateHalfWay[i] = Agent.NervousSystem.NeuronState(i);
        }
    
        // Testing Phase
        for (double ref_var = -1.0; ref_var <= 1.0; ref_var += 1.0)
        {
            for (double sep_var = -1.0; sep_var <= 1.0; sep_var += 1.0)
            {
                // arrange the landmarks in position  
                for (int i = 1; i <= LN; i += 1)
                {
                    landmarkPositionTest[i] = (REF + ref_var) + (i * (SEP + sep_var));
                }
                food_loc_mod = landmarkPositionTest[env];
                // write the landmarks to the output
                for (int i = 1; i <= LN; i += 1){
                    LandmarkFile2 << landmarkPositionTest[i] << " ";
                }
                LandmarkFile2 << food_loc_mod << " ";

                // setup
                Agent.ResetSensors();
                Agent.ResetPosition(0);
                for (int i = 1; i <= N; i++)
                {
                    Agent.NervousSystem.SetNeuronState(i, savedStateHalfWay[i]);
                }
                totaldist = 0.0;
                totaltime = 0.0;

                BehaviorFile2 << Agent.Position() << " ";

                // run the trial
                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    Agent.SenseLandmarks(LN,landmarkPositionTest);
                    Agent.Step(StepSize);
                    BehaviorFile2 << Agent.Position() << " ";

                    // Measure distance between them (after transients)
                    if (time > TransDuration)
                    {
                        dist = fabs(Agent.pos - food_loc_mod);
                        
                        if (dist < mindist){
                            dist = 0.0;
                        }

                        totaldist += dist;

                        totaltime += 1;
                        FitnessFile << 1 - ((totaldist / totaltime)/MinLength) << " ";
                    }
                }

                fit = 1 - ((totaldist / totaltime)/MinLength);
                if (fit < 0.0){
                    fit = 0.0;
                }
                totalfit += fit;

                totaltrials += 1;
            
                BehaviorFile2 << endl;
                LandmarkFile2 << endl;
                FitnessFile << endl;
            }
        }
        BehaviorFile1.close();
        BehaviorFile2.close();
        LandmarkFile1.close();
        LandmarkFile2.close();
        FitnessFile.close();
    }
    
    return (totalfit) / (totaltrials);
}

double RecordBehavior3(TSearch &s) {
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();
    
    // Map genotype to phenotype
    TVector<double> genotype;
    genotype = s.BestIndividual();

    TVector<double> phenotype;
    phenotype.SetBounds(1, (int)(VectSize));
    GenPhenMapping(genotype, phenotype, 1);

    CountingAgent Agent( N, phenotype);

    // Bookeeping variables
    double totaltrials = 0;
    double totaltime;
    double dist;
    double totaldist;
    double totalfit = 0.0;
    double food_loc, food_loc_mod;
    double fit;

    // Landmarks base position and variations
    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1){
        landmarkPositions[i] = REF + (i * SEP);
    }

    TVector<double> landmarkPositionTest;
    landmarkPositionTest.SetBounds(1,LN);

    // save the state
    TVector<double> savedStateStart, savedStateHalfWay;
    savedStateStart.SetBounds(1,N);
    savedStateHalfWay.SetBounds(1,N);

    // Saved each of their neural states 
    for (int i = 1; i <= N; i++)
    {
        savedStateStart[i] = Agent.NervousSystem.NeuronState(i);
    }

    // Phase
    for (int env = 1; env <= LN; env += 1){
        std::string s_env = std::to_string(env);
        ofstream BehaviorFile1;
        BehaviorFile1.open( dir + "behavior_" + current_run + "_Env" + s_env + "_stage3" + "_Phase1.dat");
        ofstream BehaviorFile2;
        BehaviorFile2.open( dir + "behavior_" + current_run + "_Env" + s_env + "_stage3" + "_Phase2.dat");

        // stores the location of the landmarks and the food
        ofstream LandmarkFile1, LandmarkFile2;
        LandmarkFile1.open(dir + "landmark_location_"+current_run+"_env"+s_env+ "_stage3" + "_phase1.dat");
        LandmarkFile2.open(dir + "landmark_location_"+current_run+"_env"+s_env+ "_stage3" +"_phase2.dat");

        ofstream FitnessFile;
        FitnessFile.open(dir + "fitness_"+current_run+"_env"+s_env+ "_stage3" + ".dat");

        // PHASE 1: Setup

        // Establish landmark and food location
        food_loc = landmarkPositions[env];
        for (int i = 1; i <= LN; i += 1){
            LandmarkFile1 << landmarkPositions[i] << " ";
        }
        LandmarkFile1 << food_loc << " ";

        //  setup the agents position + state
        Agent.ResetPosition(0);
        Agent.ResetSensors();
        for (int i = 1; i <= N; i++)
        {
            Agent.NervousSystem.SetNeuronState(i, savedStateStart[i]);
        }


        BehaviorFile1 << Agent.Position() << " ";

        // Step 2: Training Phase
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            Agent.SenseFood(food_loc);
            Agent.SenseLandmarks(LN,landmarkPositions);
            Agent.Step(StepSize);
            BehaviorFile1 << Agent.Position() << " ";
        }

        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            savedStateHalfWay[i] = Agent.NervousSystem.NeuronState(i);
        }
    
        // Testing Phase
        for (double ref_var = -2.0; ref_var <= 2.0; ref_var += 2.0)
        {
            for (double sep_var = -2.0; sep_var <= 2.0; sep_var += 2.0)
            {
                // arrange the landmarks in position  
                for (int i = 1; i <= LN; i += 1)
                {
                    landmarkPositionTest[i] = (REF + ref_var) + (i * (SEP + sep_var));
                }
                food_loc_mod = landmarkPositionTest[env];
                // write the landmarks to the output
                for (int i = 1; i <= LN; i += 1){
                    LandmarkFile2 << landmarkPositionTest[i] << " ";
                }
                LandmarkFile2 << food_loc_mod << " ";

                // setup
                Agent.ResetSensors();
                Agent.ResetPosition(0);
                for (int i = 1; i <= N; i++)
                {
                    Agent.NervousSystem.SetNeuronState(i, savedStateHalfWay[i]);
                }
                totaldist = 0.0;
                totaltime = 0.0;

                BehaviorFile2 << Agent.Position() << " ";

                // run the trial
                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    Agent.SenseLandmarks(LN,landmarkPositionTest);
                    Agent.Step(StepSize);
                    BehaviorFile2 << Agent.Position() << " ";

                    // Measure distance between them (after transients)
                    if (time > TransDuration)
                    {
                        dist = fabs(Agent.pos - food_loc_mod);
                        
                        if (dist < mindist){
                            dist = 0.0;
                        }

                        totaldist += dist;

                        totaltime += 1;
                        FitnessFile << 1 - ((totaldist / totaltime)/MinLength) << " ";
                    }
                }

                fit = 1 - ((totaldist / totaltime)/MinLength);
                if (fit < 0.0){
                    fit = 0.0;
                }
                totalfit += fit;

                totaltrials += 1;
            
                BehaviorFile2 << endl;
                LandmarkFile2 << endl;
                FitnessFile << endl;
            }
        }
        BehaviorFile1.close();
        BehaviorFile2.close();
        LandmarkFile1.close();
        LandmarkFile2.close();
        FitnessFile.close();
    }
    
    return (totalfit) / (totaltrials);
}

double RecordBehavior4(TSearch &s) {
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();
    
    // Map genotype to phenotype
    TVector<double> genotype;
    genotype = s.BestIndividual();

    TVector<double> phenotype;
    phenotype.SetBounds(1, (int)(VectSize));
    GenPhenMapping(genotype, phenotype, 1);

    CountingAgent Agent( N, phenotype);

    // Bookeeping variables
    double totaltrials = 0;
    double totaltime;
    double dist;
    double totaldist;
    double totalfit = 0.0;
    double food_loc, food_loc_mod;
    double fit;

    // Landmarks base position and variations
    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1){
        landmarkPositions[i] = REF + (i * SEP);
    }

    TVector<double> landmarkPositionTest;
    landmarkPositionTest.SetBounds(1,LN);


    // save the state
    TVector<double> trainedState, delayedState;
    trainedState.SetBounds(1,N);
    delayedState.SetBounds(1,N);

    // Saved each of their neural states 
    for (int i = 1; i <= N; i++)
    {
        trainedState[i] = Agent.NervousSystem.NeuronState(i);
    }

    // Phase
    for (int env = 1; env <= LN; env += 1){
        for (int delay=0; delay<=10; delay +=5){ 

            std::string s_env = std::to_string(env);
            std::string s_delay = std::to_string(delay);
            ofstream BehaviorFile1;
            BehaviorFile1.open( dir + "behavior_" + current_run + "_Env" + s_env + "_delay" + s_delay + "_stage4" + "_Phase1.dat");
            ofstream BehaviorFile2;
            BehaviorFile2.open( dir + "behavior_" + current_run + "_Env" + s_env + "_delay" + s_delay + "_stage4" + "_Phase2.dat");
            ofstream BehaviorFile3;
            BehaviorFile3.open( dir + "behavior_" + current_run + "_Env" + s_env + "_delay" + s_delay + "_stage4" + "_Phase3.dat");

            // stores the location of the landmarks and the food
            ofstream LandmarkFile1, LandmarkFile2, LandmarkFile3;
            LandmarkFile1.open(dir + "landmark_location_"+current_run+"_env"+s_env+"_delay" + s_delay + "_stage4" + "_phase1.dat");
            LandmarkFile2.open(dir + "landmark_location_"+current_run+"_env"+s_env+"_delay" + s_delay + "_stage4" + "_phase2.dat");
            LandmarkFile3.open(dir + "landmark_location_"+current_run+"_env"+s_env+"_delay" + s_delay + "_stage4"+ "_phase3.dat");


            ofstream FitnessFile1;
            FitnessFile1.open(dir + "fitness_"+current_run+"_env"+s_env+"_delay" + s_delay + "_stage4" + "_phase1.dat");


            // PHASE 1 : training

            // Establish food location
            food_loc = landmarkPositions[env];
            // write the landmarks to the output
            for (int i = 1; i <= LN; i += 1){
                LandmarkFile1 << landmarkPositions[i] << " ";
            }
            LandmarkFile1 << food_loc << " ";

            Agent.ResetPosition(0);
            for (int i = 1; i <= N; i++)
            {
                Agent.NervousSystem.SetNeuronState(i, trainedState[i]);
            }

            BehaviorFile1 << Agent.Position() << " ";

            // Training loop
            for (double time = 0; time < RunDuration; time += StepSize)
            {
                Agent.SenseFood(food_loc);
                Agent.SenseLandmarks(LN,landmarkPositions);
                Agent.Step(StepSize);
                BehaviorFile1 << Agent.Position() << " ";
            }

            // 2. delay
            // setup 
            Agent.ResetPosition(0); 
            Agent.ResetSensors();
            for (int i = 1; i <= LN; i += 1){
                LandmarkFile2 << landmarkPositions[i] << " ";
            }
            LandmarkFile2 << food_loc << " ";

            BehaviorFile2 << Agent.Position() << " ";
            // Delay loop
            for (double time=0; time < delay; time += StepSize){
                // not sure if it should step or not
                Agent.Step(StepSize);
                BehaviorFile2 << Agent.Position() << " ";
            }

            // After the delay - Saved each of their neural states 
            for (int i = 1; i <= N; i++)
            {
                delayedState[i] = Agent.NervousSystem.NeuronState(i);
            }

            // Testing Phase
            for (double ref_var = -2.0; ref_var <= 2.0; ref_var += 2.0)
            {
                for (double sep_var = -2.0; sep_var <= 2.0; sep_var += 2.0)
                {
                    // arrange the landmarks in position  
                    for (int i = 1; i <= LN; i += 1)
                    {
                        landmarkPositionTest[i] = (REF + ref_var) + (i * (SEP + sep_var));
                    }
                    food_loc_mod = landmarkPositionTest[env];
                    
                    // write the landmarks to the output
                    for (int i = 1; i <= LN; i += 1){
                        LandmarkFile3 << landmarkPositionTest[i] << " ";
                    }
                    LandmarkFile3 << food_loc_mod << " ";

                    // setup
                    Agent.ResetPosition(0);
                    Agent.ResetSensors();
                    for (int i = 1; i <= N; i++)
                    {
                        Agent.NervousSystem.SetNeuronState(i, delayedState[i]);
                    }
                    totaldist = 0.0;
                    totaltime = 0.0;

                    BehaviorFile3 << Agent.Position() << " ";

                    // run the trial
                    for (double time = 0; time < RunDuration; time += StepSize)
                    {
                        Agent.SenseLandmarks(LN,landmarkPositionTest);
                        Agent.Step(StepSize);
                        BehaviorFile3 << Agent.Position() << " ";

                        // Measure distance between them (after transients)
                        if (time > TransDuration)
                        {
                            dist = fabs(Agent.pos - food_loc_mod);
                            
                            if (dist < mindist){
                                dist = 0.0;
                            }

                            totaldist += dist;

                            totaltime += 1;
                            FitnessFile1 << 1 - ((totaldist / totaltime)/MinLength) << " ";
                        }
                    }

                    fit = 1 - ((totaldist / totaltime)/MinLength);
                    if (fit < 0.0){
                        fit = 0.0;
                    }
                    totalfit += fit;

                    totaltrials += 1;
                
                    BehaviorFile3 << endl;
                    LandmarkFile3 << endl;
                    FitnessFile1 << endl;
                }
            }
            BehaviorFile1.close();
            BehaviorFile2.close();
            BehaviorFile3.close();
            LandmarkFile1.close();
            LandmarkFile2.close();
            LandmarkFile3.close();
            FitnessFile1.close();
        }
    }
    return (totalfit) / (totaltrials);
}


double RecordBehaviorTest(TSearch &s) {
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();
    
    // Map genotype to phenotype
    TVector<double> genotype;
    genotype = s.BestIndividual();

    TVector<double> phenotype;
    phenotype.SetBounds(1, (int)(VectSize));
    GenPhenMapping(genotype, phenotype, 1);

    CountingAgent Agent( N, phenotype);

    // Bookeeping variables
    double totaltrials = 0;
    double totaltime;
    double dist;
    double totaldist;
    double totalfit = 0.0;
    double food_loc, food_loc_mod;
    double fit;

    // Landmarks base position and variations
    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1){
        landmarkPositions[i] = REF + (i * SEP);
    }

    TVector<double> landmarkPositionTest;
    landmarkPositionTest.SetBounds(1,LN);


    // save the state
    TVector<double> trainedState, delayedState;
    trainedState.SetBounds(1,N);
    delayedState.SetBounds(1,N);

    // Saved each of their neural states 
    for (int i = 1; i <= N; i++)
    {
        trainedState[i] = Agent.NervousSystem.NeuronState(i);
    }

    // Phase
    for (int env = 1; env <= LN; env += 1){
        for (int delay=0; delay<=10; delay +=5){ 

            std::string s_env = std::to_string(env);
            std::string s_delay = std::to_string(delay);
            ofstream BehaviorFile1;
            BehaviorFile1.open( dir + "behavior_" + current_run + "_Env" + s_env + "_delay" + s_delay + "_test" + "_Phase1.dat");
            ofstream BehaviorFile2;
            BehaviorFile2.open( dir + "behavior_" + current_run + "_Env" + s_env + "_delay" + s_delay + "_test" + "_Phase2.dat");
            ofstream BehaviorFile3;
            BehaviorFile3.open( dir + "behavior_" + current_run + "_Env" + s_env + "_delay" + s_delay + "_test" + "_Phase3.dat");

            // stores the location of the landmarks and the food
            ofstream LandmarkFile1, LandmarkFile2, LandmarkFile3;
            LandmarkFile1.open(dir + "landmark_location_"+current_run+"_env"+s_env+"_delay" + s_delay + "_test" + "_phase1.dat");
            LandmarkFile2.open(dir + "landmark_location_"+current_run+"_env"+s_env+"_delay" + s_delay + "_test" + "_phase2.dat");
            LandmarkFile3.open(dir + "landmark_location_"+current_run+"_env"+s_env+"_delay" + s_delay + "_test"+ "_phase3.dat");


            ofstream FitnessFile1;
            FitnessFile1.open(dir + "fitness_"+current_run+"_env"+s_env+"_delay" + s_delay + "_test" + "_phase1.dat");


            // PHASE 1 : training

            // Establish food location
            food_loc = landmarkPositions[env];
            // write the landmarks to the output
            for (int i = 1; i <= LN; i += 1){
                LandmarkFile1 << landmarkPositions[i] << " ";
            }
            LandmarkFile1 << food_loc << " ";

            Agent.ResetPosition(0);
            for (int i = 1; i <= N; i++)
            {
                Agent.NervousSystem.SetNeuronState(i, trainedState[i]);
            }

            BehaviorFile1 << Agent.Position() << " ";

            // Training loop
            for (double time = 0; time < RunDuration; time += StepSize)
            {
                Agent.SenseFood(food_loc);
                Agent.SenseLandmarks(LN,landmarkPositions);
                Agent.Step(StepSize);
                BehaviorFile1 << Agent.Position() << " ";
            }

            // 2. delay
            // setup 
            Agent.ResetPosition(0); 
            Agent.ResetSensors();
            for (int i = 1; i <= LN; i += 1){
                LandmarkFile2 << landmarkPositions[i] << " ";
            }
            LandmarkFile2 << food_loc << " ";

            BehaviorFile2 << Agent.Position() << " ";
            // Delay loop
            for (double time=0; time < delay; time += StepSize){
                // not sure if it should step or not
                Agent.Step(StepSize);
                BehaviorFile2 << Agent.Position() << " ";
            }

            // After the delay - Saved each of their neural states 
            for (int i = 1; i <= N; i++)
            {
                delayedState[i] = Agent.NervousSystem.NeuronState(i);
            }

            // Testing Phase
            for (double ref_var = -4.0; ref_var <= 4.0; ref_var += 2.0)
            {
                for (double sep_var = -4.0; sep_var <= 4.0; sep_var += 2.0)
                {
                    // arrange the landmarks in position  
                    for (int i = 1; i <= LN; i += 1)
                    {
                        landmarkPositionTest[i] = (REF + ref_var) + (i * (SEP + sep_var));
                    }
                    food_loc_mod = landmarkPositionTest[env];
                    
                    // write the landmarks to the output
                    for (int i = 1; i <= LN; i += 1){
                        LandmarkFile3 << landmarkPositionTest[i] << " ";
                    }
                    LandmarkFile3 << food_loc_mod << " ";

                    // setup
                    Agent.ResetPosition(0);
                    Agent.ResetSensors();
                    for (int i = 1; i <= N; i++)
                    {
                        Agent.NervousSystem.SetNeuronState(i, delayedState[i]);
                    }
                    totaldist = 0.0;
                    totaltime = 0.0;

                    BehaviorFile3 << Agent.Position() << " ";

                    // run the trial
                    for (double time = 0; time < RunDuration; time += StepSize)
                    {
                        Agent.SenseLandmarks(LN,landmarkPositionTest);
                        Agent.Step(StepSize);
                        BehaviorFile3 << Agent.Position() << " ";

                        // Measure distance between them (after transients)
                        if (time > TransDuration)
                        {
                            dist = fabs(Agent.pos - food_loc_mod);
                            
                            if (dist < mindist){
                                dist = 0.0;
                            }

                            totaldist += dist;

                            totaltime += 1;
                            FitnessFile1 << 1 - ((totaldist / totaltime)/MinLength) << " ";
                        }
                    }

                    fit = 1 - ((totaldist / totaltime)/MinLength);
                    if (fit < 0.0){
                        fit = 0.0;
                    }
                    totalfit += fit;

                    totaltrials += 1;
                
                    BehaviorFile3 << endl;
                    LandmarkFile3 << endl;
                    FitnessFile1 << endl;
                }
            }
            BehaviorFile1.close();
            BehaviorFile2.close();
            BehaviorFile3.close();
            LandmarkFile1.close();
            LandmarkFile2.close();
            LandmarkFile3.close();
            FitnessFile1.close();
        }
    }
    return (totalfit) / (totaltrials);
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
    phenotypeS.SetBounds(1, (int) (VectSize));

    // Save the genotype of the best individual
    bestVector = s.BestIndividual();
    BestIndividualFile.open( dir + "best_gen_" + current_run + ".dat");
    BestIndividualFile << bestVector << endl;
    BestIndividualFile.close();

    GenPhenMapping(bestVector, phenotypeS, 1);

    // Show the Signaler
    BestIndividualFile.open( dir + "best_ns_s_" + current_run + ".dat" );
    CountingAgent Agent( N, phenotypeS);

    // Send to file
    BestIndividualFile << "Nervous System:" << endl;
    BestIndividualFile << Agent.NervousSystem << endl;
    BestIndividualFile << "Food Sensor Weights:" << endl;
    BestIndividualFile << Agent.foodsensorweights << "\n" << endl;
    BestIndividualFile << "Landmark Sensor Weights:" << endl;
    BestIndividualFile << Agent.landmarksensorweights << "\n" << endl;
    BestIndividualFile << "Other Sensor Weights:" << endl;
    BestIndividualFile << Agent.othersensorweights << "\n" << endl;
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
            << "Min Length: " << MinLength << "\n"
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
            << "\n*********Landmark Parameters*********\n"
            << "Sep " << SEP << "\n"
            << "Ref " << REF << "\n"
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
    // are the neural states transfered between each stageof the evolution -> not currently

    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(FitnessFunction1);
    search.ExecuteSearch();

    if (search.BestPerformance() > 0.99) {
        RecordBehavior1(search);
    }
    
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(FitnessFunction2);
    search.ExecuteSearch();

    if (search.BestPerformance() > 0.99) {
        RecordBehavior2(search);
    }

    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(FitnessFunction3);
    search.ExecuteSearch();
        if (search.BestPerformance() > 0.99) {
        RecordBehavior3(search);
    }

    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(FitnessFunction4);
    search.ExecuteSearch();

    if (search.BestPerformance() > 0.99) {
        RecordBehavior4(search);
    }

    if (search.BestPerformance() > 0.99){
        RecordBehaviorTest(search);
    }

    #ifdef PRINTTOFILE
        file.close();
    #endif

    return 0;
}
