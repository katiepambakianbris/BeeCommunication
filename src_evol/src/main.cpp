#include <iostream>
#include "TSearch.h"
#include "CountingAgent.h"
#include "CTRNN.h"
#include "random.h"
#include <ctime> 
#include <sstream>
#include <filesystem>
#include <fstream>

// #define PRINTOFILE

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
// Fitness function
// Landmarks vary position by small amount
// ------------------------------------
double FitnessFunction1(TVector<double> &genotype, RandomState &rs)
{
    // Map genotype to phenotype
    TVector<double> phenotypeS;
    phenotypeS.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeS, 1);

    // construct the signaller agent
    CountingAgent AgentSignaler( N, phenotypeS);

    // Save state
    TVector<double> savedstateS;
    savedstateS.SetBounds(1,N);

    // Keep track of performance (fitness across landmarks/env)
    double totaltrials = 0;
    double totaltime;
    double distS;
    double totaldistS;
    double totalfitS = 0.0;
    double food_loc, food_loc_mod;
    double fitS;

    // ************************
    // Set up Landmarks
    // ************************

    double ref = 10;
    double sep = 10;

    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1)
    {
        landmarkPositions[i] = ref + (i * sep);
    }

    TVector<double> landmarkPositionTest;
    landmarkPositionTest.SetBounds(1,LN);  
    
    // Use this to save the neural state during learning

    // loop through different enviornment 
    // for each enviornment i, landmark i has the food 
    for (int env = 1; env <= LN; env += 1)
    {
        // *******************
        // 1. FORAGING PHASE
        // *******************

        // Step 1: Setup

        // Establish food location
        food_loc = landmarkPositions[env];
    
        // reset position + neural state of the signaller
        AgentSignaler.ResetPosition(0); 
        AgentSignaler.ResetNeuralState();

        // Step 2: Loop
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            AgentSignaler.SenseFood(food_loc);
            AgentSignaler.SenseLandmarks(LN,landmarkPositions);
            AgentSignaler.Step(StepSize);
        }
        AgentSignaler.ResetSensors();

        // 2. RECRUITMENT PHASE
        AgentSignaler.ResetPosition(0);
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            AgentSignaler.SenseOther(0.0);
            AgentSignaler.Step(StepSize);
        }
        AgentSignaler.ResetSensors();

        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            savedstateS[i] = AgentSignaler.NervousSystem.NeuronState(i);
        }

        // TEST USING DIFFERENT DISTANCES BETWEEN LANDMARKS
        for (double ref_var = 0.0; ref_var <= 0.0; ref_var += 1.0)
        {
            for (double sep_var = 0.0; sep_var <= 0.0; sep_var += 1.0)
            {
                for (int i = 1; i <= LN; i += 1)
                {
                    landmarkPositionTest[i] = (ref + ref_var) + (i * (sep + sep_var));
                }
                food_loc_mod = landmarkPositionTest[env];

                // 3. TESTING PHASE
                AgentSignaler.ResetPosition(0);
                AgentSignaler.ResetSensors();
                // Reset neural state
                for (int i = 1; i <= N; i++)
                {
                    AgentSignaler.NervousSystem.SetNeuronState(i, savedstateS[i]);
                }
                
                totaldistS = 0.0;
                totaltime = 0.0;
        
                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    AgentSignaler.SenseLandmarks(LN,landmarkPositionTest);
                    AgentSignaler.Step(StepSize);

                    // Measure distance between them (after transients)
                    if (time > TransDuration)
                    {
                        distS = fabs(AgentSignaler.pos - food_loc_mod);
                        
                        if (distS < mindist){
                            distS = 0.0;
                        }
                        totaldistS += distS;

                        totaltime += 1;
                    }
                }

                fitS = 1 - ((totaldistS / totaltime)/MinLength);
                if (fitS < 0.0){
                    fitS = 0.0;
                }
                totalfitS += fitS;

                totaltrials += 1;
            }
        }
    }
    return totalfitS / totaltrials;
}

// ------------------------------------
// Fitness function
// Landmarks vary position by small amount
// ------------------------------------
double FitnessFunction2(TVector<double> &genotype, RandomState &rs)
{
    // Map genotype to phenotype
    TVector<double> phenotypeS, phenotypeR;
    phenotypeS.SetBounds(1, (int)(VectSize/2));
    phenotypeR.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeS, 1);
    GenPhenMapping(genotype, phenotypeR, (int)(N*N + 5*N + 1));

    CountingAgent AgentSignaler( N, phenotypeS);
    CountingAgent AgentReceiver( N, phenotypeR);

    // Save state
    TVector<double> savedstateR, savedstateS;
    savedstateR.SetBounds(1,N);
    savedstateS.SetBounds(1,N);

    // Keep track of performance
    double totaltrials = 0;
    double totaltime;
    double distR, distS;
    double totaldistR, totaldistS;
    double totalfitR = 0.0, totalfitS = 0.0;
    double food_loc, food_loc_mod;
    double fitR, fitS;
    double ref = 10;
    double sep = 10;

    // Landmarks and variations
    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1)
    {
        landmarkPositions[i] = ref + (i * sep);
    }

    TVector<double> landmarkPositionTest;
    landmarkPositionTest.SetBounds(1,LN);  
    
    // Use this to save the neural state during learning
    for (int env = 1; env <= LN; env += 1)
    {
        // Establish food location
        food_loc = landmarkPositions[env];

        // 1. FORAGING PHASE
        AgentSignaler.ResetPosition(0);
        AgentSignaler.ResetNeuralState();
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            AgentSignaler.SenseFood(food_loc);
            AgentSignaler.SenseLandmarks(LN,landmarkPositions);
            AgentSignaler.Step(StepSize);
        }
        AgentSignaler.ResetSensors();

        // 2. RECRUITMENT PHASE
        AgentSignaler.ResetPosition(0);
        AgentReceiver.ResetPosition(0);
        AgentReceiver.ResetNeuralState();
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            AgentSignaler.SenseOther(AgentReceiver.pos);
            AgentReceiver.SenseOther(AgentSignaler.pos);
            AgentSignaler.Step(StepSize);
            AgentReceiver.Step(StepSize);
        }
        AgentReceiver.ResetSensors();
        AgentSignaler.ResetSensors();

        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            savedstateR[i] = AgentReceiver.NervousSystem.NeuronState(i);
            savedstateS[i] = AgentSignaler.NervousSystem.NeuronState(i);
        }

        // TEST USING DIFFERENT DISTANCES BETWEEN LANDMARKS
        for (double ref_var = 0.0; ref_var <= 0.0; ref_var += 1.0)
        {
            for (double sep_var = 0.0; sep_var <= 0.0; sep_var += 1.0)
            {
                for (int i = 1; i <= LN; i += 1)
                {
                    landmarkPositionTest[i] = (ref + ref_var) + (i * (sep + sep_var));
                }
                food_loc_mod = landmarkPositionTest[env];

                // 3. TESTING PHASE
                AgentReceiver.ResetPosition(0);
                AgentSignaler.ResetPosition(0);
                AgentReceiver.ResetSensors();
                AgentSignaler.ResetSensors();
                // Reset neural state
                for (int i = 1; i <= N; i++)
                {
                    AgentReceiver.NervousSystem.SetNeuronState(i, savedstateR[i]);
                    AgentSignaler.NervousSystem.SetNeuronState(i, savedstateS[i]);
                }
                
                totaldistR = 0.0; totaldistS = 0.0;
                totaltime = 0.0;
        
                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    AgentReceiver.SenseLandmarks(LN,landmarkPositionTest); 
                    AgentSignaler.SenseLandmarks(LN,landmarkPositionTest);
                    AgentReceiver.Step(StepSize);
                    AgentSignaler.Step(StepSize);

                    // Measure distance between them (after transients)
                    if (time > TransDuration)
                    {
                        distR = fabs(AgentReceiver.pos - food_loc_mod);
                        if (distR < mindist){
                            distR = 0.0;
                        }
                        totaldistR += distR;

                        distS = fabs(AgentSignaler.pos - food_loc_mod);
                        
                        if (distS < mindist){
                            distS = 0.0;
                        }
                        totaldistS += distS;

                        totaltime += 1;
                    }
                }
                
                fitR = 1 - ((totaldistR / totaltime)/MinLength);
                if (fitR < 0){
                    fitR = 0;
                }
                totalfitR += fitR;

                fitS = 1 - ((totaldistS / totaltime)/MinLength);
                if (fitS < 0.0){
                    fitS = 0.0;
                }
                totalfitS += fitS;

                totaltrials += 1;
            }
        }
    }
    return (totalfitR + totalfitS) / (2 * totaltrials);
}

// ------------------------------------
// Fitness function
// Landmarks vary position by small amount
// ------------------------------------
double FitnessFunction3(TVector<double> &genotype, RandomState &rs)
{
    // Map genotype to phenotype
    TVector<double> phenotypeS, phenotypeR;
    phenotypeS.SetBounds(1, (int)(VectSize/2));
    phenotypeR.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeS, 1);
    GenPhenMapping(genotype, phenotypeR, (int)(N*N + 5*N + 1));

    CountingAgent AgentSignaler( N, phenotypeS);
    CountingAgent AgentReceiver( N, phenotypeR);

    // Save state
    TVector<double> savedstateR, savedstateS;
    savedstateR.SetBounds(1,N);
    savedstateS.SetBounds(1,N);

    // Keep track of performance
    double totaltrials = 0;
    double totaltime;
    double distR, distS;
    double totaldistR, totaldistS;
    double totalfitR = 0.0, totalfitS = 0.0;
    double food_loc, food_loc_mod;
    double fitR, fitS;
    double ref = 10;
    double sep = 10;

    // Landmarks and variations
    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1)
    {
        landmarkPositions[i] = ref + (i * sep);
    }

    TVector<double> landmarkPositionTest;
    landmarkPositionTest.SetBounds(1,LN);  
    
    // Use this to save the neural state during learning
    for (int env = 1; env <= LN; env += 1)
    {
        // Establish food location
        food_loc = landmarkPositions[env];

        // 1. FORAGING PHASE
        AgentSignaler.ResetPosition(0);
        AgentSignaler.ResetNeuralState();
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            AgentSignaler.SenseFood(food_loc);
            AgentSignaler.SenseLandmarks(LN,landmarkPositions);
            AgentSignaler.Step(StepSize);
        }
        AgentSignaler.ResetSensors();

        // 2. RECRUITMENT PHASE
        AgentSignaler.ResetPosition(0);
        AgentReceiver.ResetPosition(0);
        AgentReceiver.ResetNeuralState();
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            AgentSignaler.SenseOther(AgentReceiver.pos);
            AgentReceiver.SenseOther(AgentSignaler.pos);
            AgentSignaler.Step(StepSize);
            AgentReceiver.Step(StepSize);
        }
        AgentReceiver.ResetSensors();
        AgentSignaler.ResetSensors();

        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            savedstateR[i] = AgentReceiver.NervousSystem.NeuronState(i);
            savedstateS[i] = AgentSignaler.NervousSystem.NeuronState(i);
        }

        // TEST USING DIFFERENT DISTANCES BETWEEN LANDMARKS
        for (double ref_var = -1.0; ref_var <= 1.0; ref_var += 1.0)
        {
            for (double sep_var = -1.0; sep_var <= 1.0; sep_var += 1.0)
            {
                for (int i = 1; i <= LN; i += 1)
                {
                    landmarkPositionTest[i] = (ref + ref_var) + (i * (sep + sep_var));
                }
                food_loc_mod = landmarkPositionTest[env];

                // 3. TESTING PHASE
                AgentReceiver.ResetPosition(0);
                AgentSignaler.ResetPosition(0);
                AgentReceiver.ResetSensors();
                AgentSignaler.ResetSensors();
                // Reset neural state
                for (int i = 1; i <= N; i++)
                {
                    AgentReceiver.NervousSystem.SetNeuronState(i, savedstateR[i]);
                    AgentSignaler.NervousSystem.SetNeuronState(i, savedstateS[i]);
                }
                
                totaldistR = 0.0; totaldistS = 0.0;
                totaltime = 0.0;
        
                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    AgentReceiver.SenseLandmarks(LN,landmarkPositionTest); 
                    AgentSignaler.SenseLandmarks(LN,landmarkPositionTest);
                    AgentReceiver.Step(StepSize);
                    AgentSignaler.Step(StepSize);

                    // Measure distance between them (after transients)
                    if (time > TransDuration)
                    {
                        distR = fabs(AgentReceiver.pos - food_loc_mod);
                        if (distR < mindist){
                            distR = 0.0;
                        }
                        totaldistR += distR;

                        distS = fabs(AgentSignaler.pos - food_loc_mod);
                        
                        if (distS < mindist){
                            distS = 0.0;
                        }
                        totaldistS += distS;

                        totaltime += 1;
                    }
                }
                
                fitR = 1 - ((totaldistR / totaltime)/MinLength);
                if (fitR < 0){
                    fitR = 0;
                }
                totalfitR += fitR;

                fitS = 1 - ((totaldistS / totaltime)/MinLength);
                if (fitS < 0.0){
                    fitS = 0.0;
                }
                totalfitS += fitS;

                totaltrials += 1;
            }
        }
    }
    return (totalfitR + totalfitS) / (2 * totaltrials);
}

// ------------------------------------
// Fitness function
// Landmarks vary position by small amount
// ------------------------------------
double FitnessFunction4(TVector<double> &genotype, RandomState &rs)
{
    // Map genotype to phenotype
    TVector<double> phenotypeS, phenotypeR;
    phenotypeS.SetBounds(1, (int)(VectSize/2));
    phenotypeR.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeS, 1);
    GenPhenMapping(genotype, phenotypeR, (int)(N*N + 5*N + 1));

    CountingAgent AgentSignaler( N, phenotypeS);
    CountingAgent AgentReceiver( N, phenotypeR);

    // Save state
    TVector<double> savedstateR, savedstateS;
    savedstateR.SetBounds(1,N);
    savedstateS.SetBounds(1,N);

    // Keep track of performance
    double totaltrials = 0;
    double totaltime;
    double distR, distS;
    double totaldistR, totaldistS;
    double totalfitR = 0.0, totalfitS = 0.0;
    double food_loc, food_loc_mod;
    double fitR, fitS;
    double ref = 10;
    double sep = 10;

    // Landmarks and variations
    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1)
    {
        landmarkPositions[i] = ref + (i * sep);
    }

    TVector<double> landmarkPositionTest;
    landmarkPositionTest.SetBounds(1,LN);  
    
    // Use this to save the neural state during learning
    for (int env = 1; env <= LN; env += 1)
    {
        // Establish food location
        food_loc = landmarkPositions[env];

        // 1. FORAGING PHASE
        AgentSignaler.ResetPosition(0);
        AgentSignaler.ResetNeuralState();
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            AgentSignaler.SenseFood(food_loc);
            AgentSignaler.SenseLandmarks(LN,landmarkPositions);
            AgentSignaler.Step(StepSize);
        }
        AgentSignaler.ResetSensors();

        // 2. RECRUITMENT PHASE
        AgentSignaler.ResetPosition(0);
        AgentReceiver.ResetPosition(0);
        AgentReceiver.ResetNeuralState();
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            AgentSignaler.SenseOther(AgentReceiver.pos);
            AgentReceiver.SenseOther(AgentSignaler.pos);
            AgentSignaler.Step(StepSize);
            AgentReceiver.Step(StepSize);
        }
        AgentReceiver.ResetSensors();
        AgentSignaler.ResetSensors();

        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            savedstateR[i] = AgentReceiver.NervousSystem.NeuronState(i);
            savedstateS[i] = AgentSignaler.NervousSystem.NeuronState(i);
        }

        // TEST USING DIFFERENT DISTANCES BETWEEN LANDMARKS
        for (double ref_var = -2.0; ref_var <= 2.0; ref_var += 1.0)
        {
            for (double sep_var = -2.0; sep_var <= 2.0; sep_var += 1.0)
            {
                for (int i = 1; i <= LN; i += 1)
                {
                    landmarkPositionTest[i] = (ref + ref_var) + (i * (sep + sep_var));
                }
                food_loc_mod = landmarkPositionTest[env];

                // 3. TESTING PHASE
                AgentReceiver.ResetPosition(0);
                AgentSignaler.ResetPosition(0);
                AgentReceiver.ResetSensors();
                AgentSignaler.ResetSensors();
                // Reset neural state
                for (int i = 1; i <= N; i++)
                {
                    AgentReceiver.NervousSystem.SetNeuronState(i, savedstateR[i]);
                    AgentSignaler.NervousSystem.SetNeuronState(i, savedstateS[i]);
                }
                
                totaldistR = 0.0; totaldistS = 0.0;
                totaltime = 0.0;
        
                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    AgentReceiver.SenseLandmarks(LN,landmarkPositionTest); 
                    AgentSignaler.SenseLandmarks(LN,landmarkPositionTest);
                    AgentReceiver.Step(StepSize);
                    AgentSignaler.Step(StepSize);

                    // Measure distance between them (after transients)
                    if (time > TransDuration)
                    {
                        distR = fabs(AgentReceiver.pos - food_loc_mod);
                        if (distR < mindist){
                            distR = 0.0;
                        }
                        totaldistR += distR;

                        distS = fabs(AgentSignaler.pos - food_loc_mod);
                        
                        if (distS < mindist){
                            distS = 0.0;
                        }
                        totaldistS += distS;

                        totaltime += 1;
                    }
                }
                
                fitR = 1 - ((totaldistR / totaltime)/MinLength);
                if (fitR < 0){
                    fitR = 0;
                }
                totalfitR += fitR;

                fitS = 1 - ((totaldistS / totaltime)/MinLength);
                if (fitS < 0.0){
                    fitS = 0.0;
                }
                totalfitS += fitS;

                totaltrials += 1;
            }
        }
    }
    return (totalfitR + totalfitS) / (2 * totaltrials);
}

// ------------------------------------
// Fitness function
// Landmarks vary position by small amount
// ------------------------------------
double RecordBehavior(TSearch &s) //, RandomState &rs)
{
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();
    
    TVector<double> genotype;
    genotype = s.BestIndividual();

    // Map genotype to phenotype
    TVector<double> phenotypeS, phenotypeR;
    phenotypeS.SetBounds(1, (int)(VectSize/2));
    phenotypeR.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeS, 1);
    GenPhenMapping(genotype, phenotypeR, (int)(N*N + 5*N + 1));

    CountingAgent AgentSignaler( N, phenotypeS);
    CountingAgent AgentReceiver( N, phenotypeR);

    // Save state
    TVector<double> savedstateR, savedstateS;
    savedstateR.SetBounds(1,N);
    savedstateS.SetBounds(1,N);

    // Keep track of performance
    double totaltrials = 0;
    double totaltime;
    double distR, distS;
    double totaldistR, totaldistS;
    double totalfitR = 0.0, totalfitS = 0.0;
    double food_loc, food_loc_mod;
    double fitR, fitS;
    double ref = 10;
    double sep = 10;

    // Landmarks and variations
    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1)
    {
        landmarkPositions[i] = ref + (i * sep);
    }

    TVector<double> landmarkPositionTest;
    landmarkPositionTest.SetBounds(1,LN);  
    
    // Use this to save the neural state during learning
    for (int env = 1; env <= LN; env += 1)
    {
        std::string s_env = std::to_string(env);
        ofstream SignalerBehaviorFile1, ReceiverBehaviorFile1;
        SignalerBehaviorFile1.open( dir + "behavior_Signaler_" + current_run + "_Env" + s_env + "_Phase1.dat");
        ReceiverBehaviorFile1.open( dir + "behavior_Receiver_" + current_run + "_Env" + s_env + "_Phase1.dat");
        ofstream SignalerBehaviorFile2, ReceiverBehaviorFile2;
        SignalerBehaviorFile2.open( dir + "behavior_Signaler_" + current_run + "_Env" + s_env + "_Phase2.dat");
        ReceiverBehaviorFile2.open( dir + "behavior_Receiver_" + current_run + "_Env" + s_env + "_Phase2.dat");
        ofstream SignalerBehaviorFile3, ReceiverBehaviorFile3;
        SignalerBehaviorFile3.open( dir + "behavior_Signaler_" + current_run + "_Env" + s_env + "_Phase3.dat");
        ReceiverBehaviorFile3.open( dir + "behavior_Receiver_" + current_run + "_Env" + s_env + "_Phase3.dat");

        // TEST USING DIFFERENT DISTANCES BETWEEN LANDMARKS
        for (double ref_var = -2.0; ref_var <= 2.0; ref_var += 1.0)
        {
            for (double sep_var = -2.0; sep_var <= 2.0; sep_var += 1.0)
            {
                // Establish food location
                food_loc = landmarkPositions[env];

                // 1. FORAGING PHASE
                AgentSignaler.ResetPosition(0);
                AgentSignaler.ResetNeuralState();

                SignalerBehaviorFile1 << AgentSignaler.Position() << " ";
                ReceiverBehaviorFile1 << 0.0 << " ";

                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    AgentSignaler.SenseFood(food_loc);
                    AgentSignaler.SenseLandmarks(LN,landmarkPositions);
                    AgentSignaler.Step(StepSize);
                    SignalerBehaviorFile1 << AgentSignaler.Position() << " ";
                    ReceiverBehaviorFile1 << 0.0 << " ";
                }
                AgentSignaler.ResetSensors();

                // 2. RECRUITMENT PHASE
                AgentSignaler.ResetPosition(0);
                AgentReceiver.ResetPosition(0);
                AgentReceiver.ResetNeuralState();

                SignalerBehaviorFile2 << AgentSignaler.Position() << " ";
                ReceiverBehaviorFile2 << AgentReceiver.Position() << " ";
                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    AgentSignaler.SenseOther(AgentReceiver.pos);
                    AgentReceiver.SenseOther(AgentSignaler.pos);
                    AgentSignaler.Step(StepSize);
                    AgentReceiver.Step(StepSize);
                    SignalerBehaviorFile2 << AgentSignaler.Position() << " ";
                    ReceiverBehaviorFile2 << AgentReceiver.Position() << " ";
                }
                AgentReceiver.ResetSensors();
                AgentSignaler.ResetSensors();

                for (int i = 1; i <= LN; i += 1)
                {
                    landmarkPositionTest[i] = (ref + ref_var) + (i * (sep + sep_var));
                }
                food_loc_mod = landmarkPositionTest[env];

                // 3. TESTING PHASE
                AgentReceiver.ResetPosition(0);
                AgentSignaler.ResetPosition(0);
                AgentReceiver.ResetSensors();
                AgentSignaler.ResetSensors();
                
                totaldistR = 0.0; totaldistS = 0.0;
                totaltime = 0.0;
        
                SignalerBehaviorFile3 << AgentSignaler.Position() << " ";
                ReceiverBehaviorFile3 << AgentReceiver.Position() << " ";

                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    AgentReceiver.SenseLandmarks(LN,landmarkPositionTest); 
                    AgentSignaler.SenseLandmarks(LN,landmarkPositionTest);
                    AgentReceiver.Step(StepSize);
                    AgentSignaler.Step(StepSize);
                    SignalerBehaviorFile3 << AgentSignaler.Position() << " ";
                    ReceiverBehaviorFile3 << AgentReceiver.Position() << " ";

                    // Measure distance between them (after transients)
                    if (time > TransDuration)
                    {
                        distR = fabs(AgentReceiver.pos - food_loc_mod);
                        if (distR < mindist){
                            distR = 0.0;
                        }
                        totaldistR += distR;

                        distS = fabs(AgentSignaler.pos - food_loc_mod);
                        
                        if (distS < mindist){
                            distS = 0.0;
                        }
                        totaldistS += distS;

                        totaltime += 1;
                    }
                }
                
                fitR = 1 - ((totaldistR / totaltime)/MinLength);
                if (fitR < 0){
                    fitR = 0;
                }
                totalfitR += fitR;

                fitS = 1 - ((totaldistS / totaltime)/MinLength);
                if (fitS < 0.0){
                    fitS = 0.0;
                }
                totalfitS += fitS;

                totaltrials += 1;
            
                SignalerBehaviorFile1 << endl;
                ReceiverBehaviorFile1 << endl;
                SignalerBehaviorFile2 << endl;
                ReceiverBehaviorFile2 << endl;
                SignalerBehaviorFile3 << endl;
                ReceiverBehaviorFile3 << endl;

            }   
        }
        SignalerBehaviorFile1.close();
        ReceiverBehaviorFile1.close();
        SignalerBehaviorFile2.close();
        ReceiverBehaviorFile2.close();
        SignalerBehaviorFile3.close();
        ReceiverBehaviorFile3.close();
    }
    return (totalfitR + totalfitS) / (2 * totaltrials);
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
    BestIndividualFile << AgentReceiver.NervousSystem << endl;
    BestIndividualFile << AgentReceiver.foodsensorweights << "\n" << endl;
    BestIndividualFile << AgentReceiver.landmarksensorweights << "\n" << endl;
    BestIndividualFile << AgentReceiver.othersensorweights << "\n" << endl;
    BestIndividualFile.close();

}

std::string date_as_string(){
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);

    std::ostringstream oss;
    // oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    oss << std::put_time(&tm, "%Y-%m-%d");
    return oss.str();
}

void output_config(std::string dir, std::string run, std::string batch){
    // open the file
    ofstream configfile;
    configfile.open (dir + "config_" + run + ".dat");

    // print the run information
    configfile << "*********CONFIG FILE *********" << "\n" << "\n"
            << "*********Housekeeping Info*********" << "\n" << "\n"
            << "Date of experiement: " << date_as_string() << "\n" << "\n"
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
            << "\n Nervious System Parameters"
            << "N: " << N << "\n"
            << "WR: " << WR << "\n"
            << "SR: " << SR << "\n"
            << "BR: " << BR << "\n"
            << "TMIN: " << TMIN << "\n"
            << "TMAX: " << TMAX << "\n"
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
    #ifdef PRINTOFILE
        std::cout << "PRINTOFILE is ON\n";
    #else
        std::cout << "PRINTOFILE is OFF\n";
    #endif

    // check that argv[1] has been provided
    if (argc < 3){
        // send an error message to the terminal
        std::cerr << "Error: missing run or array index number.\n";
        return 1;
    }
    // Define output home directory
    std::string result_dir =  "/Users/katiepambakian/Documents/BSc Computer Science/Y3/Dissertation/BeeCommunication/results/";
    std::string dir = result_dir + date_as_string() +"/batch_"+ argv[2] +"/run_"+ argv[1] +"/";
    // there is not acutally an error here
    std::filesystem::create_directories(dir);

    // print to the config file
    output_config(dir, argv[1], argv[2]);

    std::string current_run = argv[1];
    // Random seed -> unique to each run
    long randomseed = static_cast<long>(time(NULL));
    randomseed += atoi(argv[1]);
    // save the seed to a file
    ofstream seedfile;
    seedfile.open (dir + "seed_" + current_run + ".dat");
    seedfile << randomseed << endl;
    seedfile.close();

    // if logging is enabled
    #ifdef PRINTOFILE
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

    // /* Initialize and seed the search */
    // search.InitializeSearch();
    
    // /* Evolve */
    // search.SetSearchTerminationFunction(TerminationFunction);
    // search.SetEvaluationFunction(FitnessFunction1);
    // search.ExecuteSearch();

    // search.SetSearchTerminationFunction(TerminationFunction);
    // search.SetEvaluationFunction(FitnessFunction2);
    // search.ExecuteSearch();

    // search.SetSearchTerminationFunction(TerminationFunction);
    // search.SetEvaluationFunction(FitnessFunction3);
    // search.ExecuteSearch();

    // search.SetSearchTerminationFunction(NULL);
    // search.SetEvaluationFunction(FitnessFunction4);
    // search.ExecuteSearch();

    if (search.BestPerformance() > 0.99) {
        RecordBehavior(search);
    }

    #ifdef PRINTTOFILE
        evolfile.close();
    #endif

    return 0;
}
