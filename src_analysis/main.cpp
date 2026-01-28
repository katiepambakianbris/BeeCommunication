#include <iostream>
#include "TSearch.h"
#include "CountingAgent.h"
#include "CTRNN.h"
#include "random.h"

#define PRINTOFILE

// Task params
const int LN = 2;                   // Number of landmarks in the environment
const double StepSize = 0.1;
const double RunDuration = 300.0;
const double TransDuration = 150.0;
const double MinLength = 50.0;      
const double mindist = 5.0;         

// EA params
const int POPSIZE = 96;
const int GENS = 1000; //10000;
const double MUTVAR = 0.05;
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
// Behavior
// ------------------------------------
double RecordBehavior(TVector<double> genotype, std::string current_run) //, RandomState &rs)
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
    float totaltrials = 0;
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
        SignalerBehaviorFile1.open( "behavior_Signaler_" + current_run + "_Env" + s_env + "_Phase1.dat");
        ReceiverBehaviorFile1.open( "behavior_Receiver_" + current_run + "_Env" + s_env + "_Phase1.dat");
        ofstream SignalerBehaviorFile2, ReceiverBehaviorFile2;
        SignalerBehaviorFile2.open( "behavior_Signaler_" + current_run + "_Env" + s_env + "_Phase2.dat");
        ReceiverBehaviorFile2.open( "behavior_Receiver_" + current_run + "_Env" + s_env + "_Phase2.dat");
        ofstream SignalerBehaviorFile3, ReceiverBehaviorFile3;
        SignalerBehaviorFile3.open( "behavior_Signaler_" + current_run + "_Env" + s_env + "_Phase3.dat");
        ReceiverBehaviorFile3.open( "behavior_Receiver_" + current_run + "_Env" + s_env + "_Phase3.dat");

        // TEST USING DIFFERENT DISTANCES BETWEEN LANDMARKS
        for (double ref_var = -5.0; ref_var <= 5.0; ref_var += 2.0)
        {
            for (double sep_var = -5.0; sep_var <= 5.0; sep_var += 2.0)
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

// ------------------------------------
// The main program
// ------------------------------------
int main (int argc, const char* argv[])
{
    std::string current_run = argv[1];
    ifstream genefile; // opens the file
    genefile.open("best_gen_" + current_run + ".dat");
    TVector<double> genotype(1, VectSize);
    genefile >> genotype;
    genefile.close();
    RecordBehavior(genotype, current_run);
    return 0;
}
