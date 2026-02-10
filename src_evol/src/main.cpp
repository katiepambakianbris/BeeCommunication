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
int VectSize = 2 * (N*N + 5*N);  // Double the amount of parameters, one for Receiver, one for Signaler

// Environment variables
// Signaller Start Zone
const int SIGNALLERSTART = -5;
const int SIGNALLEREND = -1;
// Reciver start location
const int RECEIVERSTART = 0;
// landmark zone
const int LANDMARKZONESTART = 5;
const int LANDMARKZONEEND = 15;

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
    int lenLandmarkZone = LANDMARKZONEEND - LANDMARKZONESTART;
    int max = lenLandmarkZone / N;

    int y = rs.UniformRandom(0,max);
    int x0 = rs.UniformRandom(0,y) + LANDMARKZONESTART;

    landmarkPositions[1] = x0;

    if (LN >1){
        // generate the offset J
        int j = rs.UniformRandom(0,max);

        for (int i = 2;i <= LN; i ++){
            landmarkPositions[i] = x0 + ((i-1)*j);
        }
    }

    // check that the last landmark is not out of range
    assert(landmarkPositions[LN] <= LANDMARKZONEEND);
   
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

// ---------------------------------------------------------
// stage 1: Evolution of receiver
// only looking at the distance between the receiver and the 
// flower/food
// no signaller in the task
// the sensor of the receiver is set to correspond to which
// of the landmarks the receiver should go to
// ---------------------------------------------------------

double stage1(TVector<double> &genotype, RandomState &rs){

    // trial variables 
    double food_location;
    double totaltrials = 0;
    double totaltime;
    double total_score_receiver = 0;
    double distance_food_receiver;

    // **** create the receiver ****

    // Map genotype to phenotype
    TVector<double> phenotypeReceiver;
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    // N*N +5N is the size of one agents parameter set
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // create vectors to use to save their state
    TVector<double> savedStateReceiver, savedStateSignaller;

    // **** Generate landmarks (using leapfrog method) ****
    TVector<double> landmarkPositions;
    genLandmarks_LeapFrog(rs, landmarkPositions);

    // set each of the landmarks to be the location of the food
    for (int env =1; env<=LN;env++){

        // set the food to be at the ith landmark location
        food_location = landmarkPositions[env];

        // Initialise the agent for this trial
        AgentReceiver.SetPosition(0);
        AgentReceiver.SetOther(env);

        // Let the Receiver and Signaller Explore the enviornment (for 300) -> not scored
        for (double time=0; time < RunDuration; time += StepSize){
            AgentReceiver.SenseFood(food_location);
            AgentReceiver.SenseLandmarks(LN, landmarkPositions);
            AgentReceiver.Step(StepSize);
        }
        AgentReceiver.SetPosition(0);

        // now we are scoring but in EASY mode
        // for (double time=0; time < RunDuration; time += StepSize){
        //     AgentReceiver.SenseFood(food_location);
        //     AgentReceiver.SenseLandmarks(LN, landmarkPositions);
        //     AgentReceiver.Step(StepSize);

        //     // get the absolute distance to the food
        //     distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);
        //     total_score_receiver += easy_score(distance_food_receiver);
        //     totaltrials ++;
        // }
        // AgentReceiver.SetPosition(0);

        // now we are scoring but in HARD mode
        for (double time=0; time < RunDuration; time += StepSize){
            AgentReceiver.SenseFood(food_location);
            AgentReceiver.SenseLandmarks(LN, landmarkPositions);
            AgentReceiver.Step(StepSize);

            // get the absolute distance to the food
            distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);
            total_score_receiver += hard_score(distance_food_receiver);
            totaltrials ++;
        }
    }
    return total_score_receiver / totaltrials;
}

// ---------------------------------------------------------
// stage 2: Evolution of signaller
// only looking a the distance between the Signaller and the
// receiver (if that is equal to the distance of the landmark)
// ---------------------------------------------------------

double stage2(TVector<double> &genotype, RandomState &rs){
    // trial variables 
    double food_location;
    double totaltrials = 0;
    double totaltime;
    double total_score_receiver = 0;
    double distance_to_receiver;

    // **** Initalize the receiver and signaller ****

    // Map genotype to phenotype
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    // N*N +5N is the size of one agents parameter set
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));

    CountingAgent AgentSignaler(N, phenotypeSignaller);
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // create vectors to use to save their state
    TVector<double> savedStateReceiver, savedStateSignaller;

    // **** Generate landmarks (using leapfrog method) ****
    TVector<double> landmarkPositions;
    genLandmarks_LeapFrog(rs, landmarkPositions);

    // set each of the landmarks to be the location of the food
    for (int env =1; env<=LN;env++){
        // establish food location
        food_location = landmarkPositions[env];

        // **** Initialise the agents for this trial ****

        // put the signaller in a random location (in the signaller box)
        int location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
        AgentSignaler.SetPosition(location);

        // put the agent in its home position
        AgentReceiver.SetPosition(0);

        // Let the Signaller Explore the enviornment (for 300) -> not scored (find the food)
        for (double time=0; time < RunDuration; time += StepSize){
            AgentSignaler.SenseFood(food_location);
            AgentSignaler.SenseLandmarks(LN, landmarkPositions);
            AgentSignaler.Step(StepSize);
        }

        AgentSignaler.SetPosition(location);
        AgentReceiver.SetPosition(0);

        // Let the Signaller Explore the other agent (for 300) -> not scored (find his friend)
        for (double time=0; time < RunDuration; time += StepSize){
            AgentSignaler.SenseFood(food_location);
            AgentSignaler.SenseLandmarks(LN, landmarkPositions);
            AgentSignaler.SenseOther(AgentReceiver.pos);
            AgentSignaler.Step(StepSize);
        }

        AgentSignaler.SetPosition(location);
        AgentReceiver.SetPosition(0);

        // now it needs to go to the right distance from the reciever
        // now we are scoring but in EASY mode
        // for (double time=0; time < RunDuration; time += StepSize){
        //     AgentSignaler.SenseFood(food_location);
        //     AgentSignaler.SenseLandmarks(LN, landmarkPositions);
        //     AgentSignaler.SenseOther(AgentReceiver.pos);
        //     AgentSignaler.Step(StepSize);

        //     // get the absolute distance to the food
        //     distance_to_receiver = fabs(AgentSignaler.GetPosition() - AgentReceiver.GetPosition());
        //     total_score_receiver += easy_score(distance_to_receiver);
        //     totaltrials ++;
        // }
        // AgentSignaler.SetPosition(location);
        // AgentReceiver.SetPosition(0);

        // now we are scoring but in HARD mode
        for (double time=0; time < RunDuration; time += StepSize){
            AgentSignaler.SenseFood(food_location);
            AgentSignaler.SenseLandmarks(LN, landmarkPositions);
            AgentSignaler.SenseOther(AgentReceiver.pos);
            AgentSignaler.Step(StepSize);

            // get the absolute distance to the food
            distance_to_receiver = fabs(AgentSignaler.GetPosition() - AgentReceiver.GetPosition());

            total_score_receiver += hard_score(distance_to_receiver);
            totaltrials ++;
        }
    }
    return total_score_receiver / totaltrials;
}

// ---------------------------------------------------------
// stage 3: The Joint Task
// looking at the distance between the receiver and the food
// ---------------------------------------------------------

double stage3(TVector<double> &genotype, RandomState &rs){

    // trial variables 
    double food_location;
    double totaltrials = 0;
    double totaltime;
    double total_score_receiver = 0;
    double distance_food_receiver;

    // **** Initalize the receiver and signaller ****

    // Map genotype to phenotype
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    // N*N +5N is the size of one agents parameter set
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));

    CountingAgent AgentSignaler(N, phenotypeSignaller);
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // create vectors to use to save their state
    TVector<double> savedStateReceiver, savedStateSignaller;

    // Generate landmarks (using leapfrog method)
    TVector<double> landmarkPositions;
    genLandmarks_LeapFrog(rs, landmarkPositions);

    // set each of the landmarks to be the location of the food
    for (int env =1; env<=LN;env++){
        // establish food location
        food_location = landmarkPositions[env];

        // **** Initialise the agents for this trial ****
        // put the signaller in a random location
        int location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
        AgentSignaler.SetPosition(location);
        // put the agent in its home position
        AgentReceiver.SetPosition(0);

        // Let the Signaller and Receiver Explore the enviornment (for 300) -> not scored (find the food)
        for (double time=0; time < RunDuration; time += StepSize){
            AgentSignaler.SenseFood(food_location);
            AgentSignaler.SenseLandmarks(LN, landmarkPositions);
            AgentReceiver.SenseFood(food_location);
            AgentReceiver.SenseLandmarks(LN, landmarkPositions);
            AgentSignaler.Step(StepSize);
            AgentReceiver.Step(StepSize);
        }
        AgentSignaler.SetPosition(location);
        AgentReceiver.SetPosition(0);

        // now it needs to go to the right distance from the reciever
        // now we are scoring but in EASY mode
        // for (double time=0; time < RunDuration; time += StepSize){
        //     AgentSignaler.SenseFood(food_location);
        //     AgentSignaler.SenseLandmarks(LN, landmarkPositions);
        //     AgentSignaler.SenseOther(AgentReceiver.pos);
        //     AgentReceiver.SenseOther(AgentSignaler.pos);
        //     AgentReceiver.SenseFood(food_location);
        //     AgentReceiver.SenseLandmarks(LN, landmarkPositions);

        //     AgentSignaler.Step(StepSize);
        //     AgentReceiver.Step(StepSize);

        //     // get the absolute distance to the food
        //     distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);
        //     total_score_receiver += easy_score(distance_food_receiver);
        //     totaltrials ++;
        // }

        // AgentSignaler.SetPosition(location);
        // AgentReceiver.SetPosition(0);

        // now we are scoring but in HARD mode
        for (double time=0; time < RunDuration; time += StepSize){
            AgentSignaler.SenseFood(food_location);
            AgentSignaler.SenseLandmarks(LN, landmarkPositions);
            AgentSignaler.SenseOther(AgentReceiver.pos);
            AgentReceiver.SenseOther(AgentSignaler.pos);
            AgentReceiver.SenseFood(food_location);
            AgentReceiver.SenseLandmarks(LN, landmarkPositions);

            AgentSignaler.Step(StepSize);
            AgentReceiver.Step(StepSize);

            // get the absolute distance to the food
            distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);
            total_score_receiver += hard_score(distance_food_receiver);
            totaltrials ++;
        }
    }
    return total_score_receiver / totaltrials;
}


// ------------------------------------------
// For the Best Genotype record the behaviour
// using the same steps as in stage 3 -> but i think i want to do all the tasks so edit this
// ------------------------------------------
double RecordBehavior(TSearch &s, RandomState &rs) {
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();
    
    // Map genotype to phenotype
    TVector<double> genotype;
    genotype = s.BestIndividual();

    // trial variables 
    double food_location;
    double totaltrials = 0;
    double totaltime;
    double total_score_receiver = 0;
    double distance_food_receiver;

    // **** Initalize the receiver and signaller ****

    // Map genotype to phenotype
    TVector<double> phenotypeReceiver, phenotypeSignaller;
    phenotypeReceiver.SetBounds(1, (int)(VectSize/2));
    phenotypeSignaller.SetBounds(1, (int)(VectSize/2));
    GenPhenMapping(genotype, phenotypeSignaller, 1);
    // N*N +5N is the size of one agents parameter set
    GenPhenMapping(genotype, phenotypeReceiver, (int)(N*N + 5*N +1));

    CountingAgent AgentSignaler(N, phenotypeSignaller);
    CountingAgent AgentReceiver(N, phenotypeReceiver);

    // create vectors to use to save their state
    TVector<double> savedStateReceiver, savedStateSignaller;

    // Generate landmarks (using leapfrog method)
    TVector<double> landmarkPositions;

    // **** create the files to save the behaviour/env ***
    // storing them all in the same file, each gets a new line (so 9 trials per file)
    // the stages are separated into three files
    ofstream SignalerBehaviorFile1, ReceiverBehaviorFile1, SignalerBehaviorFile2, ReceiverBehaviorFile2,SignalerBehaviorFile3, ReceiverBehaviorFile3;
    SignalerBehaviorFile1.open( dir + "behavior_Signaler_stage1_" + current_run);
    ReceiverBehaviorFile1.open( dir + "behavior_Receiver_stage1_" + current_run);
    SignalerBehaviorFile2.open( dir + "behavior_Signaler_stage2_" + current_run);
    ReceiverBehaviorFile2.open( dir + "behavior_Receiver_stage2_" + current_run);
    SignalerBehaviorFile3.open( dir + "behavior_Signaler_stage3_" + current_run);
    ReceiverBehaviorFile3.open( dir + "behavior_Receiver_stage3_" + current_run);
    
    // stores the location of the landmarks and the food -> this is each line 
    ofstream LandmarkFile;
    LandmarkFile.open(dir + "landmark_location_stage1_"+current_run);

    // set each of the landmarks to be the location of the food
    for (int env =1; env<=LN;env++){
        // test each for 3 positions of the landmarks
        for (int i=0; i<3; i++){
            // env setup
            genLandmarks_LeapFrog(rs, landmarkPositions);
             // establish food location
            food_location = landmarkPositions[env];
            // write the landmarks to the output
            for (int i = 1; i <= LN; i += 1){
                LandmarkFile << landmarkPositions[i] << " ";
            }
            LandmarkFile << food_location << " ";
        
            // Stage 1: 
            AgentReceiver.SetPosition(0);
            AgentReceiver.SetOther(env);
            SignalerBehaviorFile1 << AgentSignaler.GetPosition() << " ";
            ReceiverBehaviorFile1 << AgentReceiver.GetPosition() << " ";

            // Let the Receiver and Signaller Explore the enviornment (for 300) -> not scored
            for (double time=0; time < RunDuration; time += StepSize){
                AgentReceiver.SenseFood(food_location);
                AgentReceiver.SenseLandmarks(LN, landmarkPositions);
                AgentReceiver.Step(StepSize);
                SignalerBehaviorFile1 << AgentSignaler.GetPosition() << " ";
                ReceiverBehaviorFile1 << AgentReceiver.GetPosition() << " ";
            }
            AgentReceiver.SetPosition(0);

            // now we are scoring but in EASY mode
            // for (double time=0; time < RunDuration; time += StepSize){
            //     AgentReceiver.SenseFood(food_location);
            //     AgentReceiver.SenseLandmarks(LN, landmarkPositions);
            //     AgentReceiver.Step(StepSize);
            //     SignalerBehaviorFile1 << AgentSignaler.GetPosition() << " ";
            //     ReceiverBehaviorFile1 << AgentReceiver.GetPosition() << " ";

            //     // get the absolute distance to the food
            //     distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);
            //     total_score_receiver += easy_score(distance_food_receiver);
            //     totaltrials ++;
            // }
            // AgentReceiver.SetPosition(0);

            // now we are scoring but in HARD mode
            for (double time=0; time < RunDuration; time += StepSize){
                AgentReceiver.SenseFood(food_location);
                AgentReceiver.SenseLandmarks(LN, landmarkPositions);
                AgentReceiver.Step(StepSize);
                SignalerBehaviorFile1 << AgentSignaler.GetPosition() << " ";
                ReceiverBehaviorFile1 << AgentReceiver.GetPosition() << " ";

                // get the absolute distance to the food
                distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);
                total_score_receiver += hard_score(distance_food_receiver);
                totaltrials ++;
            }


            // Stage 2
            int distance_to_receiver = 0;

            // put the signaller in a random location (in the signaller box)
            int location = rs.UniformRandom(SIGNALLERSTART,SIGNALLEREND);
            AgentSignaler.SetPosition(location);
            // put the agent in its home position
            AgentReceiver.SetPosition(0);
            SignalerBehaviorFile2 << AgentSignaler.GetPosition() << " ";
            ReceiverBehaviorFile2 << AgentReceiver.GetPosition() << " ";

            // Let the Signaller Explore the enviornment (for 300) -> not scored (find the food)
            for (double time=0; time < RunDuration; time += StepSize){
                AgentSignaler.SenseFood(food_location);
                AgentSignaler.SenseLandmarks(LN, landmarkPositions);
                AgentSignaler.Step(StepSize);
                SignalerBehaviorFile2 << AgentSignaler.GetPosition() << " ";
                ReceiverBehaviorFile2 << AgentReceiver.GetPosition() << " ";
            }

            AgentSignaler.SetPosition(location);
            AgentReceiver.SetPosition(0);

            // Let the Signaller Explore the other agent (for 300) -> not scored (find his friend)
            for (double time=0; time < RunDuration; time += StepSize){
                AgentSignaler.SenseFood(food_location);
                AgentSignaler.SenseLandmarks(LN, landmarkPositions);
                AgentSignaler.SenseOther(AgentReceiver.pos);
                AgentSignaler.Step(StepSize);
                SignalerBehaviorFile2 << AgentSignaler.GetPosition() << " ";
                ReceiverBehaviorFile2 << AgentReceiver.GetPosition() << " ";
            }

            AgentSignaler.SetPosition(location);
            AgentReceiver.SetPosition(0);

            // now it needs to go to the right distance from the reciever
            // now we are scoring but in EASY mode
            // for (double time=0; time < RunDuration; time += StepSize){
            //     AgentSignaler.SenseFood(food_location);
            //     AgentSignaler.SenseLandmarks(LN, landmarkPositions);
            //     AgentSignaler.SenseOther(AgentReceiver.pos);
            //     AgentSignaler.Step(StepSize);
            //     SignalerBehaviorFile2 << AgentSignaler.GetPosition() << " ";
            //     ReceiverBehaviorFile2 << AgentReceiver.GetPosition() << " ";

            //     // get the absolute distance to the food
            //     distance_to_receiver = fabs(AgentSignaler.GetPosition() - AgentReceiver.GetPosition());
            //     total_score_receiver += easy_score(distance_to_receiver);
            //     totaltrials ++;
            // }
            // AgentSignaler.SetPosition(location);
            // AgentReceiver.SetPosition(0);

            // now we are scoring but in HARD mode
            for (double time=0; time < RunDuration; time += StepSize){
                AgentSignaler.SenseFood(food_location);
                AgentSignaler.SenseLandmarks(LN, landmarkPositions);
                AgentSignaler.SenseOther(AgentReceiver.pos);
                AgentSignaler.Step(StepSize);

                SignalerBehaviorFile2 << AgentSignaler.GetPosition() << " ";
                ReceiverBehaviorFile2 << AgentReceiver.GetPosition() << " ";

                // get the absolute distance to the food
                distance_to_receiver = fabs(AgentSignaler.GetPosition() - AgentReceiver.GetPosition());

                total_score_receiver += hard_score(distance_to_receiver);
                totaltrials ++;
            }

            // stage 3

            // **** Initialise the agents for this trial ****
            // put the signaller in a random location (same as previous)
            AgentSignaler.SetPosition(location);
            // put the agent in its home position
            AgentReceiver.SetPosition(0);

            // **** record the initial setup in the file
            SignalerBehaviorFile3 << AgentSignaler.GetPosition() << " ";
            ReceiverBehaviorFile3 << AgentReceiver.GetPosition() << " ";

            // Let the Signaller and Receiver Explore the enviornment (for 300) -> not scored (find the food)
            for (double time=0; time < RunDuration; time += StepSize){
                AgentSignaler.SenseFood(food_location);
                AgentSignaler.SenseLandmarks(LN, landmarkPositions);
                AgentReceiver.SenseFood(food_location);
                AgentReceiver.SenseLandmarks(LN, landmarkPositions);
                AgentSignaler.Step(StepSize);
                AgentReceiver.Step(StepSize);

                SignalerBehaviorFile3 << AgentSignaler.GetPosition() << " ";
                ReceiverBehaviorFile3 << AgentSignaler.GetPosition()  << " ";
            }
            AgentSignaler.SetPosition(location);
            AgentReceiver.SetPosition(0);

            SignalerBehaviorFile3 << AgentSignaler.GetPosition() << " ";
            ReceiverBehaviorFile3 << AgentReceiver.GetPosition() << " ";

            // now it needs to go to the right distance from the reciever
            // now we are scoring but in EASY mode
            // for (double time=0; time < RunDuration; time += StepSize){
            //     AgentSignaler.SenseFood(food_location);
            //     AgentSignaler.SenseLandmarks(LN, landmarkPositions);
            //     AgentSignaler.SenseOther(AgentReceiver.pos);
            //     AgentReceiver.SenseOther(AgentSignaler.pos);
            //     AgentReceiver.SenseFood(food_location);
            //     AgentReceiver.SenseLandmarks(LN, landmarkPositions);

            //     AgentSignaler.Step(StepSize);
            //     AgentReceiver.Step(StepSize);

            //     SignalerBehaviorFile3 << AgentSignaler.GetPosition() << " ";
            //     ReceiverBehaviorFile3 << AgentReceiver.GetPosition() << " ";

            //     // get the absolute distance to the food
            //     distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);
            //     total_score_receiver += easy_score(distance_food_receiver);
            //     totaltrials ++;
            // }

            // AgentSignaler.SetPosition(location);
            // AgentReceiver.SetPosition(0);

            // now we are scoring but in HARD mode
            for (double time=0; time < RunDuration; time += StepSize){
                AgentSignaler.SenseFood(food_location);
                AgentSignaler.SenseLandmarks(LN, landmarkPositions);
                AgentSignaler.SenseOther(AgentReceiver.pos);
                AgentReceiver.SenseOther(AgentSignaler.pos);
                AgentReceiver.SenseFood(food_location);
                AgentReceiver.SenseLandmarks(LN, landmarkPositions);

                AgentSignaler.Step(StepSize);
                AgentReceiver.Step(StepSize);

                SignalerBehaviorFile3 << AgentSignaler.GetPosition() << " ";
                ReceiverBehaviorFile3 << AgentReceiver.GetPosition() << " ";

                // get the absolute distance to the food
                distance_food_receiver = fabs(AgentReceiver.GetPosition() - food_location);
                total_score_receiver += hard_score(distance_food_receiver);
                totaltrials ++;
            }

            // new line for a new trial
            SignalerBehaviorFile1 << endl;
            ReceiverBehaviorFile1 << endl;
            SignalerBehaviorFile2 << endl;
            ReceiverBehaviorFile2 << endl;
            SignalerBehaviorFile3 << endl;
            ReceiverBehaviorFile3 << endl;

            LandmarkFile << endl;
        }
    }
    // close the files for that env
    SignalerBehaviorFile1.close();
    ReceiverBehaviorFile1.close();
    SignalerBehaviorFile2.close();
    ReceiverBehaviorFile2.close();
    SignalerBehaviorFile3.close();
    ReceiverBehaviorFile3.close();
    LandmarkFile.close();

    return 1;
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
            // << "\n*********Landmark Parameters*********\n"
            // << "Sep " << SEP << "\n"
            // << "Ref " << REF << "\n"
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
    std::string result_dir = "/user/home/yj23812/BeeCommunication/results/"+ date_as_string() +"/" + slurm_job_id;
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


    // Stage 2: Evolution of Signaller
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(stage2);
    search.ExecuteSearch();

    // Stage 3: Join task
    search.SetSearchTerminationFunction(TerminationFunction);
    search.SetEvaluationFunction(stage3);
    search.ExecuteSearch();

    // TODO: Stage 4: Testing


    if (search.BestPerformance() > 0.99) {
        RecordBehavior(search, search.getRandomState());
    } else if (search.BestPerformance() < 0.1){
        RecordBehavior(search, search.getRandomState());
    }

    #ifdef PRINTTOFILE
        file.close();
    #endif

    return 0;
}
