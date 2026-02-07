

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
                    landmarkPositionTest[i] = (REF + ref_var) + (i * (SEP + sep_var));
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

    // Landmarks and variations
    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1)
    {
        landmarkPositions[i] = REF + (i * SEP);
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
                    landmarkPositionTest[i] = (REF + ref_var) + (i * (SEP + sep_var));
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

    // Landmarks and variations
    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1)
    {
        landmarkPositions[i] = REF + (i * SEP);
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
                    landmarkPositionTest[i] = (REF + ref_var) + (i * (SEP + sep_var));
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


    // Landmarks and variations
    TVector<double> landmarkPositions;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    for (int i = 1; i <= LN; i += 1)
    {
        landmarkPositions[i] = REF + (i * SEP);
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
                    landmarkPositionTest[i] = (REF + ref_var) + (i * (SEP + sep_var));
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