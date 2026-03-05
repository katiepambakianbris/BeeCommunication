#include <iostream>
#include <cmath>
#include <cstdio>
#include <fstream>

#define PRINTTOFILE

double otherSensor=0.0;

double SenseOther(double currentPos, double pos_other)
{
	double dist;
	dist = fabs(pos_other - currentPos);
	if (dist < 5)
	{
		return 1.0/(1.0 + exp(8.0 * (dist - 1.0)));		
	}
    return 0.0;
    
}


// the purpose of this file is to investigate the impact on distance of the other agent on the agent
int main (int argc, const char* argv[]){

    double currentPos = 0;

    std::ofstream file("sensor_output.csv");
    file << "distance,sensor\n";

    for (double dist = 0.0; dist <= 3.0; dist += 0.05){
        double otherPos = currentPos + dist;
        double sensor = SenseOther(currentPos, otherPos);

        file << dist << "," << sensor << "\n";
    
    }
    file.close();

    std::cout << "Data written to sensor_output.csv\n";

    return 0;

}