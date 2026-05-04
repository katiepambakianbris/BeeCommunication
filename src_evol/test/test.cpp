#include <iostream>
#include <cmath>
#include <cstdio>
#include <fstream>


#define PRINTTOFILE

double otherSensor=0.0;

double LANDMARKZONESTART = 10;
double LN = 3;

double SenseOther(int f, int h,double currentPos, double pos_other)
{
	double dist;
	dist = fabs(pos_other - currentPos);
	if (dist < 5)
	{
		return 1.0/(1.0 + exp(f*( (h *dist) - 1.0)));		
	}
    return 0.0;
    
}

// TVector<double> genLandmarks_Simple(double trial, TVector<double> &landmarkPositions){

//     double start = LANDMARKZONESTART +10; // 20
//     double spacing = 10; // space them 10 appart

//     for (int i = 1;i <= LN; i ++){
//         landmarkPositions[i] = start + (i-1)*(spacing+trial);
//     }

//     return landmarkPositions;
// }


// the purpose of this file is to investigate the impact on distance of the other agent on the agent
int main (int argc, const char* argv[]){

   double currentPos = 0;

    std::ofstream file("sensor_output.csv");
    file << "distance,sensor,f,h\n";

    std::vector<std::pair<int,int>> params = {
        {4,1}, {8,1}, {4,2}, {8,2}
    };

    for (auto [f, h] : params) {
        for (double dist = 0.0; dist <= 3.0; dist += 0.05) {
            double otherPos = currentPos + dist;
            double sensor = SenseOther(f, h, currentPos, otherPos);

            file << dist << "," << sensor << "," << f << "," << h << "\n";
        }
    }

    file.close();

    std::cout << "Data written to sensor_output.csv\n";
    return 0;

}