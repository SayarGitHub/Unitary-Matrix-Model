#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define srand48(x) srand((int)(x))
#define drand48() ((double)rand() / RAND_MAX)

using namespace std;

complex<double> I(0.0, 1.0);

int N = 10;
double EPS = 0.5;
int GAP = 100;

double getAction(complex<double> theta[], double g)
{
    double lambda = g * g * N;
    complex<double> action(0.0, 0.0);
    for (int i = 0; i < N; i++)
    {
        complex<double> complexVND(0.0, 0.0);
        for (int j = 0; j < N; j++)
        {
            if (i != j)
                complexVND += log(sin(0.5 * abs((theta[i] - theta[j]))));
        }
        action += -(exp(I * theta[i]) + exp(-I * theta[i])) * (N / lambda) - complexVND;
    }
    return real(action);
}

double *MCStep(complex<double> thetaOld[], complex<double> thetaNew[], int iterations, double g, bool isTherm, ofstream &f_a_rate)
{

    double oldAction;
    double newAction;
    int n = 0;
    double *result = new double[3];
    result[0] = 0.0;
    result[1] = 0.0;
    result[2] = 0.0;
    double p = 0;
    double pSq = 0.0;
    double pError = 0.0;
    int accept = 0;
    int no_calls = 0;
    double lambda = g * g * N;

    for (int i = 0; i < iterations; i++)
    {
        no_calls++;
        oldAction = getAction(thetaOld, g);
        int changeIndex = (int)N * drand48();
        complex<double> noise(2 * (drand48() - 0.5), 0.0);
        thetaNew[changeIndex] = thetaOld[changeIndex] + EPS * noise;
        newAction = getAction(thetaNew, g);
        double deltaS = newAction - oldAction;
        double u = drand48();

        if (exp(-deltaS) >= u || deltaS < 0)
        {
            accept++;
            thetaOld[changeIndex] = thetaNew[changeIndex];
            // cout<<"Accept"<<endl;
        }
        else
        {
            thetaNew[changeIndex] = thetaOld[changeIndex];
            newAction = oldAction;
            // cout<<"Reject"<<endl;
        }

        //Acceptance rate

        if (!isTherm && no_calls % 100 == 0 && no_calls > 1)
        {

            f_a_rate << ((double)accept / double(no_calls)) * 100 << endl;
            accept = 0;
            no_calls = 0;
        }

        if (!isTherm && i % GAP == 0)
        {
            n++;
            for (int k = 0; k < N; k++)
            {
                double temp = real(exp(I * thetaNew[k])) / N;
                p += temp;
                pSq += temp * temp;
            }
        }
    }

    if (!isTherm)
    {
        p /= n;
        pSq /= n;
        pError = sqrt(abs(pSq - p * p) / (double)n);
        result[0] = p;
        result[1] = pError;
        result[2] = (g < sqrt(2 / (double)N)) ? 1 - lambda / 4 : 1 / lambda;
        return result;
    }
    return result;
}

void write(vector<double> v, string filename)
{
    ofstream file;
    file.open(filename);

    for (int i = 0; i < v.size(); i++)
    {
        file << v[i] << endl;
    }
    file.close();
}

int main()
{

    srand48(time(NULL));
    int therm = 1e3;
    int sweeps = 1e4;
    int countMax = 100;
    vector<double> gVector(countMax);
    vector<double> pVector(countMax);
    vector<double> pErrorVector(countMax);
    vector<double> pExpectedVector(countMax);
    double *result;
    double g = 0.01;
    static ofstream f_a_rate;
    f_a_rate.open("Acceptance_rate.txt");

    for (int count = 0; count < countMax; count++)
    {
        complex<double> thetaOld[N];
        complex<double> thetaNew[N];
        for (int i = 0; i < N; i++)
        {
            thetaOld[i].real(0.01 * drand48());
            thetaOld[i].imag(0.01 * drand48());
            thetaNew[i] = thetaOld[i];
        }
        // Thermalizing

        MCStep(thetaOld, thetaNew, therm, g, true, f_a_rate);

        // MC Generation

        result = MCStep(thetaOld, thetaNew, sweeps, g, false, f_a_rate);
        gVector.at(count) = g;
        pVector.at(count) = result[0];
        pErrorVector.at(count) = result[1];
        pExpectedVector.at(count) = result[2];
        cout << "Done with g=" << g << endl;
        g += 0.05;
    }

    write(gVector, "g_values.txt");
    write(pVector, "p_avg_values.txt");
    write(pErrorVector, "p_err_values.txt");
    write(pExpectedVector, "p_expec_values.txt");

    f_a_rate.close();

    return 0;
}