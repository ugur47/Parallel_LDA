#ifndef GSSDK_H
#define GSSDK_H
#include <vector>
using namespace std; 


#define  PI 3.1415926

unsigned getRandomValue(vector<double> &);
double sum(vector<double> &);
unsigned getRandomValue(unsigned );
void normalize(vector<double> &_p);
vector<unsigned> findValue(vector<double> &p, double v);
vector<double> subArray(const vector<double> &p, vector<unsigned> &id);

#endif