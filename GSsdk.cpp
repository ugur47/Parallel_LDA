#include "GSsdk.h"
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;


double sum(vector<double> &_p){
	double retVal = 0; 
	for (unsigned i = 0; i < _p.size(); ++i)
	{
		retVal += _p[i];
	}
	return retVal;
}


unsigned getRandomValue(vector<double> &_p){	
	unsigned _range = _p.size();
	double temp = sum(_p)*((double)rand())/RAND_MAX;
	for (unsigned i = 0 ; i < _range ;++i)
	{
		if ((temp=temp-_p[i]) < 0 )
		{
			return i;
		}
	}
	return _range-1;
}

unsigned getRandomValue(unsigned _range){	
	double temp = ((double)rand())/RAND_MAX;
	double p = ((double)1)/_range;
	for (unsigned i = 0 ; i < _range ;++i)
	{
		if ((temp=temp-p) < 0 )
		{
			return i;
		}
	}
	return _range-1;
}


void normalize(vector<double> &_p){
	double s = sum(_p);
	for (unsigned i = 0 ; i < _p.size() ; ++i)
	{
		_p[i] = _p[i]/s;
	}	
}


/*returns all elements of p, whose value is in good proximate of v*/
vector<unsigned> findValue(vector<double> &p, double v){
	vector<unsigned> retVal;

	for (unsigned i = 0 ; i < p.size() ; ++i)
	{
		if ((v-p[i])>-0.001&&(v-p[i])<0.001)
		{
			retVal.push_back(i);
		}
	}
	return retVal;
}


/*returns sub array of p, whose index is given by id*/
vector<double> subArray(const vector<double> &p, vector<unsigned> &id){
	vector<double> retVal;
	for ( unsigned i = 0 ; i < id.size() ; i++)
	{
		retVal.push_back(p[id[i]]);
	}
	return retVal;
}