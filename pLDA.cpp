#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <time.h>

#include <random>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include "GSsdk.h"
#include <cmath>


double get_CPU_time_usage(clock_t clock1,clock_t clock2)
{
    double diffticks=clock1-clock2;
    double diffms=(diffticks*1000)/CLOCKS_PER_SEC;
    return diffms;
} 

double calculate(vector<vector<int> > &Cwt, vector<int> &Ct, vector<vector<int> > &Ctd, vector<int> &Cd,const vector<double> &alpha,  const vector<double> &beta,
		 vector<vector<double> > &theta, vector<vector<double> > &phi, const vector<int> &did, const vector<int> &wid,int T, int W, int D, int N, int P,  vector<vector<int> > &DocumentIndex) {
    cout << "Calculating Ph&Theta." << endl;
    double temp1 = (beta[0] * W);
    double temp2 = (alpha[0] * T);
    int i,j,k;
    double t1,t2,temp, temp4;
    //calculate phi values
    #pragma omp parallel for num_threads(P) shared(Cwt,Ct,temp1,beta,W,T,phi) private(t1,t2,j) schedule(static)
    for (i = 0; i < W; i++)
    {
        for (j = 0; j < T; j++)
        {
            t1 = (Cwt[i][j] + beta[i]);
            t2 = (temp1 + Ct[j]);
            phi[i][j] = t1 / t2;
        }
    }
    
    #pragma omp barrier
    
    //calculate theta values
    #pragma omp parallel for num_threads(P) shared(Cd,Ctd,temp2,alpha,T,D,theta) private(t1,t2,j) schedule(static)     
    for (i = 0; i < T; i++)
    {
        for (j = 0; j < D; j++)
        {
            t1 = (Ctd[i][j] + alpha[i]);
            t2 = ( temp2 + Cd[j]);
            theta[i][j] = t1 / t2;
        }
    }
    cout << "Calculating Perp" << endl;
    //calculate perplexity
    double perp = 0;
#pragma omp parallel for num_threads(P) shared(W,T,D,N,phi,theta,DocumentIndex) private(temp,temp4,j,k) schedule(static) reduction(+:perp)
    for (i = 0; i < D; i++)
    {
        temp4 = 0;
        for (j = 0; j < DocumentIndex[i].size(); j++)
        {
                temp = 0;
                for (k = 0; k < T; k++)
                {
                    temp = temp + (phi[wid[DocumentIndex[i][j]]][k] * theta[k][i]);
                }
                temp4 = temp4 + log(temp);
        }
        perp += temp4;
    }

    perp = -perp/N;
    return exp(perp);
}

void gs( const vector<int> &wid, const vector<int> &did, const vector<double> &alpha,  const vector<double> &beta, 
		vector<int> &z, vector<vector<int> > &Ctd, vector<vector<int> > &Cwt, vector<int> &Ct, vector<int> &Cd, vector<vector<double> > &theta, vector<vector<double> > &phi,
	 vector<double> &perplexity, int numberOfiteration, int N, int P, int T, int W, int D, vector<vector<int> > &DocumentIndex,vector<double> &CPUtimes){ 
                clock_t begin=clock();
		double t5 = (W*beta[0]);
		double t6 = (((double)N)/P);
                int cPU = 0;
		for( int iter = 0; iter < numberOfiteration; iter++){ 
			cout << "iter:" << iter << endl;
			int p, n, t;
			double t1,t2,t3,t4;
			vector<double> Prob(T);
			omp_set_num_threads(P);
			cout << "Thread Count: " << P << endl;			
			#pragma omp parallel sections num_threads(P) shared(wid,did,z,Ctd,Cwt,Ct, t5, t6) private(p,n,t,Prob,t1,t2,t3,t4) 
			{
				cout << "active:" << omp_get_num_threads() << endl;	
				#pragma omp section 
				{
					int p = omp_get_thread_num();
					//cout << "p:" << p <<endl;
					//make division float
					int t1 = (int)( t6*(p+1) );
					for( int n = (int)( t6*p ); n < t1; n++) { 
						int t = z[n];
						Ctd[t][did[n]]--; Cwt[wid[n]][t]--; Ct[t]--;
						vector<double> Prob(T);
						for (t = 0; t < T; t++) {
							double t2 = (Cwt[wid[n]][t]+ beta[wid[n]]);
							double t3 = (Ctd[t][did[n]] + alpha[t]);
							double t4 = (Ct[t] + t5);
							Prob[t] = (t2 * t3) / t4;
						}
						t = getRandomValue(Prob);
						z[n] = t;
						Ctd[t][did[n]]++; Cwt[wid[n]][t]++; Ct[t]++;
					}
					
				}
			}
			//perplexity[iter] = calculate(Cwt, Ct, Ctd, Cd, alpha,  beta, theta, phi, did, wid, T, W, D, N, P, DocumentIndex);
			#pragma omp barrier
			if (iter % 10 == 9) {
			  //perplexity[cPU] = calculate(Cwt, Ct, Ctd, Cd, alpha,  beta, theta, phi, did, wid, T, W, D, N, P, DocumentIndex);
			  clock_t end=clock();
                                //CPUtimes[cPU] = double(get_CPU_time_usage(end,begin))/1000;
			        cPU++;
				cout << "Synchonize" << endl;
				//clear vectors
				for (int i = 0; i < W; i++)
				{
					for (int j = 0; j < T; j++)
					{
						Cwt[i][j] = 0;
					}
				}
				for (int i = 0; i < T; i++)
				{
					Ct[i] = 0;
					for (int j = 0; j < D; j++)
					{
						Ctd[i][j] = 0;
					}
				}
				//update vectors
				for (int i = 0; i < N; i++)
				{
					Cwt[wid[i]][z[i]]++;
					Ctd[z[i]][did[i]]++;
					Ct[z[i]]++;
				}
				//calculate theta and phi
				//perplexity[iter/100] = calculate(Cwt, Ct, Ctd, Cd, alpha,  beta, theta, phi, did, wid, T, W, D, N);
			} 
 
	}

}

vector<string> readFile(string fileName) {
	try{
		ifstream wordsDB(fileName.c_str());
		if(!wordsDB)
			throw 20;
		//string currentWord;
		size_t vecSize = 1000;
		size_t increment = 1000;
		vector <string> wordsDB_mem(vecSize);

		size_t i = 0;
		while(getline(wordsDB, wordsDB_mem[i++]))
		{
			if(i == vecSize-1)
			{
				vecSize += increment;
				wordsDB_mem.resize(vecSize);
			}
		}
		// Remove the extra blank elements.
		wordsDB_mem.resize(i);

		return wordsDB_mem;
	}
	catch (int e) 
	{
		cout << "An exception occurred while reading file: " << fileName << " Nr. " << e << '\n';
	}
	
}

void writetoFile(string fileName, vector<vector<string> > data) {
	ofstream myfile;
	myfile.open (fileName.c_str());
	for (unsigned i = 0; i < data.size(); i++)
	{
		for (unsigned j = 0; j < data[0].size(); j++)
		{
			myfile << data[i][j] << " ";
		}
		myfile << "\n";
	}
	myfile.close();
}


void writetoFile(string fileName, vector<int> data) {
	ofstream myfile;
	myfile.open (fileName.c_str());
	for (unsigned i = 0; i < data.size(); i++)
	{
		myfile << data[i] << " ";
	}
	myfile.close();
}

void writetoFile(string fileName, vector<double> data) {
	ofstream myfile;
	myfile.open (fileName.c_str());
	for (unsigned i = 0; i < data.size(); i++)
	{
		myfile << std::setprecision(7) << data[i] << " ";
	}
	myfile.close();
}


void writetoFile(string fileName, vector<vector<int> > data) {
	ofstream myfile;
	myfile.open (fileName.c_str());
	for (unsigned i = 0; i < data.size(); i++)
	{
		for (unsigned j = 0; j < data[0].size(); j++)
		{
			myfile << data[i][j] << " ";
		}
		myfile << "\n";
	}
	myfile.close();
}

void writetoFile(string fileName, vector<vector<double> > data) {
	ofstream myfile;
	myfile.open (fileName.c_str());
	for (unsigned i = 0; i < data.size(); i++)
	{
		for (unsigned j = 0; j < data[0].size(); j++)
		{
			myfile << std::setprecision(7) << data[i][j] << " ";
		}
		myfile << "\n";
	}
	myfile.close();

}

void writetoFileDoc(string fileName, vector<vector<double> > data) {
	ofstream myfile;
	myfile.open (fileName.c_str());
	for (unsigned i = 0; i < data[0].size(); i++)
	{
		for (unsigned j = 0; j < data.size(); j++)
		{
			myfile << std::setprecision(7) << data[j][i] << ",";
		}
		myfile << "\n";
	}
	myfile.close();

}



void writeTop(string fileName, vector<vector<double> > &phi, int numberOfTops, int T, int W) {
	vector<vector<int> > ids(T, vector<int>(numberOfTops));
	vector<vector<double> > prob(T, vector<double>(numberOfTops));
	for (int i = 0; i < T; i++)
	{
		for (int j = 0; j < numberOfTops; j++)
		{
			prob[i][j] = 0;
			ids[i][j] = 0;
		}
		for (int j = 0; j < W; j++)
		{
			for (int k = 0; k < numberOfTops; k++)
			{
				if(phi[j][i] > prob[i][k]) {

					for (int z = numberOfTops-1; z > k; z--)
					{
						prob[i][z] = prob[i][z-1];
						ids[i][z] = ids[i][z-1];
					}
					prob[i][k] = phi[j][i];
					ids[i][k] = j;
					break;
				}
			}
		}
	}
	//print topics
	ofstream myfile;
	myfile.open (fileName.c_str());
	for (unsigned i = 0; i < T; i++)
	{
		for (unsigned j = 0; j < numberOfTops; j++)
		{
			myfile << ids[i][j] << ":" << prob[i][j] << ",";
		}
		myfile << "\n";
	}
	myfile.close();
}

void printTop(vector<vector<double> > &phi, int numberOfTops, int T, int W, vector<string> dict) {
	vector<vector<string> > tops(T, vector<string>(numberOfTops));
	vector<vector<double> > prob(T, vector<double>(numberOfTops));
	for (int i = 0; i < T; i++)
	{
		for (int j = 0; j < numberOfTops; j++)
		{
			prob[i][j] = 0;
			tops[i][j] = "%invalid%";
		}
		for (int j = 0; j < W; j++)
		{
			for (int k = 0; k < numberOfTops; k++)
			{
				if(phi[j][i] > prob[i][k]) {

					for (int z = numberOfTops-1; z > k; z--)
					{
						prob[i][z] = prob[i][z-1];
						tops[i][z] = tops[i][z-1];
					}
					prob[i][k] = phi[j][i];
					tops[i][k] = dict[j];
					break;
				}
			}
		}
	}
	//print topics
	for (int i = 0; i < T; i++)
	{
		cout << "************* Topic " << i << " ****************************"<< endl;
		for (int j = 0; j < numberOfTops; j++)
		{
			cout << tops[i][j] << " ";
		}
		cout << endl;
	}
	writetoFile("tops.txt", tops);
}

int main(int argc, char* argv[]){
	//read arguments
	if (argc < 4) { // Check the value of argc. If not enough parameters have been passed, inform user and exit.
        std::cout << "Usage myprogram corpus.txt dictionary.txt topicCount iterationCount"; // Inform the user of how to use the program
        std::cin.get();
        exit(0);
    }
	string courpusName(argv[1]);
	string dictName("vocab.nytimes.txt");
	int numberOfiteration = atoi(argv[2]);
	int T = atoi(argv[3]);		//number of topics
	
	cout << courpusName << " " << dictName << " " << T << " " << numberOfiteration << endl;
	
	//read corpus
	vector<string> lines = readFile(courpusName);
	cout << "Corpus File Read Completed." << endl;
	int D = atoi(lines[0].c_str());
	int W = atoi(lines[1].c_str());
	int NNZ = atoi(lines[2].c_str());
	cout << "D:" << D << " W:" << W << " NNZ:" << NNZ << endl;
	
	//read dictionary
	vector<string> dict = readFile(dictName);
	cout << "Dictionary Read Completed." << endl;

	//declarations
	
	int mod = 1000;
	size_t vecSize = 10000;
	size_t increment = 10000;
	vector<int> wid(vecSize), did(vecSize), z(vecSize);
	
	vector<double> alpha(T);
	vector<double> beta(W);
	vector<int> Ct(T);
	vector<vector<int> > Ctd(T, vector<int>(D));
	vector<vector<int> > Cwt(W, vector<int>(T));
	vector<int> Cd(D);
	vector<vector<double> > phi(W, vector<double>(T));
	vector<vector<double> > theta(T, vector<double>(D));
	vector<double> perplexity(numberOfiteration);
        vector<double> CPUtimes(numberOfiteration);
	vector<vector<int> > DocumentIndex(D, vector<int>(mod+1));	
        vector<int> counter(D);
	for(int i = 0; i < D ; i++)
	{
	   counter[i] = 0;
	}
	

	for (int i = 0; i < T; i++)
	{
		alpha[i] = 0.5;
	}
	for (int i = 0; i < W; i++)
	{
		beta[i] = 0.1;
	}
	//read corpus line by line
	int cur = 0;
	int size = lines.size();
	for (int i = 3; i < NNZ; i++)
	{
		//cout << "Line:" << i << endl;
		std::istringstream is( lines[i] );
		int docID, wordID, count;
		is >> docID >> wordID >> count;
		//cout << docID << " " <<wordID << " "<< count << endl;
		
		docID--; wordID --;
		for (int j = 0; j < count; j++)
		{
			did[cur] = docID;
			wid[cur] = wordID;

			int r = getRandomValue(T);
			//cout << r << endl;
			Ctd[r][docID]++; Cwt[wordID][r]++; Ct[r]++; Cd[docID]++;
			z[cur] = r;
			
                        DocumentIndex[docID][counter[docID]] = cur;
			counter[docID]++;
                        if(counter[docID]%mod == 0)
			{
			  DocumentIndex[docID].resize(counter[docID]+mod);
                        }
			cur++;
			
			if(cur == vecSize-1)
			{
				//cout << "Resized vektors!" << endl;
				vecSize += increment;

				wid.resize(vecSize);
				did.resize(vecSize);
				z.resize(vecSize);
				
			}
		}
		
	}
	// Remove the extra blank elements.
	wid.resize(cur);
	did.resize(cur);
	z.resize(cur);
        for(int i = 0; i < D; i++)
	{
	  DocumentIndex[i].resize(counter[i]);
	}


	
	int N = wid.size();
	cout << "max: " << omp_get_max_threads() <<endl; 
	int P = omp_get_max_threads();
	cout << "Gibbs Sampling starting..." << endl;
	clock_t begin=clock();
	gs( wid, did, alpha, beta, z, Ctd, Cwt, Ct, Cd, theta, phi, perplexity, numberOfiteration, N, P, T, W, D, DocumentIndex,CPUtimes);
	clock_t end=clock();
	cout << "Gibbs Sampling ended..." << endl;
	cout << "Time elapsed: " << double(get_CPU_time_usage(end,begin)) << " ms ("<<double(get_CPU_time_usage(end,begin))/1000<<" sec) \n\n";
    
	perplexity[perplexity.size()-1] = calculate(Cwt, Ct, Ctd, Cd, alpha,  beta, theta, phi, did, wid, T, W, D, N, P,  DocumentIndex);

	writeTop("topics.txt", phi, 100,T, W);

	
	//write the files
	writetoFileDoc("doctopic.txt" , theta);
	/*
	writetoFile("Ct.txt" , Ct);
	writetoFile("Cwt.txt" , Cwt);
	writetoFile("Ctd.txt" , Ctd);
	writetoFile("z.txt" , z);
	writetoFile("perp.txt", perplexity);
	*/
	writetoFile("perp.txt", perplexity);
	writetoFile("CPU.txt", CPUtimes);
	cout << "Write operations finished" << endl;
	int wait;
	cin >> wait;
	return 0;
}
