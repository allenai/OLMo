#include <iostream>
#include "forward.h"
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>
#include <fstream>

#define mymin(a,b) ((a)<(b)?(a):(b))
#define mymax(a,b) ((a)>(b)?(a):(b))

void print_matrix(std::string name, float* A, int N, int M){
	std::cout<<name<<std::endl;
	for(int i = 0; i < N; i++){
		for(int j = 0; j < M; j++){
			std::cout << A[i*M+j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout<<std::endl;
}

void oracle_mmadd(float* A, float* B, float* bias, float* C, int n, int m, int t){
	// triple loop matmul and add bias
	for (int i = 0; i < n; i++){
		for (int j = 0; j < t; j++){
			float sum = 0;
			for (int k = 0; k < m; k++){
				sum += A[i*m+k] * B[k*t+j];
			}
			C[i*t+j] += sum + bias[j];
		}
	}
}

void compute_reduction(float *in, float *out, int n, int m, int gs){
	int ng;
	if(gs == -1){
		ng = 1;
		gs = m;
	}else{
		ng = m/gs;
	}
	for(int i = 0; i < n; i++){
		for(int j0 = 0; j0 < m; j0+=gs){
			int j = j0/gs;
			out[i*ng+j] = 0;
			for(int j1 = j0; j1 < j0+gs; j1++){
				out[i*ng+j] += in[i*m+j1];
			}
		}
	}
}

void quantize_sim(float* A, float* BQ, float* scales, float* zeros, int n, int m, int bits, int gs){
	//find scales and zeros arrays
	if(gs == -1){
		gs = n;
	}
	float range = (1<<bits) - 1;
	int packed = 32 / bits;

	for(int i0 = 0; i0 < n; i0+=gs){
		int row = i0/gs;
		for(int j = 0; j < m; j++){
			float min = A[i0*m + j];
			float max = A[i0*m + j];
			for(int i1 = i0; i1 < i0+gs; i1++){
				min = mymin(min, A[i1*m+j]);
				max = mymax(max, A[i1*m+j]);
			}
			scales[row*m + j] = (max-min)/range;
			zeros[row*m + j ] = min;
		}
		for(int j = 0; j < m; j++){
			for (int i1 = i0; i1 < i0+gs; i1++){
				uint32_t acc = 0;
				int temp = (A[i1*m+j] - zeros[row*m+j])/scales[row*m+j];
				float val = ((float) temp + zeros[row*m+j]) * scales[row*m+j];
				BQ[i1*m+j] = val;
			}
		}
	}

}

void quantize(float* A, int* BQ, float* scales, float* zeros, int n, int m, int bits, int gs){
	//find scales and zeros arrays
	if(gs == -1){
		gs = n;
	}
	float range = (1<<bits) - 1;
	int packed = 32 / bits;

	for(int i0 = 0; i0 < n; i0+=gs){
		int row = i0/gs;
		for(int j = 0; j < m; j++){
			float min = A[i0*m + j];
			float max = A[i0*m + j];
			for(int i1 = i0; i1 < i0+gs; i1++){
				min = mymin(min, A[i1*m+j]);
				max = mymax(max, A[i1*m+j]);
			}
			scales[row*m + j] = (max-min)/range;
			zeros[row*m + j ] = min;
		}
		for(int j = 0; j < m; j++){
			if(bits == 3){
				for (int i1 = i0; i1 < i0+gs; i1+=32){
					uint32_t acc = 0;
					int temp0 = ((int)((A[(i1+0)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 0;
					int temp1 = ((int)((A[(i1+1)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 3;
					int temp2 = ((int)((A[(i1+2)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 6;
					int temp3 = ((int)((A[(i1+3)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 9;
					int temp4 = ((int)((A[(i1+4)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 12;
					int temp5 = ((int)((A[(i1+5)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 15;
					int temp6 = ((int)((A[(i1+6)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 18;
					int temp7 = ((int)((A[(i1+7)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 21;
					int temp8 = ((int)((A[(i1+8)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 24;
					int temp9 = ((int)((A[(i1+9)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 27;
					int temp10_0 = ((int)((A[(i1+10)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 30;
					int temp10_1 = ((int)((A[(i1+10)*m+j] - zeros[row*m+j])/scales[row*m+j])) >> 2;
					int temp11 = ((int)((A[(i1+11)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 1;
					int temp12 = ((int)((A[(i1+12)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 4;
					int temp13 = ((int)((A[(i1+13)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 7;
					int temp14 = ((int)((A[(i1+14)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 10;
					int temp15 = ((int)((A[(i1+15)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 13;
					int temp16 = ((int)((A[(i1+16)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 16;
					int temp17 = ((int)((A[(i1+17)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 19;
					int temp18 = ((int)((A[(i1+18)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 22;
					int temp19 = ((int)((A[(i1+19)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 25;
					int temp20 = ((int)((A[(i1+20)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 28;
					int temp21_0 = ((int)((A[(i1+21)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 31;
					int temp21_1 = ((int)((A[(i1+21)*m+j] - zeros[row*m+j])/scales[row*m+j])) >> 1;
					int temp22 = ((int)((A[(i1+22)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 2;
					int temp23 = ((int)((A[(i1+23)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 5;
					int temp24 = ((int)((A[(i1+24)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 8;
					int temp25 = ((int)((A[(i1+25)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 11;
					int temp26 = ((int)((A[(i1+26)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 14;
					int temp27 = ((int)((A[(i1+27)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 17;
					int temp28 = ((int)((A[(i1+28)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 20;
					int temp29 = ((int)((A[(i1+29)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 23;
					int temp30 = ((int)((A[(i1+30)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 26;
					int temp31 = ((int)((A[(i1+31)*m+j] - zeros[row*m+j])/scales[row*m+j])) << 29;

					int acc0 = 0, acc1 = 0, acc2 = 0;
					
					acc0 |= temp0;
					acc0 |= temp1;
					acc0 |= temp2;
					acc0 |= temp3;
					acc0 |= temp4;
					acc0 |= temp5;
					acc0 |= temp6;
					acc0 |= temp7;
					acc0 |= temp8;
					acc0 |= temp9;
					acc0 |= temp10_0;

					acc1 |= temp10_1;
					acc1 |= temp11;
					acc1 |= temp12;
					acc1 |= temp13;
					acc1 |= temp14;
					acc1 |= temp15;
					acc1 |= temp16;
					acc1 |= temp17;
					acc1 |= temp18;
					acc1 |= temp19;
					acc1 |= temp20;
					acc1 |= temp21_0;

					acc2 |= temp21_1;
					acc2 |= temp22;
					acc2 |= temp23;
					acc2 |= temp24;
					acc2 |= temp25;
					acc2 |= temp26;
					acc2 |= temp27;
					acc2 |= temp28;
					acc2 |= temp29;
					acc2 |= temp30;
					acc2 |= temp31;

					BQ[(3*i1/32)*m+j] = acc0;
					BQ[(3*i1/32+1)*m+j] = acc1;
					BQ[(3*i1/32+2)*m+j] = acc2;
			}

			}else{
				for (int i1 = i0; i1 < i0+gs; i1+=packed){
					uint32_t acc = 0;
					for (int i2 = i1; i2 < i1+packed; i2++){
						int temp = (A[i2*m+j] - zeros[row*m+j])/scales[row*m+j];
						acc = acc | (temp << (bits*(i2-i1)));
					}
					BQ[(i1/packed)*m+j] = acc;
				}
			}
		}
	}

}

int main(int argc, char *argv[]){
	// read n m t from args
	if(argc == 0){std::cout << "Parameters not given\n"; return 0;}
	int n = atoi(argv[1]);
	int m = atoi(argv[2]);
	int t = atoi(argv[3]);
	int bits = atoi(argv[4]);
	int gs = atoi(argv[5]);
	int ng;
	if(gs == -1){
		ng = 1;
	}else{
		ng = m/gs;
	}
	float* A = new float[n*m];
	float* AB = new float[n*m];
	float* B = new float[m*t];
	float* BQS = new float[m*t];
	float* scales = new float[t*ng];
	float* zeros = new float[t*ng];
	int* BQ = new int[m*t/8];
	int* BQB = new int[m*t/8];
	float* sums = new float[n*ng];
	float* bias = new float[t];
	float* C = new float[n*t];
	float* CB = new float[n*t];
	float* C2 = new float[n*t];
	srand(1);
	for (int i = 0; i < n*m; i++){
		A[i] = (float)rand() / RAND_MAX;
	}
	for (int i = 0; i < t*m; i++){
		B[i] = (float)rand() / RAND_MAX;
	}
	for (int i = 0; i < t; i++){
		bias[i] = (float)rand() / RAND_MAX;
	}
	for (int i = 0; i < n*t; i++){
		C[i] = 0.0;
		C2[i] = 0.0;
	}
	quantize_sim(B,BQS,scales,zeros,m,t,bits,gs);
	quantize(B,BQ,scales,zeros,m,t,bits,gs);

	quantize_sim(B,BQS,scales,zeros,m,t,bits,gs);
	quantize(B,BQ,scales,zeros,m,t,bits,gs);
	oracle_mmadd(A, BQS, bias, C, n, m, t);
	pack_input(A,AB);
	pack_qw(BQ,BQB);
	pack_output(C,CB);

	compute_reduction(A,sums,n,m,gs);
	qforward(AB,BQB,scales,zeros,bias,sums,C2,n,m,t);

	float norm = 0.0;
	for (int i = 0; i < n*t; i++){
		norm += (C[i] - C2[i]) * (C[i] - C2[i]); 
	}
	if(norm / (n*t) < 0.0001){
		int iter = 30;
		for(int _ = 0; _ < iter; _++){
			qforward(AB,BQB,scales,zeros,bias,sums,C2,n,m,t);
		}

		int num_runs = 15;
		std::vector<long int> runs(num_runs);
		for(int r = 0; r < num_runs; r++){
			auto start = std::chrono::high_resolution_clock::now();
			for(int _ = 0; _ < iter; _++){
				qforward(AB,BQB,scales,zeros,bias,sums,C2,n,m,t);
			}
			auto end = std::chrono::high_resolution_clock::now();
			runs[r] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		}

		std::sort(runs.begin(), runs.end());

		float cycles_final = runs[num_runs/2 + 1] / iter;

		std::ofstream outfile;
		outfile.open("./autogptq_extension/qigen/tmp.csv", std::ios_base::app);

		print_parameters();
		outfile << cycles_final << std::endl;
	}else{
		float cycles_final = int(10e12);

		std::ofstream outfile;
		outfile.open("./autogptq_extension/qigen/tmp.csv", std::ios_base::app);

		print_parameters();
		outfile << cycles_final << std::endl;
	}

	return 0;	
}

