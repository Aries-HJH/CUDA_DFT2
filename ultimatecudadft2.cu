#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <omp.h>
#include <ctime>

using namespace std;
using namespace cv;

__device__ __managed__ double opt[2] = {0, 0};// opt[0]: max, opt[1]: min

struct ComplexNum {
	double real;
	double imagin;
};

void flatten(double* h_img, Mat img, int w, int h) {
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			h_img[i * w + j] = (double)img.at<uchar>(i, j);
		}
	}
}

void __global__ cudaGenerateMat(ComplexNum* dft_img, double* d_img, int h, int w, int N) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;
	if (n < N) {
		double R = dft_img[n].real;
		double I = dft_img[n].imagin;
		double g = sqrt(R*R + I*I);
		g = log(g+1);
		d_img[n] = g;
	}
}

void __global__ cudaTransscale(uchar* ucharimg, double* d_img, int h, int w, int N) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;
	if (n < N) {
		if( ((int(n / w) < (h / 2))) && ((int(n % w) < (w / 2))))
			ucharimg[n] = (uchar)((d_img[ (int(n / w) + h/2) * w + (int(n % w) + w / 2)] - opt[1]) * (255.0 / (opt[0] - opt[1])));
		else if( ((int(n / w) < (h / 2))) && ((int(n % w) >= (w / 2))))
			ucharimg[n] = (uchar)((d_img[ (int(n / w) + h/2) * w + (int(n % w) - w / 2)] - opt[1]) * (255.0 / (opt[0] - opt[1])));
		else if( ((int(n / w) >= (h / 2))) && ((int(n % w) < (w / 2))))
			ucharimg[n] = (uchar)((d_img[ (int(n / w) - h/2) * w + (int(n % w) + w / 2)] - opt[1])  * (255.0 / (opt[0] - opt[1])));
		else if( ((int(n / w) >= (h / 2))) && ((int(n % w) >= (w / 2))))
			ucharimg[n] = (uchar)((d_img[ (int(n / w) - h/2) * w + (int(n % w) - w / 2)] - opt[1])  * (255.0 / (opt[0] - opt[1])));
	}
}

void resMat(double* d_img, Mat& out, int h, int w) {
	for (int u = 0; u < h; u++) {
		for (int v = 0; v < w; v++) {
			out.at<uchar>(u, v) = (uchar)d_img[u * w + v];
		}
	}
}

void __global__ cudaresMat(double* d_img, uchar* ucharimg, int h, int w, int N) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;
  ucharimg[n] = (uchar)d_img[n];
}

void __global__ cudadft2(double *d_img, ComplexNum *dft_img, const int N, int w, int h) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;
	//int n = blockIdx.x + threadIdx.x * gridDim.x;
	if (n < N) {
		double real = 0.0;
		double imagin = 0.0;
		for (int i = 0; i < N; i++) {
			double x = M_PI * 2 * ((double)(i / w) * (double)(n / w) / (double)w + (double)(i % w) * (double)(n % w) / (double)h);
			double I = d_img[i];
			real += cos(x) * I;
			imagin += -sin(x) * I;	
		}
		
		dft_img[n].real = real;
		dft_img[n].imagin = imagin;
	}
}

void __global__ cudaIdft2(double *d_img, ComplexNum *dft_img, const int N, int w, int h) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;
	if (n < N) {
		double real = 0.0;
		double imagin = 0.0;
		for (int i = 0; i < N; i++) {
			double R = dft_img[i].real;
			double I = dft_img[i].imagin;
			double x = M_PI * 2 * ((double)(n / w)*(i / w) / (double)w + (double)(n % w) * (i % w) / (double)h);

			real += R * cos(x) - sin(x) * I;
			imagin += I * cos(x) + R * sin(x);
		}
		double g = sqrt(real*real + imagin * imagin)*1.0 /(w * h);
		d_img[n] = g;
	}
}

void __global__ cudaGetMax(double *d_img, double *d_imgy, int N) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int n = bid * blockDim.x + tid;
	__shared__ double s_dimg[128];
	s_dimg[tid] = (n < N) ? d_img[n] : 0.0;
	__syncthreads();

	for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
		if (tid < offset) {
			if ( s_dimg[tid] < s_dimg[tid + offset])
				s_dimg[tid] = s_dimg[tid + offset];
		}
		__syncthreads();
	}
	if (tid == 0) {
		d_imgy[bid] = s_dimg[0];
	}
}

void __global__ cudaGetMin(double *d_img, double *d_imgy, int N) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int n = bid * blockDim.x + tid;
	__shared__ double s_dimg[128];
	s_dimg[tid] = (n < N) ? d_img[n] : 0.0;
	__syncthreads();

	for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
		if (tid < offset) {
			if ( s_dimg[tid] > s_dimg[tid + offset])
				s_dimg[tid] = s_dimg[tid + offset];
		}
		__syncthreads();
	}
	if (tid == 0) {
		d_imgy[bid] = s_dimg[0];
	}
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		cout << "Parameter is not enough, one parameter is missing." << endl;
		return 0;
	}
	Mat img = imread(argv[1], 0);
	int w = img.cols;
	int h = img.rows;
	// malloc host data
	const int N = w * h;
	const int M = sizeof(double) * N;
	// host image
	double *h_img = (double*) malloc(M);
	//host Complex
	ComplexNum *dft_img = (ComplexNum*)malloc(sizeof(ComplexNum) * N); //dft
	
	flatten(h_img, img, w, h);
	const int block_size = 128;
	const int grid_size = N / 128;

	//cuda image
	double *d_img;
	double *d_imgy;
	double *h_imgy = (double*)malloc(sizeof(double) * (grid_size+1));
	cudaMalloc((void **)&d_imgy, sizeof(double) * (grid_size+1));
	// cuda dft image
	ComplexNum *dft_cuda;
	cudaMalloc((void **)&d_img, M);
	cudaMalloc((void **)&dft_cuda, N * sizeof(ComplexNum));
	// Transfer data to the device
	cudaMemcpy(d_img, h_img, M, cudaMemcpyHostToDevice);
	
	clock_t start_time, end_time;
	start_time = clock();	
	cudadft2<<<grid_size+1, block_size>>>(d_img, dft_cuda, N, w, h);
	end_time = clock();
	cout << "DFT2 time: " << (double) (end_time - start_time) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
	
	// generate Mat
	uchar *cudaucharimg;
	cudaMalloc((void **)&cudaucharimg, sizeof(uchar)*N);
	
	// generate the fourier spectrum
	cudaGenerateMat<<<grid_size+1, block_size>>>(dft_cuda, d_img, h, w, N);

	// get Max
	cudaGetMax<<<grid_size+1, block_size>>>(d_img, d_imgy, N);
	cudaMemcpy(h_imgy, d_imgy, sizeof(double) * (grid_size+1), cudaMemcpyDeviceToHost);
  	opt[0] = h_imgy[0];
	for(int i = 1; i < grid_size+1; i++) {
		if (h_imgy[i] > opt[0]) opt[0] = h_img[i];
	}

	// get Min
	cudaGetMin<<<grid_size+1, block_size>>>(d_img, d_imgy, N);
 	cudaMemcpy(h_imgy, d_imgy, sizeof(double) * (grid_size+1), cudaMemcpyDeviceToHost);
	opt[1] = h_imgy[0];
	for(int i = 0; i < grid_size+1; i++) {
		if (h_imgy[i] < opt[1]) opt[1] = h_imgy[i];
	}
  
	// save fourier spectrum
	uchar* ucharimg = (uchar*)malloc(sizeof(uchar) * N);
	cudaTransscale<<<grid_size+1, block_size>>>(cudaucharimg, d_img, h, w, N);
	cudaMemcpy(ucharimg, cudaucharimg, sizeof(uchar) * N, cudaMemcpyDeviceToHost);
	Mat rg = Mat(h, w, CV_8UC1, ucharimg);
	imwrite("./cuda_p.jpg", rg);

	// inverse fourier transform
	start_time = clock();	
	cudaIdft2<<<grid_size+1, block_size>>>(d_img, dft_cuda, N, w, h);
	end_time = clock();
	cout << "IDFT2 time: " << (double) (end_time - start_time) * 1000 / CLOCKS_PER_SEC << "ms" << endl;

	cudaresMat<<<grid_size+1, block_size>>>(d_img, cudaucharimg, h, w, N);
	cudaMemcpy(ucharimg, cudaucharimg, sizeof(uchar) * N, cudaMemcpyDeviceToHost);

	Mat res = Mat(h, w, CV_8UC1, ucharimg);
	imwrite("res.jpg", res);

	free(ucharimg);
	free(h_imgy);
	free(h_img);
	free(dft_img);
	cudaFree(d_imgy);
	cudaFree(dft_cuda);
	cudaFree(cudaucharimg);
	cudaFree(d_img);
	return 0;
}
