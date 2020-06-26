

#include <iostream>
#include <numeric>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <ctime>

using namespace std;

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define random(x) (rand()%x)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}


void printMatrix(double* inputMatrix, const int rows, const int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << inputMatrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */

__global__ void harnessZeroKernel(double *d_augmentedMatrix, const int rowId1, const int rowId2, const int size) {
	__shared__ double blockR1[512];
	__shared__ double blockR2[512];
	const int tIdx = threadIdx.x;
	const int bIdx = blockIdx.x;
	const int colI = blockIdx.x * blockDim.x + threadIdx.x;
	if (colI < size * 2) {
		blockR1[tIdx] = d_augmentedMatrix[rowId1 * 2 * size + blockDim.x * bIdx + tIdx];
		blockR2[tIdx] = d_augmentedMatrix[rowId2 * 2 * size + blockDim.x * bIdx + tIdx];
		__syncthreads();
		d_augmentedMatrix[rowId1 * 2 * size + blockDim.x * bIdx + tIdx] = blockR1[tIdx] + blockR2[tIdx];
	}
}

__global__ void computeRowsKernel(double *d_augmentedMatrix, const int rowId, const int size) {
	__shared__ double blockR[512];
	__shared__ double Aii;
	const int tIdx = threadIdx.x;
	const int bIdx = blockIdx.x;
	const int colI = blockIdx.x * blockDim.x + threadIdx.x;
	if (colI < size * 2) {
		blockR[tIdx] = d_augmentedMatrix[rowId * 2 * size + blockDim.x * bIdx + tIdx];
		Aii = d_augmentedMatrix[rowId * 2 * size + rowId];
		__syncthreads();
		blockR[tIdx] = blockR[tIdx] / Aii;
		d_augmentedMatrix[rowId * 2 * size + blockDim.x * bIdx + tIdx] = blockR[tIdx];
	}
}

__global__ void computeColsKernel(double *d_augmentedMatrix, const int colId, const int size) {
	__shared__ double blockC[16][16];    // which col need to be zero
	__shared__ double blockCCurent[16][16];   // which col is the current col
	__shared__ double ARow[16];        // the pivot row
	const int tIdx = threadIdx.x;
	const int tIdy = threadIdx.y;
	const int rowI = blockIdx.y * blockDim.y + threadIdx.y;
	const int colI = blockIdx.x * blockDim.x + threadIdx.x;
	if (colI < size * 2 && rowI < size) {
		blockC[tIdy][tIdx] = d_augmentedMatrix[rowI * size * 2 + colId];
		if (blockC[tIdy][tIdx] != 0) {
			blockCCurent[tIdy][tIdx] = d_augmentedMatrix[rowI * size * 2 + colI];
			ARow[tIdx] = d_augmentedMatrix[colId * size * 2 + colI];
			__syncthreads();
			if (rowI != colId) {   // current row can't sub by current row
				blockCCurent[tIdy][tIdx] = blockCCurent[tIdy][tIdx] - blockC[tIdy][tIdx] * ARow[tIdx];
			}
			d_augmentedMatrix[rowI * size * 2 + colI] = blockCCurent[tIdy][tIdx];
			//d_augmentedMatrix[rowI * size * 2 + colI] = ARow[tIdx];
		}
	}
}

__global__ void augmentMatrixKernel(double *d_augmentedMatrix, double *d_inputMatrix, const int rows, const int cols) {
	const int rowI = blockIdx.y * blockDim.y + threadIdx.y;
	const int colI = blockIdx.x * blockDim.x + threadIdx.x;

	if (colI < cols && rowI < rows) {
			// initialize augmentedMatrix
			if (colI < cols / 2) {
				d_augmentedMatrix[rowI * cols + colI] = d_inputMatrix[rowI * cols / 2 + colI];
			}
			else if (colI - cols / 2 == rowI) {
				d_augmentedMatrix[rowI * cols + colI] = 1;
			} else {
				d_augmentedMatrix[rowI * cols + colI] = 0;
			}

	}
}

__global__ void getInverseMatrixKernel(double *d_augmentedMatrix, double *d_inverseMatrix, const int rows, const int cols) {
	const int rowI = blockIdx.y * blockDim.y + threadIdx.y;
	const int colI = blockIdx.x * blockDim.x + threadIdx.x;

	if (colI < cols / 2 && rowI < rows) {
			// initialize augmentedMatrix
			d_inverseMatrix[rowI * cols / 2 + colI] = d_augmentedMatrix[rowI * cols + colI + cols / 2];
	}
}

/**
 * Host function that copies the data and launches the work on GPU
 */
double *gpuMatrixInverse(double *inputMatrix, const int rows, const int cols)
{
	double *h_inverseMatrix;
	//double *h_augmentedMatrix;
	double *d_inputMatrix;
	double *d_inverseMatrix;
	double *d_augmentedMatrix;
	const int length = rows * cols;
	const int size = rows;
	//printMatrix(inputMatrix, rows, cols);
	cout << endl;
	// initialization
	h_inverseMatrix = (double *)malloc(length * sizeof(double));
	//h_augmentedMatrix = (double *)malloc(length * 2 * sizeof(double));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_augmentedMatrix, sizeof(double) * length * 2));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_inputMatrix, sizeof(double) * length));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_inverseMatrix, sizeof(double) * length));
	CUDA_CHECK_RETURN(cudaMemcpy(d_inputMatrix, inputMatrix, sizeof(double) * length, cudaMemcpyHostToDevice));


	dim3 blockSize1(16, 16);
	dim3 gridSize1(cols * 2.0 / blockSize1.x + 1, rows * 1.0 / blockSize1.y + 1);
	augmentMatrixKernel<<<gridSize1, blockSize1>>>(d_augmentedMatrix, d_inputMatrix, rows, cols * 2);
	cudaDeviceSynchronize();

	int i = 0;
	while (i < size) {
		if (inputMatrix[i * size + i] != 0) {
			dim3 blockSize2(256);
			dim3 gridSize2(cols * 2.0 / blockSize2.x + 1, 1);
			computeRowsKernel<<<gridSize2, blockSize2>>>(d_augmentedMatrix, i, size);
			cudaDeviceSynchronize();
		} else {
			int nonZeroRowIndex = 0;
			for (int j = 0; j < size; j++) {
				if (inputMatrix[j * size + i] != 0) {
					nonZeroRowIndex = j;
					break;
				}
			}
			dim3 blockSize3(256);
			dim3 gridSize3(cols * 2.0 / blockSize3.x + 1, 1);
			harnessZeroKernel<<<gridSize3, blockSize3>>>(d_augmentedMatrix, i, nonZeroRowIndex, size);
			cudaDeviceSynchronize();
			dim3 blockSize4(256);
			dim3 gridSize4(cols * 2.0 / blockSize4.x + 1, 1);
			computeRowsKernel<<<gridSize4, blockSize4>>>(d_augmentedMatrix, i, size);
			cudaDeviceSynchronize();
		}

		dim3 blockSize5(16, 16);
		dim3 gridSize5(cols * 2.0 / blockSize5.x + 1, rows * 1.0 / blockSize5.y + 1);
		computeColsKernel<<<gridSize5, blockSize5>>>(d_augmentedMatrix, i, size);
		cudaDeviceSynchronize();
		i++;
	}

	dim3 blockSize6(16, 16);
	dim3 gridSize6(cols * 2.0 / blockSize6.x + 1, rows * 1.0 / blockSize6.y + 1);
	getInverseMatrixKernel<<<gridSize1, blockSize1>>>(d_augmentedMatrix, d_inverseMatrix, rows, cols * 2);

	CUDA_CHECK_RETURN(cudaMemcpy(h_inverseMatrix, d_inverseMatrix, sizeof(double) * length, cudaMemcpyDeviceToHost));
	//CUDA_CHECK_RETURN(cudaMemcpy(h_augmentedMatrix, d_augmentedMatrix, sizeof(double) * length * 2, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(d_augmentedMatrix));
	CUDA_CHECK_RETURN(cudaFree(d_inverseMatrix));
	CUDA_CHECK_RETURN(cudaFree(d_inputMatrix));

	return h_inverseMatrix;
}

double *createTest(const int rows, const int cols)
{
	double *data = (double *)malloc(rows * cols * sizeof(double));

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; ++j) {
			data[i * rows + j]  = 0.0;
			if (i == j) {
				data[i * rows + j]  = 1.0;
			}
		}
	}

	return data;
}


int main(void)
{
	const int rows = 10;
	const int cols = 10;
	double *testMatrix = createTest(rows, cols);
	double *inverseMatrixGPU;


	// GPU code

	clock_t start1,end1;
	start1 = clock();
	inverseMatrixGPU = gpuMatrixInverse(testMatrix, rows, cols);
	end1 = clock();
	double dur1 = (double)(end1 - start1);
	cout << "\n running time on GPU is " << dur1 / CLOCKS_PER_SEC << " secs！\n" << endl;

	if (rows < 20) {
		printMatrix(inverseMatrixGPU, rows, cols);
		double *resultMatrix = (double *)malloc(cols * rows * sizeof(double));
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				resultMatrix[i * cols + j] = 0;
			}
		}
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				for (int k = 0; k < cols; k++) {
					resultMatrix[i * cols + j] +=  testMatrix[i * cols + k] * inverseMatrixGPU[k * cols + j];
				}
			}
		}

		cout << "\nTest the result from GPU\n" << endl;
		printMatrix(resultMatrix, rows, cols);

	}
	/* Free memory */
	delete[] inverseMatrixGPU;

	return 0;
}
