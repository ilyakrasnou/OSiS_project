#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_image.h"
#include "kernel.h"

#include <stdio.h>
#include <time.h>
#include <string>

#define C2IDX(i, j, w) (( i ) * ( w ) + ( j ))

const bool saveImages = true;
const size_t channels = 3;

//чтобы не мучаться - не переменная или указатель, а массив единичной длины
__constant__ unsigned char gdivisor[1];
__constant__ unsigned char goffset[1];
unsigned char divisor = 1;
unsigned char offset = 0;
const size_t threadsX = 32, threadsY = 8;

unsigned char FILTER5_CPU[] = {0, 0, 0, 0, 0,
							   0, 0, 0, 0, 0,
							   0, 0, 1, 0, 0,
							   0, 0, 0, 0, 0,
							   0, 0, 0, 0, 0 };
const size_t FILTER5_SIZE = sizeof(FILTER5_CPU) / sizeof(unsigned char);
__constant__ unsigned char FILTER5_GPU[FILTER5_SIZE];

unsigned char FILTER3_CPU[] = { 0,  0,  0,
  							    0,  1,  0,
							    0,  0,  0 };
const size_t FILTER3_SIZE = sizeof(FILTER3_CPU) / sizeof(unsigned char);
__constant__ unsigned char FILTER3_GPU[FILTER3_SIZE];

//сравниваем результат на GPU с результатом на CPU
size_t verify(const unsigned char * a, const unsigned char * gold, const size_t len) {
	for (size_t i = 0; i < len; ++i) if (a[i] != gold[i]) return i;
	return -1;
}

void expandBoundaries(unsigned char *dst, const unsigned char *src, const size_t w, const size_t h, const size_t we, const size_t he) {
	const size_t halfDim = (we - w) / 2;
	for (size_t ip = 0, i = 0; ip < he && i < h; ip < halfDim || ip >= he - 2*halfDim ? i : ++i, ++ip)
		for (size_t jp = 0, j = 0; jp < we && j < w; jp < halfDim || jp >= we - 2*halfDim ? j : ++j, ++jp)
			for (size_t k = 0; k < channels; ++k)
				dst[C2IDX(ip, jp * channels + k, we * channels)] = src[C2IDX(i, j * channels + k, w * channels)];
}

unsigned char applyFilterCPU(unsigned char *seq, const unsigned char *filter, const size_t size) {
	unsigned int result = 0;
	for (size_t i = 0; i < size; ++i)
		result += seq[i] * filter[i];
	result = result / divisor + offset;
	return (unsigned char) result;
}

void filter3CPU(unsigned char *dst, const unsigned char *src, const size_t w, const size_t h) {
	const size_t we = w + 2;
	for (size_t i = 0, ip = 1; i < h; ++i, ++ip) {
		for (size_t j = 0, jp = 1; j < w; ++j, ++jp) {
			for (size_t c = 0; c < channels; ++c) {
				unsigned char temp[FILTER3_SIZE] = {
					src[C2IDX(ip - 1, (jp - 1) * channels + c, we * channels)], src[C2IDX(ip - 1, (jp + 0) * channels + c, we * channels)],src[C2IDX(ip - 1, (jp + 1) * channels + c, we * channels)],
					src[C2IDX(ip + 0, (jp - 1) * channels + c, we * channels)], src[C2IDX(ip + 0, (jp + 0) * channels + c, we * channels)],src[C2IDX(ip + 0, (jp + 1) * channels + c, we * channels)],
					src[C2IDX(ip + 1, (jp - 1) * channels + c, we * channels)], src[C2IDX(ip + 1, (jp + 0) * channels + c, we * channels)],src[C2IDX(ip + 1, (jp + 1) * channels + c, we * channels)]
				};
				dst[C2IDX(i, j * channels + c, w * channels)] = applyFilterCPU(temp, FILTER3_CPU, FILTER3_SIZE);
			}
		}
	}
}

void filter5CPU(unsigned char *dst, const unsigned char *src, const size_t w, const size_t h) {
	const size_t we = w + 4;
	for (size_t i = 0, ip = 2; i < h; ++i, ++ip) {
		for (size_t j = 0, jp = 2; j < w; ++j, ++jp) {
			for (size_t c = 0; c < channels; ++c) {
				unsigned char temp[FILTER5_SIZE] = {
					src[C2IDX(ip - 2, (jp - 2) * channels + c, we * channels)], src[C2IDX(ip - 2, (jp - 1) * channels + c, we * channels)], src[C2IDX(ip - 2, (jp + 0) * channels + c, we * channels)],src[C2IDX(ip - 2, (jp + 1) * channels + c, we * channels)],src[C2IDX(ip - 2, (jp + 2) * channels + c, we * channels)],
					src[C2IDX(ip - 1, (jp - 2) * channels + c, we * channels)], src[C2IDX(ip - 1, (jp - 1) * channels + c, we * channels)], src[C2IDX(ip - 1, (jp + 0) * channels + c, we * channels)],src[C2IDX(ip - 1, (jp + 1) * channels + c, we * channels)],src[C2IDX(ip - 1, (jp + 2) * channels + c, we * channels)],
					src[C2IDX(ip + 0, (jp - 2) * channels + c, we * channels)], src[C2IDX(ip + 0, (jp - 1) * channels + c, we * channels)], src[C2IDX(ip + 0, (jp + 0) * channels + c, we * channels)],src[C2IDX(ip + 0, (jp + 1) * channels + c, we * channels)],src[C2IDX(ip + 0, (jp + 2) * channels + c, we * channels)],
					src[C2IDX(ip + 1, (jp - 2) * channels + c, we * channels)], src[C2IDX(ip + 1, (jp - 1) * channels + c, we * channels)], src[C2IDX(ip + 1, (jp + 0) * channels + c, we * channels)],src[C2IDX(ip + 1, (jp + 1) * channels + c, we * channels)],src[C2IDX(ip + 1, (jp + 2) * channels + c, we * channels)],
					src[C2IDX(ip + 2, (jp - 2) * channels + c, we * channels)], src[C2IDX(ip + 2, (jp - 1) * channels + c, we * channels)], src[C2IDX(ip + 2, (jp + 0) * channels + c, we * channels)],src[C2IDX(ip + 2, (jp + 1) * channels + c, we * channels)],src[C2IDX(ip + 2, (jp + 2) * channels + c, we * channels)]
				};
				dst[C2IDX(i, j * channels + c, w * channels)] = applyFilterCPU(temp, FILTER5_CPU, FILTER5_SIZE);
			}
		}
	}
}

__device__ unsigned char applyFilterGPU(unsigned char* seq, const unsigned char* filter, const size_t size) {
	unsigned int result = 0;
	for (size_t i = 0; i < size; ++i)
		result += seq[i] * filter[i];
	result = result / gdivisor[0] + goffset[0];
	return unsigned char(result);
}

__global__ void filter3GPU(int *dst, const int *src, const size_t pitchDst, const size_t pitchSrc, const size_t h) {
	__shared__ int smemIn[threadsY + 2][(threadsX)* channels + 2];
	__shared__ int smemOut[threadsY][(threadsX)* channels];
	size_t Y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t X = (blockIdx.x * blockDim.x) * channels + threadIdx.x;

	if (X < pitchSrc && Y < h + 2) {
		// загрузка изображения
		for (size_t i = 0; i < channels; ++i)
			smemIn[threadIdx.y][threadIdx.x + threadsX * i] = src[C2IDX(Y, X + threadsX * i, pitchSrc)];
		// 2 крайних справа
		if (threadIdx.x <= 1)
			smemIn[threadIdx.y][threadIdx.x + threadsX * channels] = src[C2IDX(Y, X + threadsX * channels, pitchSrc)];
		// ну и оставшиеся 2 края снизу
		if (threadIdx.y >= blockDim.y - 2 && Y + 2 < h + 2) {
			for (size_t i = 0; i < channels; ++i)
				smemIn[threadIdx.y + 2][threadIdx.x + threadsX * i] = src[C2IDX(Y + 2, X + threadsX * i, pitchSrc)];
			// 2 крайних справа
			if (threadIdx.x < 2)
				smemIn[threadIdx.y + 2][threadIdx.x + threadsX * channels] = src[C2IDX(Y + 2, X + threadsX * channels, pitchSrc)];
		}
	}
	// синхронизируем нити, чтобы гарантировать, 
	// что в разделяемую память записаны все данные
	__syncthreads();
	// по ширине 4
	// + 1 с каждого края
	unsigned char data[3][6 * channels];
	for (size_t i = 0; i < 3; ++i) {
		memcpy(data[i], &smemIn[threadIdx.y + i][threadIdx.x * channels], 6 * channels);
	}
	// оптимальное чтение - 4 байта, при этом каждый пиксель - 3 байта
	// поэтому выбираем 4 * 3
	char result[4 * channels] = { 0 };
	for (size_t i = 0; i < 4; ++i) {
		for (size_t c = 0; c < channels; ++c) {
			unsigned char temp[9] = {
				data[0][(i + 0) * channels + c], data[0][(i + 1) * channels + c], data[0][(i + 2) * channels + c],
				data[1][(i + 0) * channels + c], data[1][(i + 1) * channels + c], data[1][(i + 2) * channels + c],
				data[2][(i + 0) * channels + c], data[2][(i + 1) * channels + c], data[2][(i + 2) * channels + c]
			};
			result[i * channels + c] = applyFilterGPU(temp, FILTER3_GPU, FILTER3_SIZE);
		}
	}
	memcpy(&smemOut[threadIdx.y][threadIdx.x * channels], result, 4 * channels);

	if (X < pitchDst && Y < h)
		for (size_t i = 0; i < channels; ++i)
			dst[C2IDX(Y, X + threadsX * i, pitchDst)] = smemOut[threadIdx.y][threadsX * i + threadIdx.x];
}

__global__ void filter5GPU(int *dst, const int *src, const size_t pitchDst, const size_t pitchSrc, const size_t h) {
	__shared__ int smemIn[threadsY + 4][(threadsX)* channels + 4];
	__shared__ int smemOut[threadsY][(threadsX)* channels];
	size_t Y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t X = (blockIdx.x * blockDim.x) * channels + threadIdx.x;

	if (X < pitchSrc && Y < h + 4) {
		// загрузка изображения
		for (size_t i = 0; i < channels; ++i)
			smemIn[threadIdx.y][threadIdx.x + threadsX * i] = src[C2IDX(Y, X + threadsX * i, pitchSrc)];
		// 4 крайних справа
		if (threadIdx.x < 4)
			smemIn[threadIdx.y][threadIdx.x + threadsX * channels] = src[C2IDX(Y, X + threadsX * channels, pitchSrc)];
		// ну и оставшиеся 4 края снизу
		if (threadIdx.y >= blockDim.y - 4 && Y + 4 < h + 4) {
			for (size_t i = 0; i < channels; ++i)
				smemIn[threadIdx.y + 4][threadIdx.x + threadsX * i] = src[C2IDX(Y + 4, X + threadsX * i, pitchSrc)];
			// 4 крайних справа
			if (threadIdx.x < 4)
				smemIn[threadIdx.y + 4][threadIdx.x + threadsX * channels] = src[C2IDX(Y + 4, X + threadsX * channels, pitchSrc)];
		}
	}
	// синхронизируем нити, чтобы гарантировать, 
	// что в разделяемую память записаны все данные
	__syncthreads();
	// ширина 4
	// + 2 с каждого края
	unsigned char data[5][8 * channels];
	for (size_t i = 0; i < 5; ++i) {
		memcpy(data[i], &smemIn[threadIdx.y + i][threadIdx.x * channels], 8 * channels);
	}
	// оптимальное чтение - 4 байта, при этом каждый пиксель - 3 байта
	// поэтому выбираем 4 * 3
	char result[4 * channels] = { 0 };
	for (size_t i = 0; i < 4; ++i) {
		for (size_t c = 0; c < channels; ++c) {
			unsigned char temp[] = {
				data[0][(i + 0) * channels + c], data[0][(i + 1) * channels + c], data[0][(i + 2) * channels + c], data[0][(i + 3) * channels + c],data[0][(i + 4) * channels + c],
				data[1][(i + 0) * channels + c], data[1][(i + 1) * channels + c], data[1][(i + 2) * channels + c], data[1][(i + 3) * channels + c],data[1][(i + 4) * channels + c],
				data[2][(i + 0) * channels + c], data[2][(i + 1) * channels + c], data[2][(i + 2) * channels + c], data[2][(i + 3) * channels + c],data[2][(i + 4) * channels + c],
				data[3][(i + 0) * channels + c], data[3][(i + 1) * channels + c], data[3][(i + 2) * channels + c], data[3][(i + 3) * channels + c],data[3][(i + 4) * channels + c],
				data[4][(i + 0) * channels + c], data[4][(i + 1) * channels + c], data[4][(i + 2) * channels + c], data[4][(i + 3) * channels + c],data[4][(i + 4) * channels + c],
			};
			result[i * channels + c] = applyFilterGPU(temp, FILTER5_GPU, FILTER5_SIZE);
		}
	}
	memcpy(&smemOut[threadIdx.y][threadIdx.x * channels], result, 4 * channels);

	if (X < pitchDst && Y < h)
		for (size_t i = 0; i < channels; ++i)
			dst[C2IDX(Y, X + threadsX * i, pitchDst)] = smemOut[threadIdx.y][threadsX * i + threadIdx.x];
}

extern "C" void __declspec(dllexport) runFilter3(unsigned char* filter, unsigned char divisor_,
	unsigned char offset_, const char* cFileame, const unsigned char compare)
{
	std::string filename(cFileame);
	unsigned char *hOrigin = NULL, *hSrc = NULL, *hDst = NULL, *hResult = NULL, *dSrc = NULL, *dDst = NULL;
	unsigned int w, h, c, we, he;
	size_t pitchSrc, pitchDst;
	for (size_t i = 0; i < FILTER3_SIZE; ++i)
	{
		FILTER3_CPU[i] = filter[i];
	}
	divisor = divisor_;
	offset = offset_;
	__loadPPM(filename.c_str(), &hOrigin, &w, &h, &c);

	// расширение границ
	we = w + 2, he = h + 2;
	hSrc = (unsigned char*)malloc(sizeof(unsigned char) * we * he * c);
	hResult = (unsigned char*)malloc(sizeof(unsigned char) * w * h * c);
	expandBoundaries(hSrc, hOrigin, w, h, we, he);

	int countDevice;
	bool cudaIsAvailable = cudaGetDeviceCount(&countDevice) == cudaSuccess;
	float acceleration = 0;
	if (!cudaIsAvailable || compare) {
		// CPU, если не поддерживается CUDA
		if (!cudaIsAvailable) {
			fprintf(stdout, "No device available for execution, processing on CPU\n");
			fflush(stdout);
		}
		clock_t startCPU, stopCPU;
		startCPU = clock();
		filter3CPU(hResult, hSrc, w, h);
		stopCPU = clock();
		float elapsedTimeCPU = (float)(stopCPU - startCPU);
		fprintf(stdout, "Elapsed CPU time: %.3f\n", elapsedTimeCPU);
		fflush(stdout);
		acceleration = elapsedTimeCPU;
		if (saveImages) {
			std::string fileNameFilteredCPU = filename;
			fileNameFilteredCPU.replace(fileNameFilteredCPU.end() - 4, fileNameFilteredCPU.end(), "f_cpu.ppm");
			__savePPM(fileNameFilteredCPU.c_str(), hResult, w, h, c);
		}
	}
	if (cudaIsAvailable) {
		// GPU
		cudaMemcpyToSymbol(FILTER3_GPU, FILTER3_CPU, sizeof(FILTER3_CPU));
		cudaMemcpyToSymbol(gdivisor, &divisor, sizeof(divisor));
		cudaMemcpyToSymbol(goffset, &offset, sizeof(offset));

		hDst = (unsigned char*)malloc(sizeof(unsigned char) * w * h * c);
		memset(hDst, 0, w * h * c);
		cudaMallocPitch(&dSrc, &pitchSrc, we * c, he);
		cudaMallocPitch(&dDst, &pitchDst, w * c, h);
		cudaMemset2D(dDst, pitchDst, 0, w * c, h);
		cudaMemcpy2D(dSrc, pitchSrc, hSrc, we * c, we * c, he, cudaMemcpyHostToDevice);
		dim3 dimBlock(threadsX, threadsY);
		//округление вверх
		dim3 dimGrid((w + 127) / 128, (h + 7) / 8);
		cudaEvent_t startGPU, stopGPU;
		cudaEventCreate(&startGPU);
		cudaEventCreate(&stopGPU);
		cudaEventRecord(startGPU, 0);
		filter3GPU <<< dimGrid, dimBlock >>> ((int*)dDst, (const int*)dSrc, pitchDst >> 2, pitchSrc >> 2, h);
		cudaEventRecord(stopGPU, 0);
		cudaEventSynchronize(stopGPU);
		float elapsedTimeGPU;
		cudaEventElapsedTime(&elapsedTimeGPU, startGPU, stopGPU);
		fprintf(stdout, "Elapsed GPU time: %.3f\n", elapsedTimeGPU);
		fflush(stdout);
		cudaEventDestroy(startGPU);
		cudaEventDestroy(stopGPU);
		acceleration = acceleration / elapsedTimeGPU;
		cudaMemcpy2D(hDst, w * c, dDst, pitchDst, w * c, h, cudaMemcpyDeviceToHost);
		if (saveImages) {
			std::string fileNameFilteredGPU = filename;
			fileNameFilteredGPU.replace(fileNameFilteredGPU.end() - 4, fileNameFilteredGPU.end(), "f_gpu.ppm");
			__savePPM(fileNameFilteredGPU.c_str(), hDst, w, h, c);
		}

		cudaFree(dSrc);
		cudaFree(dDst);
		cudaDeviceReset();
	}

	if (compare && cudaIsAvailable) {
		size_t errorIdx = verify(hDst, hResult, w * h * c);
		if (errorIdx != -1) fprintf(stderr, "Error at %zu\n", errorIdx);
		else fprintf(stdout, "Verified!\nAcceleration: %.3f\n", acceleration);
		fflush(stdout);
	}
	if (hDst != NULL) free(hDst);
	if (hResult != NULL) free(hResult);
	if (hSrc != NULL) free(hSrc);
	if (hOrigin != NULL) free(hOrigin);
}

extern "C" void __declspec(dllexport) runFilter5(unsigned char* filter, unsigned char divisor_,
	unsigned char offset_, const char* cFileame, const unsigned char compare)
{
	std::string filename(cFileame);
	unsigned char *hOrigin = NULL, *hSrc = NULL, *hDst = NULL, *hResult = NULL, *dSrc = NULL, *dDst = NULL;
	unsigned int w, h, c, we, he;
	size_t pitchSrc, pitchDst;
	for (size_t i = 0; i < FILTER5_SIZE; ++i)
	{
		FILTER5_CPU[i] = filter[i];
	}
	divisor = divisor_;
	offset = offset_;
	__loadPPM(filename.c_str(), &hOrigin, &w, &h, &c);

	// Расширение границ
	we = w + 4, he = h + 4;
	hSrc = (unsigned char*)malloc(sizeof(unsigned char) * we * he * c);
	hResult = (unsigned char*)malloc(sizeof(unsigned char) * w * h * c);
	expandBoundaries(hSrc, hOrigin, w, h, we, he);

	int countDevice;
	bool cudaIsAvailable = cudaGetDeviceCount(&countDevice) == cudaSuccess;
	float acceleration = 0;
	if (!cudaIsAvailable || compare) {
		// CPU, если не поддерживается CUDA
		if (countDevice == 0) {
			fprintf(stdout, "No device available for execution, processing on CPU\n");
			fflush(stdout);
		}
		clock_t startCPU, stopCPU;
		startCPU = clock();
		filter5CPU(hResult, hSrc, w, h);
		stopCPU = clock();
		float elapsedTimeCPU = (float)(stopCPU - startCPU);
		fprintf(stdout, "Elapsed CPU time: %.3f\n", elapsedTimeCPU);
		fflush(stdout);
		acceleration = elapsedTimeCPU;
		if (saveImages) {
			std::string fileNameFilteredCPU = filename;
			fileNameFilteredCPU.replace(fileNameFilteredCPU.end() - 4, fileNameFilteredCPU.end(), "f_cpu.ppm");
			__savePPM(fileNameFilteredCPU.c_str(), hResult, w, h, c);
		}
	}
	if (cudaIsAvailable) {
		// GPU
		cudaMemcpyToSymbol(FILTER5_GPU, FILTER5_CPU, sizeof(FILTER5_CPU));
		cudaMemcpyToSymbol(gdivisor, &divisor, sizeof(divisor));
		cudaMemcpyToSymbol(goffset, &offset, sizeof(offset));

		hDst = (unsigned char*)malloc(sizeof(unsigned char) * w * h * c);
		memset(hDst, 0, w * h * c);
		cudaMallocPitch(&dSrc, &pitchSrc, we * c, he);
		cudaMallocPitch(&dDst, &pitchDst, w * c, h);
		cudaMemset2D(dDst, pitchDst, 0, w * c, h);
		cudaMemcpy2D(dSrc, pitchSrc, hSrc, we * c, we * c, he, cudaMemcpyHostToDevice);
		dim3 dimBlock(threadsX, threadsY);
		//округление вверх
		dim3 dimGrid((w + 127) / 128, (h + 7) / 8);
		cudaEvent_t startGPU, stopGPU;
		cudaEventCreate(&startGPU);
		cudaEventCreate(&stopGPU);
		cudaEventRecord(startGPU, 0);
		filter5GPU <<< dimGrid, dimBlock >>> ((int*)dDst, (const int*)dSrc, pitchDst >> 2, pitchSrc >> 2, h);
		cudaEventRecord(stopGPU, 0);
		cudaEventSynchronize(stopGPU);
		float elapsedTimeGPU;
		cudaEventElapsedTime(&elapsedTimeGPU, startGPU, stopGPU);
		fprintf(stdout, "Elapsed GPU time: %.3f\n", elapsedTimeGPU);
		fflush(stdout);
		cudaEventDestroy(startGPU);
		cudaEventDestroy(stopGPU);
		acceleration = acceleration / elapsedTimeGPU;
		cudaMemcpy2D(hDst, w * c, dDst, pitchDst, w * c, h, cudaMemcpyDeviceToHost);
		if (saveImages) {
			std::string fileNameFilteredGPU = filename;
			fileNameFilteredGPU.replace(fileNameFilteredGPU.end() - 4, fileNameFilteredGPU.end(), "f_gpu.ppm");
			__savePPM(fileNameFilteredGPU.c_str(), hDst, w, h, c);
		}

		cudaFree(dSrc);
		cudaFree(dDst);
		cudaDeviceReset();
	}

	if (compare && cudaIsAvailable) {
		size_t errorIdx = verify(hDst, hResult, w * h * c);
		if (errorIdx != -1) fprintf(stderr, "Error at %zu\n", errorIdx);
		else fprintf(stdout, "Verified!\nAcceleration: %.3f\n", acceleration);
		fflush(stdout);
	}

	if (hDst != NULL) free(hDst);
	if (hResult != NULL) free(hResult);
	if (hSrc != NULL) free(hSrc);
	if (hOrigin != NULL) free(hOrigin);
}

// для тестрирования
//int main()
//{
//	std::string filename = "D:\\Univer\\AVS\\ImageProcessing\\world.ppm";
//	//std::cin >> filename;
//	/*unsigned char arr[FILTER3_SIZE] = { 0,  0,  0,
//								0,  -1,  0,
//								0,  0,  0 };*/
//	unsigned char arr[FILTER5_SIZE] = { 0, 0, 0, 0, 0,
//										0, 0, 0, 0, 0,
//										0, 0, -1, 0, 0,
//										0, 0, 0, 0, 0,
//										0, 0, 0, 0, 0 };
//	int input;
//	unsigned char divisor_, offset_;
//	std::cout << "Enter 5*5 filter: " << std::endl;
//	/*for (size_t i = 0; i < FILTER5_SIZE; ++i) {
//		std::cin >> input;
//		arr[i] = input;
//	}*/
//	std::cout << "Enter div: " << std::endl;
//	std::cin >> input;
//	divisor_ = input;
//	std::cout << "Enter offset: " << std::endl;
//	std::cin >> input;
//	offset_ = input;
//	runFilter5(arr, divisor_, offset_, filename.c_str(), 1);
//	std::fflush(stdout);
//}
