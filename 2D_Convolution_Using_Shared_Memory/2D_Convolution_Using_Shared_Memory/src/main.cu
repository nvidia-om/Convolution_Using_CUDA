/**
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/***********************************************************************************\
* Standard Includes
\***********************************************************************************/
#include <iostream>
#include <string>
#include <assert.h>
/* For the CUDA runtime routines (prefixed with "cuda_") */
#include <cuda_runtime.h>

/***********************************************************************************\
* # Defines
\***********************************************************************************/
#define TRUE (1)
#define FALSE (0)

/* Number of rows and columns of the global memory */
#define NUM_OF_GLOBAL_ROWS (16000)
#define NUM_OF_GLOBAL_COLS (16000)

/* Number of threads in each block */
#define BLOCK_SIZE (32)

/* Convolution Kernel size */
#define KERNEL_SIZE (2)

/* Number of row and columns of the local memory */
#define NUM_OF_LOCAL_ROWS (BLOCK_SIZE + (2 * KERNEL_SIZE))
#define NUM_OF_LOCAL_COLS (BLOCK_SIZE + (2 * KERNEL_SIZE))

/***********************************************************************************\
* Enums
\***********************************************************************************/
typedef enum Status_Tag
{
	PASSED,
	FAILED
}Status_T;

typedef enum Cuda_Event_Tag
{
	ALLOCATE_DEVICE_MATRIX_A,
	ALLOCATE_DEVICE_MATRIX_B,
	COPY_MATRIX_A_FROM_HOST_TO_DEVICE,
	LAUNCH_KERNEL_CONV2DDEVICE,
	DEVICE_SYNCHRONIZATION,
	COPY_MATRIX_B_FROM_DEVICE_TO_HOST,
	FREE_DEVICE_MATRIX_A,
	FREE_DEVICE_MATRIX_B
}Cuda_Event_T;

typedef enum Corner_Cell_Name_Tag
{
	LEFT_TOP_PADDING_CORNER,
	RIGHT_TOP_PADDING_CORNER,
	LEFT_BOTTOM_PADDING_CORNER,
	RIGHT_BOTTOM_PADDING_CORNER,
	NUM_OF_CORNERS
}Corner_Cell_Name_T;

typedef enum Ver_Side_Cell_Name_Tag
{
	LEFT_PADDING_CELL,
	RIGHT_PADDING_CELL,
	NUM_OF_VER_SIDES
}Ver_Side_Cell_Name_T;

typedef enum Hor_Side_Cell_Name_Tag
{
	TOP_PADDING_CELL,
	BOTTOM_PADDING_CELL,
	NUM_OF_HOR_SIDES
}Hor_Side_Cell_Name_T;

/***********************************************************************************\
* Structures
\***********************************************************************************/
typedef struct Result_Tag
{
	Status_T status;
	int index;
}Result_T;

typedef struct Cell_Tag
{
	int r_idx; /* Row index */
	int c_idx; /* Column index */
}Cell_T;

/***********************************************************************************\
* Function Macros
\***********************************************************************************/
#define MATRIX_TO_ARRAY_INDEX(r_idx, c_idx, num_cols) ((r_idx*num_cols) + c_idx)

/***********************************************************************************\
* CUDA Kernel Device code for 2D Convolution
\***********************************************************************************/
__global__ void
conv2DDevice(const int *in,  int *out)
{
	int g_col_idx = blockDim.x * blockIdx.x + threadIdx.x;
	int g_row_idx = blockDim.y * blockIdx.y + threadIdx.y;
	int l_col_idx = threadIdx.x + KERNEL_SIZE;
	int l_row_idx = threadIdx.y + KERNEL_SIZE;

	__shared__ int local[NUM_OF_LOCAL_ROWS*NUM_OF_LOCAL_COLS];

	/* Convert from matrix indexing to array indexing */
	int g_idx = MATRIX_TO_ARRAY_INDEX(g_row_idx, g_col_idx, NUM_OF_GLOBAL_COLS);
	int l_idx = MATRIX_TO_ARRAY_INDEX(l_row_idx, l_col_idx, NUM_OF_LOCAL_COLS);

	if ((g_row_idx < NUM_OF_GLOBAL_ROWS) && (g_col_idx < NUM_OF_GLOBAL_COLS))
	{
		/* Read input elements into shared memory */
		
		/* Fill the internal (BLOCK_SIZE*BLOCK_SIZE) matrix */
		local[l_idx] = in[g_idx];

		/* Fill the left and right padding columns of local memory */
		if (threadIdx.x < KERNEL_SIZE)
		{
			Cell_T l_ver_side[NUM_OF_VER_SIDES];
			Cell_T g_ver_side[NUM_OF_VER_SIDES];

			/* Find left and right padding column indices of local memory */
			l_ver_side[LEFT_PADDING_CELL].r_idx = l_row_idx;
			l_ver_side[LEFT_PADDING_CELL].c_idx = l_col_idx - KERNEL_SIZE;

			l_ver_side[RIGHT_PADDING_CELL].r_idx = l_row_idx;
			l_ver_side[RIGHT_PADDING_CELL].c_idx = l_col_idx + BLOCK_SIZE;

			/* Find indices of global memory whose data needs to be filled 
			   into the left and right padding columns of local memory */
			g_ver_side[LEFT_PADDING_CELL].r_idx = g_row_idx;
			g_ver_side[LEFT_PADDING_CELL].c_idx = g_col_idx - KERNEL_SIZE;

			g_ver_side[RIGHT_PADDING_CELL].r_idx = g_row_idx;
			g_ver_side[RIGHT_PADDING_CELL].c_idx = g_col_idx + BLOCK_SIZE;

			for (int cell = LEFT_PADDING_CELL; cell < NUM_OF_VER_SIDES; ++cell)
			{
				bool within_bounds = FALSE;

				/* Check if the cell is within bounds of global matrix */
				if (LEFT_PADDING_CELL == cell) {
					within_bounds = (g_ver_side[cell].c_idx >= 0);
				}

				if (RIGHT_PADDING_CELL == cell) {
					within_bounds = (g_ver_side[cell].c_idx < NUM_OF_GLOBAL_COLS);
				}

				/* Copy corner into local memory if it is within the bounds of global matrix */
				/* Convert from matrix indexing to array indexing */
				int pad_l_idx = MATRIX_TO_ARRAY_INDEX(l_ver_side[cell].r_idx, l_ver_side[cell].c_idx, NUM_OF_LOCAL_COLS);
				int pad_g_idx = MATRIX_TO_ARRAY_INDEX(g_ver_side[cell].r_idx, g_ver_side[cell].c_idx, NUM_OF_GLOBAL_COLS);
				if (TRUE == within_bounds) {
					local[pad_l_idx] = in[pad_g_idx];
				}
				else {
					local[pad_l_idx] = 0;
				}
			}
		}

		/* Fill the top and bottom padding rows */
		if (threadIdx.y < KERNEL_SIZE)
		{
			Cell_T l_hor_side[NUM_OF_HOR_SIDES];
			Cell_T g_hor_side[NUM_OF_HOR_SIDES];

			/* Find top and bottom padding row indices of local memory */
			l_hor_side[TOP_PADDING_CELL].r_idx = l_row_idx - KERNEL_SIZE;
			l_hor_side[TOP_PADDING_CELL].c_idx = l_col_idx;

			l_hor_side[BOTTOM_PADDING_CELL].r_idx = l_row_idx + BLOCK_SIZE;
			l_hor_side[BOTTOM_PADDING_CELL].c_idx = l_col_idx;

			/* Find indices of global memory whose data needs to be filled 
			   into the top and bottom padding rows of local memory */
			g_hor_side[TOP_PADDING_CELL].r_idx = g_row_idx - KERNEL_SIZE;
			g_hor_side[TOP_PADDING_CELL].c_idx = g_col_idx;

			g_hor_side[BOTTOM_PADDING_CELL].r_idx = g_row_idx + BLOCK_SIZE;
			g_hor_side[BOTTOM_PADDING_CELL].c_idx = g_col_idx;

			for (int cell = TOP_PADDING_CELL; cell < NUM_OF_HOR_SIDES; ++cell)
			{
				bool within_bounds = FALSE;

				/* Check if the cell is within bounds of global matrix */
				if (TOP_PADDING_CELL == cell) {
					within_bounds = (g_hor_side[cell].r_idx >= 0);
				}

				if (BOTTOM_PADDING_CELL == cell) {
					within_bounds = (g_hor_side[cell].r_idx < NUM_OF_GLOBAL_ROWS);
				}

				/* Copy corner into local memory if it is within the bounds of global matrix */
				/* Convert from matrix indexing to array indexing */
				int pad_l_idx = MATRIX_TO_ARRAY_INDEX(l_hor_side[cell].r_idx, l_hor_side[cell].c_idx, NUM_OF_LOCAL_COLS);
				int pad_g_idx = MATRIX_TO_ARRAY_INDEX(g_hor_side[cell].r_idx, g_hor_side[cell].c_idx, NUM_OF_GLOBAL_COLS);
				if (TRUE == within_bounds) {
					local[pad_l_idx] = in[pad_g_idx];
				}
				else {
					local[pad_l_idx] = 0;
				}
			}
		}

		/* Fill the corners */
		if ((threadIdx.x) < KERNEL_SIZE && (threadIdx.y < KERNEL_SIZE))
		{
			Cell_T l_corner[NUM_OF_CORNERS];
			Cell_T g_corner[NUM_OF_CORNERS];

			/* Find left top, right top, left bottom and right bottom padding
			   corner indices of local memory */
			l_corner[LEFT_TOP_PADDING_CORNER].r_idx = l_row_idx - KERNEL_SIZE;
			l_corner[LEFT_TOP_PADDING_CORNER].c_idx = l_col_idx - KERNEL_SIZE;

			l_corner[RIGHT_TOP_PADDING_CORNER].r_idx = l_row_idx - KERNEL_SIZE;
			l_corner[RIGHT_TOP_PADDING_CORNER].c_idx = l_col_idx + BLOCK_SIZE;

			l_corner[LEFT_BOTTOM_PADDING_CORNER].r_idx = l_row_idx + BLOCK_SIZE;
			l_corner[LEFT_BOTTOM_PADDING_CORNER].c_idx = l_col_idx - KERNEL_SIZE;

			l_corner[RIGHT_BOTTOM_PADDING_CORNER].r_idx = l_row_idx + BLOCK_SIZE;
			l_corner[RIGHT_BOTTOM_PADDING_CORNER].c_idx = l_col_idx + BLOCK_SIZE;

			/* Find indices of global memory whose data needs to be filled 
			   into the left top, right top, left bottom and right bottom padding
			   corners of local memory */
			g_corner[LEFT_TOP_PADDING_CORNER].r_idx = g_row_idx - KERNEL_SIZE;
			g_corner[LEFT_TOP_PADDING_CORNER].c_idx = g_col_idx - KERNEL_SIZE;

			g_corner[RIGHT_TOP_PADDING_CORNER].r_idx = g_row_idx - KERNEL_SIZE;
			g_corner[RIGHT_TOP_PADDING_CORNER].c_idx = g_col_idx + BLOCK_SIZE;

			g_corner[LEFT_BOTTOM_PADDING_CORNER].r_idx = g_row_idx + BLOCK_SIZE;
			g_corner[LEFT_BOTTOM_PADDING_CORNER].c_idx = g_col_idx - KERNEL_SIZE;

			g_corner[RIGHT_BOTTOM_PADDING_CORNER].r_idx = g_row_idx + BLOCK_SIZE;
			g_corner[RIGHT_BOTTOM_PADDING_CORNER].c_idx = g_col_idx + BLOCK_SIZE;

			for (int corner = LEFT_TOP_PADDING_CORNER; corner < NUM_OF_CORNERS; ++corner)
			{
				bool within_bounds = FALSE;

				/* Check if the corner is within bounds of global matrix */
				if (LEFT_TOP_PADDING_CORNER == corner){
					within_bounds = ((g_corner[corner].r_idx >= 0) && (g_corner[corner].c_idx >= 0));
				}

				if (RIGHT_TOP_PADDING_CORNER == corner){
					within_bounds = ((g_corner[corner].r_idx >= 0) && (g_corner[corner].c_idx < NUM_OF_GLOBAL_COLS));
				}

				if (LEFT_BOTTOM_PADDING_CORNER == corner){
					within_bounds = ((g_corner[corner].r_idx < NUM_OF_GLOBAL_ROWS) && (g_corner[corner].c_idx >= 0));
				}

				if (RIGHT_BOTTOM_PADDING_CORNER == corner){
					within_bounds = ((g_corner[corner].r_idx < NUM_OF_GLOBAL_ROWS) && (g_corner[corner].c_idx < NUM_OF_GLOBAL_COLS));
				}

				/* Copy corner into local memory if it is within the bounds of global matrix */
				/* Convert from matrix indexing to array indexing */
				int pad_l_idx = MATRIX_TO_ARRAY_INDEX(l_corner[corner].r_idx, l_corner[corner].c_idx, NUM_OF_LOCAL_COLS);
				int pad_g_idx = MATRIX_TO_ARRAY_INDEX(g_corner[corner].r_idx, g_corner[corner].c_idx, NUM_OF_GLOBAL_COLS);
				if (TRUE == within_bounds){
					local[pad_l_idx] = in[pad_g_idx];
				}
				else {
					local[pad_l_idx] = 0;
				}
			}
		}

		__syncthreads();

		/* Apply convolution */
		int result = 0;
		for (int row_offset = -KERNEL_SIZE; row_offset <= KERNEL_SIZE; ++row_offset)
		{
			for (int col_offset = -KERNEL_SIZE; col_offset <= KERNEL_SIZE; ++col_offset)
			{
				/* Convert local matrix row and column to local element index */
				int l_ele_idx = MATRIX_TO_ARRAY_INDEX((l_row_idx + row_offset), (l_col_idx + col_offset), NUM_OF_LOCAL_COLS);
				result += local[l_ele_idx];
			}
		}

		/* Store the result */
		out[g_idx] = result;
	}
}

/***********************************************************************************\
* Host code for 2D Convolution and comparing the result with device 2D Convolution
\***********************************************************************************/
Result_T checkResult(int* h_A, int* h_B)
{
	Result_T result;
	result.status = PASSED;
	result.index = -1;

	int mat_size = NUM_OF_GLOBAL_ROWS * NUM_OF_GLOBAL_COLS;
	
	for (int ele_idx = 0; ele_idx < mat_size; ++ele_idx)
	{
		/* Convert input element index to input matrix row and column */
		int mat_row_num = ele_idx / NUM_OF_GLOBAL_COLS;
		int mat_col_num = ele_idx % NUM_OF_GLOBAL_COLS;

		int sum = 0;

		for (int row_offset = -KERNEL_SIZE; row_offset <= KERNEL_SIZE; ++row_offset)
		{
			for (int col_offset = -KERNEL_SIZE; col_offset <= KERNEL_SIZE; ++col_offset)
			{
				/* Get kernel matrix row and column */
				int mat_ker_row_num = mat_row_num + row_offset;
				int mat_ker_col_num = mat_col_num + col_offset;

				if ((mat_ker_row_num >= 0) && (mat_ker_row_num < NUM_OF_GLOBAL_ROWS) &&
					(mat_ker_col_num >= 0) && (mat_ker_col_num < NUM_OF_GLOBAL_COLS))
				{
					/* Convert kernel matrix row and column to kernel element index */
					int ker_ele_idx = MATRIX_TO_ARRAY_INDEX(mat_ker_row_num, mat_ker_col_num, NUM_OF_GLOBAL_COLS);

					if (ker_ele_idx >= 0)
					{
						sum += h_A[ker_ele_idx];
					}
				}
			}
		}

		if (h_B[ele_idx] != sum) {
			result.status = FAILED;
			result.index = ele_idx;
			return result;
		}
	}

	return result;
}

/***********************************************************************************\
* Host code to initialize input matrix
\***********************************************************************************/
void initHostInputMatrix(int* h_A)
{
	for (int idx = 0; idx < (NUM_OF_GLOBAL_ROWS*NUM_OF_GLOBAL_COLS); ++idx)
	{
		h_A[idx] = (idx / NUM_OF_GLOBAL_COLS) + 1;
	}
}

/***********************************************************************************\
* Function to check CUDA error
\***********************************************************************************/
inline cudaError_t checkCuda(cudaError_t result, Cuda_Event_T cuda_event)
{
	char error_string[100];

	switch (cuda_event)
	{
	case ALLOCATE_DEVICE_MATRIX_A:
		strcpy(error_string, "Failed to allocate device matrix A");
		break;
	case ALLOCATE_DEVICE_MATRIX_B:
		strcpy(error_string, "Failed to allocate device matrix B");
		break;
	case COPY_MATRIX_A_FROM_HOST_TO_DEVICE:
		strcpy(error_string, "Failed to copy matrix A from host to device");
		break;
	case LAUNCH_KERNEL_CONV2DDEVICE:
		strcpy(error_string, "Failed to launch conv2DDevice kernel");
		break;
	case DEVICE_SYNCHRONIZATION:
		strcpy(error_string, "Failed to synchronize");
		break;
	case COPY_MATRIX_B_FROM_DEVICE_TO_HOST:
		strcpy(error_string, "Failed to copy matrix B from device to host");
		break;
	case FREE_DEVICE_MATRIX_A:
		strcpy(error_string, "Failed to free device matrix A");
		break;
	case FREE_DEVICE_MATRIX_B:
		strcpy(error_string, "Failed to free device matrix B");
		break;
	default:
		strcpy(error_string, "NOT DUE TO ONE OF THE CUDA EVENTS");
		break;
	}

	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s (error code: %s)\n", error_string, cudaGetErrorString(result));
		assert(result == cudaSuccess);
		exit(EXIT_FAILURE);
	}

	return result;
}

/***********************************************************************************\
* Host main routine
\***********************************************************************************/
int main(void)
{
	/* Print the matrix dimension to be used, and compute its size */
	int numElements = NUM_OF_GLOBAL_ROWS * NUM_OF_GLOBAL_COLS;
	size_t size = numElements * sizeof(int);
	printf("[Convolution of matrix of (%d, %d) elements]\n", NUM_OF_GLOBAL_ROWS, NUM_OF_GLOBAL_COLS);

	/* Allocate the host input matrix A */
	int *h_A = (int *)malloc(size);

	/* Allocate the host output matrix B */
	int *h_B = (int *)malloc(size);

	/* Verify that allocations succeeded */
	if (h_A == NULL || h_B == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix!\n");
		exit(EXIT_FAILURE);
	}

	/* Initialize the host input matrix */
	initHostInputMatrix(h_A);

	/* Allocate the device input matrix A */
	int *d_A = NULL;
	checkCuda(cudaMalloc((void **)&d_A, size), ALLOCATE_DEVICE_MATRIX_A);

	/* Allocate the device output matrix B */
	int *d_B = NULL;
	checkCuda(cudaMalloc((void **)&d_B, size), ALLOCATE_DEVICE_MATRIX_B);

	/* Copy the host input matrix A in host memory to the device input matrix in
	   device memory */
	printf("Copy input data from the host memory to the CUDA device\n");
	checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), COPY_MATRIX_A_FROM_HOST_TO_DEVICE);

	/* Launch the 2D Convolution CUDA Kernel */
	dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_dim(((NUM_OF_GLOBAL_COLS + BLOCK_SIZE - 1) / BLOCK_SIZE), ((NUM_OF_GLOBAL_ROWS + BLOCK_SIZE - 1) / BLOCK_SIZE));
	printf("CUDA kernel launch with (%d,%d) blocks of (%d,%d) threads\n", grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);
	conv2DDevice<<<grid_dim, block_dim>>>(d_A, d_B);
	checkCuda(cudaGetLastError(), LAUNCH_KERNEL_CONV2DDEVICE);

	checkCuda(cudaDeviceSynchronize(), DEVICE_SYNCHRONIZATION);

	/* Copy the device result vector in device memory to the host result vector
	   in host memory */
	printf("Copy output data from the CUDA device to the host memory\n");
	checkCuda(cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost), COPY_MATRIX_B_FROM_DEVICE_TO_HOST);

	/* Verify that the result vector is correct */
	Result_T result = checkResult(h_A, h_B);
	if(FAILED == result.status)
	{
		fprintf(stderr, "Result verification failed at element %d!\n", result.index);
		exit(EXIT_FAILURE);
	}

	printf("Test PASSED\n");

	/* Free device global memory */
	checkCuda(cudaFree(d_A), FREE_DEVICE_MATRIX_A);
	checkCuda(cudaFree(d_B), FREE_DEVICE_MATRIX_B);

	/* Free host memory */
	free(h_A);
	free(h_B);

	printf("Done\n");

	return 0;
}

