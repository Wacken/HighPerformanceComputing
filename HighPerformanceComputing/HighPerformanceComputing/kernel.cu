__host__ void matrixMult(float *A, float *B, float *C, int n) 
{
	int size = n * n * sizeof(float);
	float* Ad; float* Bd; float* Cd;
	cudaMalloc((void**)&Ad, size);
	cudaMalloc((void**)&Bd, size);
	cudaMalloc((void**)&Cd, size);
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Cd, C, size, cudaMemcpyHostToDevice);
	/* ... perform multiplication on device ... */
	cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);
	cudaFree(Ad); cudaFree(Bd); cudaFree(Cd);
}

__global__ void matrixMultKernel(float* Ad, float* Bd, float* Cd, int n) 
{
	int i = threadIdx.x;
	int k = threadIdx.y;
	float Celem = 0;
	for (int j = 0; j<n; j++) {
		float Aelem = Ad[i*n + j];
		float Belem = Bd[j*n + k];
		Celem += Aelem * Belem;
	}
	Cd[i*n + k] += Celem;
}

int main() 
{
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid(n / TILE_SIZE, n / TILE_SIZE);
	matrixMultKernel << <dimGrid, dimBlock >> >(Ad, Bd, Cd, n);

}