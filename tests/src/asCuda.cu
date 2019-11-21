/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL Header Notice in
 * each file and include the License file (licence.txt). If applicable,
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 *
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2019 Pascal Horton, University of Bern.
 */

#include "asCuda.cuh"

__global__ void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) y[i] = x[i] + y[i];
}

__global__ void addStreams(int n, float *x, float *y, int offset) {
    int index = offset + blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) y[i] = x[i] + y[i];
}

bool CudaProcessSum() {
    checkCudaErrors(cudaSetDevice(0));

    int N = 1 << 20;

    float *hx, *dx = nullptr;
    hx = (float *)malloc(N * sizeof(float));
    checkCudaErrors(cudaMalloc((void **)&dx, N * sizeof(float)));

    float *hy, *dy = nullptr;
    hy = (float *)malloc(N * sizeof(float));
    checkCudaErrors(cudaMalloc((void **)&dy, N * sizeof(float)));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        hx[i] = 1.0f;
        hy[i] = 2.0f;
    }

    checkCudaErrors(cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dy, hy, N * sizeof(float), cudaMemcpyHostToDevice));

    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, dx, dy);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(hy, dy, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) maxError = fmax(maxError, std::fabs(hy[i] - 3.0f));
    if (maxError > 0) {
        std::cout << "Max error: " << maxError << std::endl;
    }

    // Free memory
    free(hx);
    checkCudaErrors(cudaFree(dx));
    free(hy);
    checkCudaErrors(cudaFree(dy));

    return 0;
}

bool CudaProcessSumWithStreams() {
    checkCudaErrors(cudaSetDevice(0));

    const int nStreams = 8;
    cudaStream_t streams[nStreams];

    int N = 1 << 20;

    int streamSize = N / nStreams;

    float *hx, *dx = nullptr;
    hx = (float *)malloc(N * sizeof(float));
    checkCudaErrors(cudaMalloc((void **)&dx, N * sizeof(float)));

    float *hy, *dy = nullptr;
    hy = (float *)malloc(N * sizeof(float));
    checkCudaErrors(cudaMalloc((void **)&dy, N * sizeof(float)));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        hx[i] = 1.0f;
        hy[i] = 2.0f;
    }

    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);

        int offset = i * streamSize;

        checkCudaErrors(
            cudaMemcpyAsync(&dx[offset], &hx[offset], streamSize * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        checkCudaErrors(
            cudaMemcpyAsync(&dy[offset], &hy[offset], streamSize * sizeof(float), cudaMemcpyHostToDevice, streams[i]));

        int numBlocks = (streamSize + blockSize - 1) / blockSize;
        addStreams<<<numBlocks, blockSize, 0, streams[i]>>>(N, dx, dy, offset);

        checkCudaErrors(
            cudaMemcpyAsync(&hy[offset], &dy[offset], streamSize * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
    }

    checkCudaErrors(cudaDeviceSynchronize());

    for (auto &stream : streams) cudaStreamDestroy(stream);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) maxError = fmax(maxError, std::fabs(hy[i] - 3.0f));
    if (maxError > 0) {
        std::cout << "Max error: " << maxError << std::endl;
    }

    // Free memory
    free(hx);
    checkCudaErrors(cudaFree(dx));
    free(hy);
    checkCudaErrors(cudaFree(dy));

    return 0;
}