#include <iostream>
#define N 1024

void test00();
void test01();
void test02();
void test03();
void test04();
void test05();

void CPUAdd(int *a, int *b, int *c) {
    int tid = 0;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid++;
    }
}

__global__ void CUDAAdd1(int a, int b, int *c) {
    *c = a + b;
}

__global__ void CUDAAdd2(int *a, int *b, int *c) {
    int tid = blockIdx.x; // N个线程块 x 1个线程/线程块 = N个并行线程
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

__global__ void CUDAAdd3(int *a, int *b, int *c) {
    int tid = threadIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main(int argc, char const *argv[]) {
    // test00();
    // test01();
    // test02();
    // test03();
    // test04();
    return 0;
}

void test00() {
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    for (int i=0; i< count; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf("--- General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name );
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %d\n", prop.clockRate);
        printf("Device copy overlap: ");
        if (prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("Kernel execition timeout : ");
        if (prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else
            printf("Disabled\n");

        printf("--- Memory Information for device %d ---\n", i);
        printf("Total global mem: %ld\n", prop.totalGlobalMem);
        printf("Total constant Mem: %ld\n", prop.totalConstMem);
        printf("Max mem pitch: %ld\n", prop.memPitch);
        printf("Texture Alignment: %ld\n", prop.textureAlignment);

        printf("--- MP Information for device %d ---\n", i);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp: %d\n", prop.regsPerBlock);
        printf("Threads in warp: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}

void test01() {
    // 申请内存
    int c;      // Host端数据内存
    int *dev_c; // device端数据内存
    cudaMalloc((void**)&dev_c, sizeof(int));

    // 执行核函数计算
    CUDAAdd1<<<1,1>>>(2, 7, dev_c);// 注意尖括号个数

    // 拷贝计算结果,释放GPU内存
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_c);

    std::cout << "2 + 7 = " << c << std::endl;
}

void test02() {
    int a[N], b[N], c[N];

    // 赋值
    for (int i = 0; i < N; i++) {
        a[i] = -i + i * i;
        b[i] = i * i * i;
    }

    // CPU求和
    CPUAdd(a, b, c);

    for (int i = 0; i < N; i++)
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
}

void test03() {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // 赋值
    for (int i = 0; i < N; i++) {
        a[i] = -i + i * i;
        b[i] = i * i * i;
    }

    // GPU上分配内存
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    // 拷贝数据
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // GPU计算
    // N个线程块,每个线程块1个线程
    CUDAAdd2<<<N,1>>>(dev_a, dev_b, dev_c);

    // 拷贝计算结果,释放GPU内存
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    for (int i = 0; i < N; i++)
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
}

void test04() {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // 赋值
    for (int i = 0; i < N; i++) {
        a[i] = -i + i * i;
        b[i] = i * i * i;
    }

    // GPU上分配内存
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    // 拷贝数据
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // GPU计算
    // 1个线程块,开启N个线程
    CUDAAdd3<<<1,N>>>(dev_a, dev_b, dev_c);

    // 拷贝计算结果,释放GPU内存
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    for (int i = 0; i < N; i++)
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
}
