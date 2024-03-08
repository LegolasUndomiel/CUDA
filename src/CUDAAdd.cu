#include <iostream>
#define N 100

__global__ void CUDAAdd1(int a, int b, int *c) {
    *c = a + b;
}

void CPUAdd(int *a, int *b, int *c) {
    int tid = 0;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid++;
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

void test02()
{
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

int main(int argc, char const *argv[]) {
    test01();
    test02();
    return 0;
}
