#include <iostream>
#include <omp.h>
#include <stdio.h>

using namespace std;

void test01();
void test02(int);
void test03(int);

int main(int argc, char const *argv[]) {
    // printf("There are %d processors available\n", omp_get_num_procs());//
    // 返回可并行的处理器个数 printf("当前活跃的线程个数：%d\n",
    // omp_get_num_threads()); test01();
    test02(10000);
    // test03(10000);
    return 0;
}

void test01() {
// 并行时用 printf 打印信息，不能用 std::cout
// OpenMP 和 CUDA 中同样如此
#pragma omp parallel for
    for (int i = 0; i < 32; i++)
        printf("i = %d\tfrom thread No. %d\n", i,
               omp_get_thread_num()); // 返回线程编号
#pragma omp parallel for
    for (int i = 0; i < 32; i++)
        cout << "i = " << i << "\tfrom thread No. " << omp_get_thread_num()
             << endl;
}

void test02(int num) {
    // long long int sum[omp_get_num_procs()] = {0};

    int *sum = new int[omp_get_num_procs()];
    for (int i = 0; i < omp_get_num_procs(); i++)
        sum[i] = 0;

    int result = 0;
#pragma omp parallel for
    for (int i = 1; i < num + 1; i++)
        sum[omp_get_thread_num()] += i;

    for (int i = 0; i < omp_get_num_procs(); i++)
        result += sum[i];

    printf("parallel result = %d\n", result);
    delete[] sum;
}

void test03(int num) {
    int result = 0;
    for (int i = 1; i < num + 1; i++)
        result += i;

    printf("serial result = %d\n", result);
}
