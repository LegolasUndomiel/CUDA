#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>

using namespace std::chrono;
using std::cout;
using std::endl;

#define PI 3.14159265358979323846

void test01(int, int);

__host__ __device__ inline float calculateSine(float, int = 1000);
__host__ __device__ inline float calculateCosine(float, int = 1000);

double trapezoidalIntegral(int, float, float, float (*fun)(float, int),
                           int = 1000);
double trapezoidalIntegralOMP(int, float, float, float (*fun)(float, int),
                              int = 1000);
double trapezoidalIntegralOMPArray(int, float, float, float (*fun)(float, int),
                                   int = 1000);
__global__ void trapezoidalIntegralCUDA(float *, int, float, float, int = 1000);

void Timer(double (*fun)(int, float, float, float (*)(float, int), int),
           float (*cal)(float, int), int, int = 1000);

int main(int argc, char const *argv[]) {
    int steps = (argc > 1) ? atoi(argv[1]) : 1000000;
    int terms = (argc > 2) ? atoi(argv[2]) : 1000;

    cout << "单线程版sin(x)积分：" << endl;
    Timer(trapezoidalIntegral, calculateSine, steps, terms);

    cout << "单线程版cos(x)积分：" << endl;
    Timer(trapezoidalIntegral, calculateCosine, steps, terms);

    cout << "多线程OpenMP版sin(x)积分：原生reduction" << endl;
    Timer(trapezoidalIntegralOMP, calculateSine, steps, terms);

    cout << "多线程OpenMP版sin(x)积分：使用数组reduction" << endl;
    Timer(trapezoidalIntegralOMPArray, calculateSine, steps, terms);

    cout << "多线程CUDA版sin(x)积分：" << endl;
    test01(steps, terms);
    return 0;
}

/**
 * @brief 计算sin(x)的近似值，使用泰勒级数的前n项和作为近似结果。
 * @param x 输入的值，用于计算sin(x)
 * @param n 指定使用泰勒级数的前n项和作为近似结果
 * @return 返回sin(x)的近似值
 */
__host__ __device__ inline float calculateSine(float x, int n) {
    float term = x;      // 初始化当前项
    float result = term; // 初始化近似结果
    float x2 = x * x;

    // sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
    for (int i = 1; i < n; i++) {
        term *= -x2 / (float)((2 * i) * (2 * i + 1)); // 计算下一项
        result += term; // 累加当前项到近似结果
    }

    return result; // 返回近似值
}

/**
 * @brief 计算cos(x)的近似值，使用泰勒级数的前n项和作为近似结果。
 * @param x 输入的值，用于计算cos(x)
 * @param n 指定使用泰勒级数的前n项和作为近似结果
 * @return 返回cos(x)的近似值
 */
__host__ __device__ inline float calculateCosine(float x, int n) {
    float term = 1.0;    // 初始化当前项
    float result = term; // 初始化近似结果
    float x2 = x * x;

    // cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ...
    for (int i = 1; i < n; i++) {
        term *= -x2 / (float)((2 * i - 1) * (2 * i)); // 计算下一项
        result += term; // 累加当前项到近似结果
    }

    return result; // 返回近似值
}

/**
 * @brief 梯形积分（Trapezoidal rule）求解积分
 * 计算范围为[a,b]，步长为(b-a)/(n-1)，积分点个数为n。
 * @param n 积分点个数
 * @param a 积分下限
 * @param b 积分上限
 * @param fun 被积函数
 * @param terms 被积函数泰勒级数的项数，默认为1000
 */
double trapezoidalIntegral(int n, float a, float b, float (*fun)(float, int),
                           int terms) {
    double sum = 0;               // 初始化和
    double h = (b - a) / (n - 1); // 步长
    float x = 0.0;                // 初始化x

    for (int i = 0; i < n; i++) {
        x = a + i * h;        // 更新x
        sum += fun(x, terms); // 累加梯形面积
    }

    // Trapezoidal rule = (h/2)*(f(a) + 2*f(a+h) + ... + 2*f(a+(n-1)*h) + f(b))
    sum -= 0.5 * (fun(a, terms) + fun(b, terms));
    sum *= h;

    return sum; // 返回积分结果
}

/**
 * @brief 多线程版(OpenMP)梯形积分（Trapezoidal rule）求解积分
 * 计算范围为[a,b]，步长为(b-a)/(n-1)，积分点个数为n。
 * @param n 积分点个数
 * @param a 积分下限
 * @param b 积分上限
 * @param fun 被积函数
 * @param terms 被积函数泰勒级数的项数，默认为1000
 */
double trapezoidalIntegralOMP(int n, float a, float b, float (*fun)(float, int),
                              int terms) {
    double sum = 0;               // 初始化和
    double h = (b - a) / (n - 1); // 步长
    float x = 0.0;                // 初始化x

    int threads = omp_get_max_threads();
    omp_set_num_threads(threads);
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++) {
        x = a + i * h;        // 更新x
        sum += fun(x, terms); // 累加梯形面积
    }

    // Trapezoidal rule = (h/2)*(f(a) + 2*f(a+h) + ... + 2*f(a+(n-1)*h) + f(b))
    sum -= 0.5 * (fun(a, terms) + fun(b, terms));
    sum *= h;

    return sum; // 返回积分结果
}

/**
 * @brief 使用OpenMP并行化计算积分，使用数组保存每个线程的结果，最后汇总总和。
 * @param n 积分点个数
 * @param a 积分下限
 * @param b 积分上限
 * @param fun 被积函数
 * @param terms 被积函数泰勒级数的项数，默认为1000
 */
double trapezoidalIntegralOMPArray(int n, float a, float b,
                                   float (*fun)(float, int), int terms) {
    double sum = 0.0;             // 初始化和
    double h = (b - a) / (n - 1); // 步长
    float x = 0.0;                // 初始化x
    int threads = omp_get_max_threads();
    double *result = new double[threads];
    for (int i = 0; i < threads; i++)
        result[i] = 0.0;

    double omp_set_num_threads(threads);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x = a + i * h;                                 // 更新x
        result[omp_get_thread_num()] += fun(x, terms); // 累加梯形面积
    }

    for (int i = 0; i < threads; i++)
        sum += result[i];

    delete[] result;

    // Trapezoidal rule = (h/2)*(f(a) + 2*f(a+h) + ... + 2*f(a+(n-1)*h) + f(b))
    sum -= 0.5 * (fun(a, terms) + fun(b, terms));
    sum *= h;

    return sum; // 返回积分结果
}

/**
 * @brief 使用CUDA并行计算梯形积分（Trapezoidal rule）
 * @param[out] sum 积分结果保存的地址
 * @param[in] n 积分区间[a, b]被分成n个区间
 * @param[in] a 积分区间下限
 * @param[in] h 积分区间宽度（h=(b-a)/(n-1)）
 * @param[in] terms 被积函数的 Taylor 级数项数，默认为1000
 */
__global__ void trapezoidalIntegralCUDA(float *sum, int n, float a, float h,
                                        int terms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = 0.0;
    if (i < n) {
        x = a + i * h;
        // 这里没有用函数指针，用函数指针会出错，原因未知
        sum[i] = calculateSine(x, terms);
    }
}

void test01(int steps, int terms) {
    float h = PI / (steps - 1); // 计算步长 (b-a)/(n-1)
    int threads = 1024;
    int blocks = (steps + threads - 1) / threads; // 计算块数，向上取整

    thrust::device_vector<float> dsums(steps); // 保存每个线程的结果
    float *dptr = thrust::raw_pointer_cast(&dsums[0]); // 获取指针

    auto start = high_resolution_clock::now();

    // 使用CUDA核函数计算梯形积分
    trapezoidalIntegralCUDA<<<blocks, threads>>>(dptr, steps, 0.0f, h, terms);

    double sum =
        thrust::reduce(dsums.begin(), dsums.end()); // 汇总每个线程的结果

    sum -= 0.5 * (calculateSine(0.0f, terms) + calculateSine(PI, terms));
    sum *= h;

    auto end = high_resolution_clock::now();
    duration<double, std::milli> duration = end - start;

    printf("计算结果：%.10f\n", sum);
    cout << "steps = " << steps << ", terms = " << terms << ", 执行时间："
         << duration.count() << "毫秒" << endl;
}

/**
 * @brief 计时器函数，用于计算一个函数的执行时间
 * @param fun 待计时的函数，需要接受五个参数：
 *            第一个为节点个数
 *            第二个和第三个分别为积分下限和上限
 *            第四个为被积函数
 *            第五个为被积函数泰勒级数的前 terms 项
 *
 * @param cal   函数的第二个输入参数，被积函数
 * @param steps 函数的第三个输入参数，复合梯形积分公式的积分点数
 * @param terms 函数的第四个输入参数，被积函数泰勒级数的前 terms 项
 */
void Timer(double (*fun)(int, float, float, float (*)(float, int), int),
           float (*cal)(float, int), int steps, int terms) {
    // 启动计时器
    auto start = high_resolution_clock::now();

    // 调用函数
    double result = fun(steps, 0.0f, PI, cal, terms);

    // 停止计时器
    auto end = high_resolution_clock::now();

    // 计算持续时间
    duration<double, std::milli> duration = end - start;

    // 打印执行时间
    printf("计算结果：%.10f\n", result);
    cout << "steps = " << steps << ", terms = " << terms << ", 执行时间："
         << duration.count() << "毫秒" << endl;
}
