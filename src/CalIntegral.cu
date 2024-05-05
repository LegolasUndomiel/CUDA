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

/**
 * 计算sin(x)的近似值，使用泰勒级数的前n项和作为近似结果。
 *
 * @param x 输入的值，用于计算sin(x)
 * @param n 指定使用泰勒级数的前n项和作为近似结果
 * @return 返回sin(x)的近似值
 */
__host__ __device__ inline float calculateSine(float, int = 1000);

/**
 * 计算cos(x)的近似值，使用泰勒级数的前n项和作为近似结果。
 *
 * @param x 输入的值，用于计算cos(x)
 * @param n 指定使用泰勒级数的前n项和作为近似结果
 * @return 返回cos(x)的近似值
 */
__host__ __device__ inline float calculateCosine(float, int = 1000);

/**
 * 梯形积分（Trapezoidal rule）求解积分
 * 计算范围为[a,b]，步长为(b-a)/(n-1)，积分点个数为n。
 * @param n 积分点个数
 * @param a 积分下限
 * @param b 积分上限
 * @param fun 被积函数
 * @param terms 被积函数泰勒级数的项数，默认为1000
 */
double trapezoidalIntegral(int n, float a, float b, float (*fun)(float, int),
                           int terms = 1000);

/**
 * 多线程版(OpenMP)梯形积分（Trapezoidal rule）求解积分
 * 计算范围为[a,b]，步长为(b-a)/(n-1)，积分点个数为n。
 * @param n 积分点个数
 * @param a 积分下限
 * @param b 积分上限
 * @param fun 被积函数
 * @param terms 被积函数泰勒级数的项数，默认为1000
 */
double trapezoidalIntegralOMP(int n, float a, float b, float (*fun)(float, int),
                              int terms = 1000);

/**
 * 使用OpenMP并行化计算积分，使用数组保存每个线程的结果，最后汇总总和。
 *
 * @param n 积分点个数
 * @param a 积分下限
 * @param b 积分上限
 * @param fun 被积函数
 * @param terms 被积函数泰勒级数的项数，默认为1000
 */
double trapezoidalIntegralOMPArray(int n, float a, float b,
                                   float (*fun)(float, int), int terms);

__global__ void trapezoidalIntegralCUDA(float *, int, float, float,
                                        float (*fun)(float, int), int = 1000);

/**
 * 计时器函数，用于计算一个函数的执行时间
 *
 * @param fun 待计时的函数，需要接受五个参数：
 *            第一个为计算结果
 *            第二个为节点个数
 *            第三个和第四个分别为积分下限和上限
 *            第五个为被积函数
 *            第六个为被积函数泰勒级数的前 terms 项
 *
 * @param cal   函数的第二个输入参数，被积函数
 * @param steps 函数的第三个输入参数，复合梯形积分公式的积分点数
 * @param terms 函数的第四个输入参数，被积函数泰勒级数的前 terms 项
 */
void Timer(double (*fun)(int, float, float, float (*)(float, int), int),
           float (*cal)(float, int), int steps, int terms);

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

    return 0;
}

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

inline float calculateCosine(float x, int n) {
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

double trapezoidalIntegral(int n, float a, float b, float (*fun)(float, int),
                           int terms) {
    double sum = 0;               // 初始化和
    double h = (b - a) / (n - 1); // 步长
    float x = 0.0;                // 初始化x

    for (int i = 0; i < n; i++) {
        x = a + i * h;        // 更新x
        sum += fun(x, terms); // 累加梯形面积
    }

    // Trapezoidal rule = (h/2)*(f(a) + 2*f(a+h) + ... + 2*f(a+(n-1)*h) +
    // f(b))
    sum -= 0.5 * (fun(a, terms) + fun(b, terms));
    sum *= h;

    return sum; // 返回积分结果
}

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

double trapezoidalIntegralOMPArray(int n, float a, float b,
                                   float (*fun)(float, int), int terms) {
    double sum = 0;               // 初始化和
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
