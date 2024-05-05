#include <chrono>
#include <iostream>

using namespace std::chrono;
using std::cout;
using std::endl;

/**
 * 计算sin(x)的近似值，使用泰勒级数的前n项和作为近似结果。
 *
 * @param x 输入的值，用于计算sin(x)
 * @param n 指定使用泰勒级数的前n项和作为近似结果
 * @return 返回sin(x)的近似值
 */
inline float calculateSine(float x, int n);

/**
 * 使用复合梯形积分公式计算数值积分
 *
 * @param n - 积分点个数
 * @param a - 积分下限
 * @param b - 积分上限
 * @param fun - 被积函数
 * @return 数值积分的结果
 */
double trapezoidalIntegral(int n, float a, float b, float (*fun)(float, int));

/**
 * 计时器函数，用于计算一个函数的执行时间
 *
 * @param fun 待计时的函数
 * @param x 函数的输入参数
 * @param n 函数的输入参数
 */
void Timer(float (*fun)(float, int), float x, int n);

int main(int argc, char const *argv[]) { return 0; }

inline float calculateSine(float x, int n) {
    float term = x;      // 初始化当前项
    float result = term; // 初始化近似结果
    float x2 = x * x;

    for (int i = 1; i < n; i++) {
        term *= -x2 / (float)((2 * i) * (2 * i + 1)); // 计算下一项
        result += term; // 累加当前项到近似结果
    }

    return result; // 返回近似值
}

void Timer(float (*fun)(float, int), float x, int n) {
    // 启动计时器
    auto start = high_resolution_clock::now();

    // 调用函数
    fun(x, n);

    // 停止计时器
    auto end = high_resolution_clock::now();

    // 计算持续时间
    duration<double, std::milli> duration = end - start;

    // 打印执行时间
    cout << "执行时间：" << duration.count() << "毫秒" << endl;
}

double trapezoidalIntegral(int n, float a, float b, float (*fun)(double)) {
    double h = (b - a) / n; // 步长
    double sum = 0;         // 初始化和

    // 计算每个梯形的高度和底基
    for (int i = 1; i <= n; i++) {
        double x = a + (i - 1) * h;              // 梯形左端点
        double xNext = a + i * h;                // 梯形右端点
        double height = fun(xNext);              // 梯形高度
        double base = fun(x);                    // 梯形底基
        double area = 0.5 * (height + base) * h; // 梯形面积
        sum += area;                             // 累加梯形面积
    }

    return sum; // 返回数值积分结果
}
