from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def gauss_noise(count: int, mu: float = 0, sigma: float = 0.1) -> list:
    """生成高斯噪声

    Args:
        count: 生成噪声的个数
        mu:    均值
        sigma: 标准差

    Returns:
        返回一个 list，包含 count 个噪声

    """

    return np.random.normal(mu, sigma, count)


def raw_data(
    start: float = 0, end: float = 1, step: float = 0.1
) -> Tuple[list, list, int]:
    """生成实验数据

    Args:
        start: x 轴起始坐标
        end:   x 轴终止坐标
        step:  生成 x 轴数据的步长

    Returns:
        返回 [x 轴列表, 加噪声的 y 轴列表, len(x)]
    """

    x = np.arange(start, end, step)  # x 轴数据列表
    y = np.sin(2 * np.pi * x)  # y 轴数据列表，其中 y = sin(x)

    t = []  # t = sin(x) + noise
    count = len(y)
    noise = gauss_noise(count, sigma=0.5)
    for i in range(0, count):
        t.append(y[i] + noise[i])

    N = len(x)

    return x, t, N


def calc_ploy(x: float, w: list, m: int) -> float:
    """多项式函数

    y(x) = w_0 + w_1 * x + ... + w_m * x ** m

    Args:
        x: 自变量
        w: 系数列表
        m: 多项式函数的最高次数

    Returns:
        返回计算结果
    """

    y = 0
    for i in range(0, m + 1):
        y += w[i] * x ** i
    return y


def draw_lines(lines: dict, smooth=False) -> None:
    """绘制曲线

    Args:
        lines:  {<label>: [x, y], <label>: [x, y], ...}
        smooth: 是否绘制光滑曲线
    """

    for line in lines.items():
        label = line[0]
        x = line[1][0]
        y = line[1][1]
        if smooth:
            spl = make_interp_spline(x, y)
            x = np.linspace(x.min(), x.max())
            y = spl(x)
        plt.plot(x, y, label=label)


def draw_scatter(points: dict) -> None:
    """绘制散点

    Args:
        points: {<label>: [x, y], <label>: [x, y], ...}
    """

    for point in points.items():
        plt.scatter(point[1][0], point[1][1], label=point[0])


def cgm(A: np.matrix, b: np.array, size: int) -> list:
    """共轭梯度法求线性方程组 Ax = b 的解

    Args:
        A:    size * size 的矩阵
        b:    size * 1    的列向量
        size: 维数，即 b.shape[0]

    Returns:
        返回线性方程组的解 x
    """

    x = np.ones(size)
    r_old = b - np.dot(A, x)
    p = r_old.copy()
    for i in range(0, 10):
        a = np.dot(r_old.T, r_old) / np.dot(np.dot(p.T, A), p)
        x = x + np.dot(a, p)
        r_new = r_old - np.dot(np.dot(a, A), p)
        if np.dot(r_new.T, r_new) < 1e-10:
            break
        bb = np.dot(r_new.T, r_new) / np.dot(r_old.T, r_old)
        p = r_new + np.dot(bb, p)
        r_old = r_new

    return x


def gen_ploy(x: list, T: list, N: int, m: int, lam: float = 0, method: int = 0) -> list:
    """根据数据点，用最小二乘法生成拟合的多项式函数

    y(x) = w_0 + w_1 * x + ... + w_m * x ** m

    Args:
        x:      原始数据 x 坐标
        T:      原始数据 y 坐标
        N:      len(x)
        m:      多项式函数的最高次数
        lam:    正则项系数
        method: 求出多项式函数系数的方法
                    0: 解析解法
                    1: 梯度下降法
                    2: 共轭梯度法

    Returns:
        返回多项式函数的系数列表 w
    """

    # 生成矩阵 X 及其转置 XT
    XT = []
    for i in range(0, m + 1):
        for j in range(0, N):
            XT.append(x[j] ** i)
    XT = np.reshape(XT, (m + 1, N))
    X = np.transpose(np.array(XT))

    if method == 0:
        # 解析解: (XT * X + lambda * E) * w = XT * T
        w = np.dot(
            np.linalg.inv(np.dot(XT, X) + lam * np.identity(m + 1)), np.dot(XT, T)
        )
    elif method == 1:
        # 梯度下降
        alpha = 0.01  # 学习率
        turn = 10000  # 迭代次数
        w = np.ones(m + 1)
        for i in range(0, turn):
            w = w - alpha * (
                np.dot(np.dot(XT, X), w)
                - np.dot(XT, T)
                + lam * np.dot(np.identity(m + 1), w)
            )
    elif method == 2:
        # 共轭梯度
        w = cgm(np.dot(XT, X) + lam * np.identity(m + 1), np.dot(XT, T), m + 1)

    return w


def draw_ploy(
    x: list, T: list, N: int, m: int, lam: float = 0, method: int = 0
) -> None:
    """绘制拟合的多项式函数曲线

    Args:
        x:      原始数据 x 坐标
        T:      原始数据 y 坐标
        N:      len(x)
        m:      多项式函数的最高次数
        lam:    正则项系数
        method: 求出多项式函数系数的方法
                    0: 解析解法
                    1: 梯度下降法
                    2: 共轭梯度法
    """

    w = gen_ploy(x, T, N, m, lam, method)
    np.set_printoptions(suppress=True)  # 取消科学计数法打印
    print("w = ", end="")
    print(w)
    y_guess = []
    for i in range(0, N):
        y_guess.append(calc_ploy(x[i], w, m))

    draw_lines(
        {
            "y(x, w), m = "
            + str(m)
            + ", lambda = "
            + str(lam)
            + ", method = "
            + str(method): (x, y_guess)
        },
        smooth=True,
    )


def calc_loss(x: list, T: list, N: int, m: int, w: list) -> float:
    """计算拟合误差

    Args:
        x: 原始数据 x 坐标
        T: 原始数据 y 坐标
        N: len(x)
        m: 多项式函数的最高次数
        w: 拟合的多项式系数向量

    Returns:
        返回误差
    """

    loss = 0
    for i in range(0, N):
        loss += (calc_ploy(x[i], w, m) - T[i]) ** 2

    return loss / 2


if __name__ == "__main__":
    # ===== 原始数据 =====

    x, T, N = raw_data(step=0.01)

    plt.title("Lab 1")
    x_sin = np.arange(0, 1, 0.1)
    y_sin = np.sin(2 * np.pi * x_sin)
    draw_lines({"sin(x)": (x_sin, y_sin)}, smooth=True)
    draw_scatter({"original data": (x, T)})

    # ===== 超参数 =====

    draw_ploy(x, T, N, m=50, lam=0, method=0)
    draw_ploy(x, T, N, m=50, lam=0, method=2)

    # ===== 绘图 =====

    plt.legend()  # 开启图例
    plt.show()
