import numpy as np
import matplotlib.pyplot as plt


def gauss_vector(mu_list: float, sigma_list: float, dim: int = 2) -> list:
    """生成一个向量，其各个维度服从正态分布

    Args:
        mu_list:    正态分布均值，其中 mu_list[i] 是第 i 维的均值
        sigma_list: 正态分布方差，其中 sigma_list[i] 是第 i 维的方差
        dim:        向量维数

    Returns:
        返回向量
    """

    vector = []
    for i in range(0, dim):
        vector.append(np.random.normal(mu_list[i], sigma_list[i], 1)[0])

    return vector


def raw_data(count: int, mu_list: list, sigma_list: list, dim: int = 2) -> list:
    """生成 count 组向量
    向量的维度是 dim 维，各个维度服从正态分布，
    其中 mu_list[i] 是第 i 维的均值，sigma_list[i] 是第 i 维的方差

    Args:
        count:      向量组数
        mu_list:    正态分布均值
        sigma_list: 正态分布方差
        dim:        向量维数

    Returns:
        返回一个 list，包含 count 个向量
    """

    vector_list = []
    for i in range(0, count):
        vector_list.append(gauss_vector(mu_list, sigma_list, dim))

    return vector_list


def raw_data_dim2(count: int, mu: float, sigma: float, weight: float) -> list:
    """生成 count 组 2 维向量，
    其 x 维度服从正态分布，均值是 mu，方差是 sigma；
    其 y 维度与 x 维度的和是 (weight * 一个 (0, 1) 内的随机数)

    Args:
        count:  向量组数
        mu:     正态分布均值
        sigma:  正态分布方差
        weight: 两个维度的和的权重

    Returns:
        返回一个 list，包含 count 个 2 维向量
    """

    vector_list = []
    for i in range(0, count):
        vector = [np.random.normal(mu, sigma, 1)[0]]
        vector.append(weight * np.random.random() - vector[0])
        vector_list.append(vector)

    return vector_list


def draw_line(w: list, start: int, stop: int, label: str, color: str) -> None:
    """绘制直线
    w = [w0, w1, w2]，绘制的直线为 w1 * x + w2 * y + w0 = 0，
    即 y = - 1/w2 * (w1 * x + w0)


    Args:
        w:     直线方程参数
        start: 起始横坐标
        stop:  终止横坐标
        label: 图例
        color: 颜色
    """

    x_list = np.arange(start, stop, step=0.01)
    y_list = (-1 / w[2]) * (w[1] * x_list + w[0])

    plt.plot(x_list, y_list, c=color, label=label)


def insert_one(vectors: list) -> list:
    """向一组向量中，每个向量的首部都加一个 1
    [
        [1, x1, x2, ...],
        ...,
        [1, z1, z2, ...],
    ]

    Args:
        vectors: 一个向量组成的 list

    Returns:
        返回添加 1 的向量组
    """

    new_vectors = []
    for vec in vectors:
        new_vectors.append([1] + vec)

    return new_vectors


def select_dim(vectors: list, dim: int) -> list:
    """将一组向量中的第 dim 维的坐标提取出来，放入一个 list

    Args:
        vectors: 向量
        dim:     需要提取的维数

    Returns:
        返回第 dim 维坐标组成的列表
    """

    index_list = []
    for vec in vectors:
        index_list.append(vec[dim])

    return index_list


def tidy_data(vectors: list, classes: list) -> list:
    """整理数据，生成的格式为：
    [
        [vector0, class],
        [vector1, class],
        ...,
        [vectorn, class]
    ]
    其中每个 class 为对应 vector 的类别号

    Args:
        vectors: 向量列表
        classes: 类别列表，classes[i] 是 vectors[i] 的类别

    Returns:
        返回整理后的数据
    """

    data = []
    for i in range(0, len(classes)):
        data.append([vectors[i], classes[i]])

    return data


def calc_loss(data: list, w: list) -> float:
    """计算分类面的误差
    误差定义为：类别 0 的点，分到类别 1 的个数 / 类别 0 的点的总个数

    Args:
        data: 训练数据
        w:    分类面系数

    Returns:
        返回误差
    """

    class0_count = 0
    error_count = 0
    for vec in data:
        X = vec[0]
        Y = vec[1]
        if Y == 0:
            # 统计类别 0 的点的总个数
            class0_count += 1
        if np.dot(np.transpose(w), X) < 0 and Y == 1:
            # X 被分到类别 0，但是实际上 X 是类别 1
            error_count += 1

    return error_count / class0_count


def gd(
    data: list, dim: int, lam: float, alpha: float = 0.01, max_turn: int = 10000
) -> list:
    """梯度下降法求分类面系数

    Args:
        data:     分类的数据列表
        dim:      向量维度数
        alpha:    学习率
        lam:      正则项系数
        max_turn: 最大迭代次数

    Returns:
        返回分类面的系数
    """

    turn = 0

    w = np.ones(dim + 1)
    while True:
        sum_l = 0
        for vec in data:
            X = vec[0]
            Y = vec[1]
            wx = np.dot(np.transpose(w), X)
            if wx >= 0:
                expwx = np.exp(wx)
                if expwx == np.inf:
                    sum_l += (Y - 1) * np.array(X)
                else:
                    sum_l += (Y - expwx / (1 + expwx)) * np.array(X)
            else:
                # e^x / (1 + e^x) = 1 / (e^(-x) + 1)
                expnwx = np.exp(-wx)
                if expnwx == np.inf:
                    sum_l += Y * np.array(X)
                else:
                    sum_l += (Y - 1 / (expnwx + 1)) * np.array(X)
        w = w + alpha * (sum_l - lam * w)

        turn += 1
        if turn >= max_turn:
            print("===== hit max_turn limit =====")
            print("loss = " + str(calc_loss(data, w)))
            break

    print("turn = " + str(turn))
    return w


def load_data(filename: str, dim: int) -> list:
    """从文件中加载数据
    文件格式必须为
    x0,x1,...,xn,c0
    ...,
    x0,x1,...,xn,cm
    其中每一行的前 dim 个数据为训练数据向量的各个维度，最后一个数据为此行数据的类别

    Args:
        filename: 文件名
        dim:      数据维度

    Returns:
        返回加载的数据
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            data_line = [float(i) for i in line.removesuffix("\n").split(",")]
            vec = data_line[:-1]
            class_id = int(data_line[-1])
            result.append([[1] + vec, class_id])

    return result


if __name__ == "__main__":
    # 从文件中加载数据
    # data = load_data("lab2/data_banknote_authentication.txt", 4)
    # w = gd(data, dim=4, lam=0)
    # calc_loss(data, w)
    # exit(0)

    class0_count = 20
    class1_count = 20

    # 满足朴素贝叶斯假设的原始数据
    class0_vector = raw_data(class0_count, [1, 1], [1, 1])
    class1_vector = raw_data(class1_count, [3, 3], [1, 1])

    # 不满足朴素贝叶斯假设的原始数据
    # class0_vector = raw_data_dim2(class0_count, 1, 1, 10)
    # class1_vector = raw_data_dim2(class1_count, 3, 1, 10)

    # 对原始向量进行增广
    class0_vector_new = insert_one(class0_vector)
    class1_vector_new = insert_one(class1_vector)

    data = tidy_data(
        class0_vector_new + class1_vector_new,
        [0] * class0_count + [1] * class1_count,
    )

    # ===== 绘图 =====

    plt.title("Lab 2")

    # 类别 0
    class0_x_list = select_dim(class0_vector, 0)
    class0_y_list = select_dim(class0_vector, 1)
    plt.scatter(class0_x_list, class0_y_list, label="class 0")

    # 类别 1
    class1_x_list = select_dim(class1_vector, 0)
    class1_y_list = select_dim(class1_vector, 1)
    plt.scatter(class1_x_list, class1_y_list, label="class 1")

    # 分类面
    w = gd(data, dim=2, lam=0)
    draw_line(w, start=-1, stop=5, label="gd", color="g")
    w = gd(data, dim=2, lam=0.01)
    draw_line(w, start=-1, stop=5, label="gd lambda", color="r")

    plt.legend()
    plt.show()
