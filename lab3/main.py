import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


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


def vectors2xylist(vectors: list) -> tuple:
    """转换向量列表格式
    将
    [
        [x0, y0],
        [x1, y1],
        ...,
        [xn, yn],
    ]
    格式的 vectors 转换为
    [x0, x1, ... xn], [y0, y1, ... yn]
    两个列表

    Args:
        vectors:  待转换向量列表

    Returns:
        返回向量列表中各组向量的 x 坐标和 y 坐标
    """

    xs = []
    ys = []
    for vec in vectors:
        xs.append(vec[0])
        ys.append(vec[1])

    return xs, ys


def k_means(train_data: list, cluster_count: int, dim: int = 2) -> tuple:
    """使用 k-means 算法进行分簇

    Args:
        train_data:    原始未分簇的数据列表
        cluster_count: 需要分几个簇
        dim:           每个数据的维度

    Returns:
        (分好的簇列表, 各个簇的中心点列表)
    """

    # 各个簇的中心点列表
    avg_list = []
    for i in range(0, cluster_count):
        avg_list.append(np.random.random((dim)))

    # 各个簇
    cluster_list = [train_data]
    for i in range(1, cluster_count):
        cluster_list.append([])

    limit = 1e-5
    while True:
        # 重新分簇
        cluster_list_new = []
        for i in range(0, cluster_count):
            cluster_list_new.append([])

        for cluster in cluster_list:
            for vec in cluster:
                # 到各个簇中心点的距离
                distance_list = []
                for i in range(0, cluster_count):
                    distance_list.append(np.linalg.norm(np.array(vec) - avg_list[i]))
                for i in range(0, cluster_count):
                    if distance_list[i] == min(distance_list):
                        cluster_list_new[i].append(vec)
        cluster_list = cluster_list_new

        # 重新计算簇的中心点
        avg_list_new = []
        for i in range(0, cluster_count):
            avg_list_new.append(np.mean(np.array(cluster_list[i]), axis=0))

        # 当所有中心点变化都不大时，判断为收敛
        finished_cluster = 0  # 变化不大的中心点个数
        for i in range(0, cluster_count):
            if np.linalg.norm(avg_list_new[i] - avg_list[i]) < limit:
                finished_cluster += 1
        if finished_cluster != cluster_count:
            avg_list = avg_list_new
        else:
            break

    return cluster_list, avg_list


def e_step(
    cluster_count: int,
    data_count: int,
    data: list,
    mu_list: list,
    sigma_list: list,
    pi_list: list,
) -> np.matrix:
    """GMM-EM 算法的 E-步骤：求 gamma_z 矩阵

    Args:
        cluster_count: 簇的个数
        data_count:    数据的个数
        data:          数据集
        mu_list:       各个簇高斯分布的均值列表
        sigma_list:    各个簇高斯分布的协方差矩阵列表
        pi_list:       各个簇高斯分布的权重列表

    Returns:
        返回 gamma_z 矩阵
    """

    gamma_z = np.zeros((data_count, cluster_count))
    for i in range(0, data_count):
        pi_gauss_sum = 0
        pi_gauss = np.zeros(cluster_count)
        for k in range(0, cluster_count):
            pi_gauss[k] = pi_list[k] * multivariate_normal.pdf(
                data[i], mu_list[k], sigma_list[k]
            )
            pi_gauss_sum += pi_gauss[k]
        for k in range(0, cluster_count):
            gamma_z[i][k] = pi_gauss[k] / pi_gauss_sum

    return gamma_z


def m_step(
    cluster_count: int,
    data_count: int,
    data: list,
    mu_list: list,
    sigma_list: list,
    pi_list: list,
    gamma_z: np.matrix,
    dim: int,
) -> tuple:
    """GMM-EM 算法的 M-步骤：更新 mu_list, sigma_list, pi_list

    Args:
        cluster_count: 簇的个数
        data_count:    数据的个数
        data:          数据集
        mu_list:       各个簇高斯分布的均值列表
        sigma_list:    各个簇高斯分布的协方差矩阵列表
        pi_list:       各个簇高斯分布的权重列表
        gamma_z:       gamma_z 矩阵
        dim:           数据的维度

    Returns:
        返回更新的 (mu_list, sigma_list, pi_list)
    """

    for k in range(0, cluster_count):
        # gamma_z 矩阵第 k 列的和
        sum_gamma_z = np.sum(gamma_z[:, k])

        # 更新 mu
        sum_gamma_z_x = np.zeros(dim)
        for i in range(0, data_count):
            sum_gamma_z_x += gamma_z[i][k] * data[i]
        mu_list[k] = sum_gamma_z_x / sum_gamma_z

        # 更新 sigma
        sum_gamma_z_x_mu = np.zeros((dim, dim))
        for i in range(0, data_count):
            sum_gamma_z_x_mu += gamma_z[i][k] * np.dot(
                data[i] - mu_list[k], np.transpose(data[i] - mu_list[k])
            )

        # 更新 pi
        pi_list[k] = sum_gamma_z / data_count

    return mu_list, sigma_list, pi_list


def gmm_em(
    train_data: list, train_data_count: int, cluster_count: int, dim: int = 2
) -> tuple:
    """使用 GMM-EM 算法进行分簇

    Args:
        train_data:       原始未分簇的数据列表
        train_data_count: 数据数量
        cluster_count:    需要分几个簇
        dim:              每个数据的维度

    Returns:
        (各个簇的高斯分布的均值, 各个簇的高斯分布的协方差, 各个簇的高斯分布的权重)
    """

    # 初始化参数
    mu_list = []
    sigma_list = []
    pi_list = []
    for i in range(0, cluster_count):
        mu_list.append(np.random.random((dim)))
        sigma_list.append(np.diag([1] * dim))
        pi_list.append(1 / cluster_count)

    ln_likehood_old = -np.inf
    turn = 0
    max_turn = 100  # 最大迭代次数
    while True:
        turn += 1

        # E 步骤
        gamma_z = e_step(
            cluster_count,
            train_data_count,
            train_data,
            mu_list,
            sigma_list,
            pi_list,
        )

        # M 步骤
        mu_list, sigma_list, pi_list = m_step(
            cluster_count,
            train_data_count,
            train_data,
            mu_list,
            sigma_list,
            pi_list,
            gamma_z,
            dim,
        )

        # 似然函数判断收敛
        ln_likehood = 0
        for i in range(0, train_data_count):
            px = 0
            for k in range(0, cluster_count):
                px += pi_list[k] * multivariate_normal.pdf(
                    train_data[i], mu_list[k], sigma_list[k]
                )
            ln_likehood += np.log(px)
        print("\r[", end="")
        print("#" * int(20 * turn / max_turn), end="")
        print("." * (20 - int(20 * turn / max_turn)), end="")
        print("]", end="")
        print("[GMM-EM] ln(likehood) =", str(ln_likehood), end="")
        if ln_likehood - ln_likehood_old < 1e-5 or turn >= max_turn:
            print("\r[" + "#" * 20 + "]", end="")
            print("[GMM-EM] ln(likehood) =", str(ln_likehood))
            if turn >= max_turn:
                print("===== hit max turn =====")
            break
        else:
            ln_likehood_old = ln_likehood

    return mu_list, sigma_list, pi_list


def gmm_em_do_cluster(
    cluster_count: int,
    data_count: int,
    data: list,
    mu_list: list,
    sigma_list: list,
    pi_list: list,
) -> list:
    """对 GMM-EM 算法处理后的数据进行聚类

    Args:
        cluster_count: 簇的个数
        data_count:    数据的个数
        data:          数据集
        mu_list:       各个簇高斯分布的均值列表
        sigma_list:    各个簇高斯分布的协方差矩阵列表
        pi_list:       各个簇高斯分布的权重列表

    Returns:
        返回聚类列表
    """

    gamma_z = e_step(
        cluster_count,
        data_count,
        data,
        mu_list,
        sigma_list,
        pi_list,
    )

    cluster_list = []
    for i in range(0, cluster_count):
        cluster_list.append([])
    for i in range(0, data_count):
        cluster_index = list(gamma_z[i]).index(max(gamma_z[i]))
        cluster_list[cluster_index].append(data[i])

    return cluster_list


def gmm_em_test(
    cluster_count: int,
    test_data_count: int,
    mu_list: list,
    sigma_list: list,
    pi_list: list,
    test_mu: list,
    test_sigma: list,
) -> list:
    """对 GMM-EM 算法聚类的效果进行测试

    Args:
        cluster_count:   簇的个数
        test_data_count: 测试数据的个数
        mu_list:         各个簇高斯分布的均值列表
        sigma_list:      各个簇高斯分布的协方差矩阵列表
        pi_list:         各个簇高斯分布的权重列表
        test_mu:         生成测试数据的均值
        test_sigma:      生成测试数据的协方差矩阵

    Returns:
        返回聚类后，各个簇的元素个数
    """

    test_cluster = []
    for i in range(0, test_data_count):
        test_cluster.append(np.random.multivariate_normal(test_mu, test_sigma))
    test_cluster_list = gmm_em_do_cluster(
        cluster_count,
        test_data_count,
        test_cluster,
        mu_list,
        sigma_list,
        pi_list,
    )

    return [len(i) for i in test_cluster_list]


def draw_clusters(
    title: str, cluster_list: list, avg_list: list, new_figure: bool = True
) -> None:
    """绘制聚类散点图

    Args:
        title:        表标题
        cluster_list: 聚类列表
        avg_list:     各聚类的中心点列表
        new_figure:   是否需要新建一张图
    """

    if new_figure:
        plt.figure()
    plt.title(title)
    for i in range(0, len(cluster_list)):
        xs, ys = vectors2xylist(cluster_list[i])
        plt.scatter(xs, ys)
        plt.scatter(avg_list[i][0], avg_list[i][1], c="b", s=100)


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
        返回加载的数据，格式为
        [
            [[x0, x1, ..., xn], c0],
            ...,
            [[x0, x1, ..., xn], c0],
        ]
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            data_line = [float(i) for i in line.removesuffix("\n").split(",")]
            vec = data_line[:-1]
            class_id = int(data_line[-1])
            result.append([vec, class_id])

    return result


def uci() -> None:
    """运行 UCI 数据集"""

    uci_data_dim = 4
    uci_cluster_count = 7
    uci_train_data_raw = load_data("lab3/iris.data", uci_data_dim)
    uci_test_data_raw = load_data("lab3/iris_test.data", uci_data_dim)

    uci_train_data = [np.array(i[0]) for i in uci_train_data_raw]
    uci_train_data_count = len(uci_train_data)
    mu_list, sigma_list, pi_list = gmm_em(
        uci_train_data, uci_train_data_count, uci_cluster_count, uci_data_dim
    )

    uci_test_data = [np.array(i[0]) for i in uci_test_data_raw]
    uci_test_data_count = len(uci_test_data)
    cluster_list = gmm_em_do_cluster(
        uci_cluster_count,
        uci_test_data_count,
        uci_test_data,
        mu_list,
        sigma_list,
        pi_list,
    )
    print([len(i) for i in cluster_list])


def custom() -> None:
    """运行自定数据集"""

    # ===== 初始参数 =====
    dim = 2  # 数据维度
    cluster_count = 3  # 簇的个数

    # ===== 训练数据 =====
    train_data = []
    train_data_count_per_cluster = 100  # 每个簇中的数据个数
    train_data_count = cluster_count * train_data_count_per_cluster  # 训练数据总个数
    for i in range(0, train_data_count_per_cluster):
        train_data.append(np.random.multivariate_normal([4, 4], [[4, 1], [1, 4]]))
        train_data.append(np.random.multivariate_normal([10, 10], [[1, 0], [0, 1]]))
        train_data.append(np.random.multivariate_normal([1, -7], [[3, 2], [2, 3]]))

    # ===== k-means =====
    cluster_list, avg_list = k_means(train_data, cluster_count, 2)
    draw_clusters("k-means", cluster_list, avg_list)

    # ===== GMM-EM =====
    mu_list, sigma_list, pi_list = gmm_em(
        train_data, train_data_count, cluster_count, dim
    )
    cluster_list = gmm_em_do_cluster(
        cluster_count, train_data_count, train_data, mu_list, sigma_list, pi_list
    )
    draw_clusters("GMM-EM", cluster_list, mu_list)

    # ===== 测试数据 =====
    test_data_count = 500
    print(
        "GMM-EM test result:",
        gmm_em_test(
            cluster_count,
            test_data_count,
            mu_list,
            sigma_list,
            pi_list,
            test_mu=[4, 4],
            test_sigma=[[4, 1], [1, 4]],
        ),
    )

    # ===== 绘图 =====
    plt.show()


if __name__ == "__main__":
    use_uci = False

    if use_uci:
        # ===== UCI 数据集 =====
        uci()
    else:
        # ===== 自定义数据集 =====
        custom()
