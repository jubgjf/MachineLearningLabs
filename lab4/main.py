import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pm


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


def center(data: list, data_count: int) -> tuple:
    """对数据列表进行均值处理：
    对每一个数据，都减去他们的均值向量，成为新的数据向量

    Args:
        data:       数据集列表
        data_count: 数据数量

    Returns:
        返回 (处理后的数据集, 原数据集的均值向量)
    """

    # 训练样本向量均值
    data_center: list = np.mean(data, axis=0)

    centered_data = []
    for d in data:
        centered_data.append(d - data_center)

    return centered_data, data_center


def pca(data: list, data_count: int, dim: int, dim_down: int) -> np.array:
    """使用 PCA 算法进行降维

    Args:
        data:       训练数据
        data_count: 数据量
        dim:        数据向量的维度
        dim_down:   降维后的维度

    Returns:
        返回按照特征值从大到小排序的特征向量矩阵，矩阵中的特征向量是列向量
    """

    centered_data, _ = center(data, data_count)

    # 样本协方差矩阵
    cov = np.zeros((dim, dim))
    for i in range(0, data_count):
        cov += np.dot(
            centered_data[i].reshape((-1, 1)),
            np.transpose(centered_data[i]).reshape((1, -1)),
        )
    cov /= data_count

    # 求特征向量和特征值
    lams, vectors = np.linalg.eig(cov)  # vectors 是列向量

    # 根据特征值对特征向量进行排序
    index = np.argsort(lams)  # 从小到大排序后的下标序列
    vectors = vectors[:, index[: -(dim_down + 1) : -1]]  # 把序列逆向排列然后取前k个

    return vectors


def dim2(data_count: int, mu: list, sigma: list) -> None:
    """使用 PCA 算法对二维数据进行降维

    Args:
        data_count: 样本数据数量
        mu:         样本数据均值
        sigma:      样本数据协方差矩阵
    """

    # ===== 初始参数 =====
    data: list = []
    dim: int = 2

    # ===== 生成训练数据 =====
    for i in range(0, data_count):
        data.append(np.random.multivariate_normal(mu, sigma))
    xs, ys = vectors2xylist(data)
    plt.scatter(xs, ys, label="origin points")

    # ===== 降维 =====
    vectors = pca(data, data_count, dim, 1 + 1)
    xs = [i for i in range(-2, 3)]
    ys = []
    for x in range(-2, 3):
        ys.append(vectors[1][0] / vectors[1][1] * x)
    plt.plot(xs, ys, c="g", label="low dim line")

    # 画出降维后的点
    low_data = []
    vec = vectors[:1]
    for i in data:
        low_data.append(np.dot(i, vectors.T[0]) * vectors.T[0])
    xs, ys = vectors2xylist(low_data)
    plt.scatter(xs, ys, c="r", label="low dim points")

    # ===== 旋转 =====
    vectors = pca(data, data_count, dim, dim)
    xs = [i for i in range(0, 2)]
    ys = []
    for x in range(0, 2):
        ys.append(vectors[0][0] / vectors[0][1] * x)
    plt.plot(xs, ys, c="b", label="new axis 0")
    xs = [i for i in range(0, 2)]
    ys = []
    for x in range(0, 2):
        ys.append(vectors[1][0] / vectors[1][1] * x)
    plt.plot(xs, ys, c="b", label="new axis 1")

    # ===== 绘图 =====
    plt.axis("scaled")  # 绘制正方形，长宽比例相同


def face(img_path: str, low_dim: int) -> None:
    """使用 PCA 算法对人脸数据进行降维

    Args:
        img_path: 图片文件路径
        low_dim:  降维到的维度
    """

    # ===== 读取文件 =====
    face = np.array(pm.open(img_path).convert("L"))  # 灰度图

    # ===== PCA =====
    vectors = pca(face, face.shape[0], face.shape[1], low_dim)
    centered_data, data_center = center(face, face.shape[0])

    # ===== 重建数据 =====
    pca_face = (
        np.dot((np.dot(np.array(centered_data), vectors)), vectors.T) + data_center
    )

    # ===== 信噪比 =====
    noise = np.sqrt(np.mean((face - pca_face) ** 2))
    snr = 20 * np.log10(255 / noise)

    # ===== 绘图 =====
    plt.imshow(pca_face)
    plt.title("dim = " + str(low_dim) + "\nSNR = " + str(snr))


if __name__ == "__main__":
    dim2(100, mu=[0, 0], sigma=[[10, 3], [3, 1]])
    # face("lab4/img/2.jpg", 10)

    # ===== 绘图 =====
    plt.legend()
    plt.show()
