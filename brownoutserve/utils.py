import random


def generate_random_int_list(n, m, min_val, max_val):

    return [[random.randint(min_val, max_val) for _ in range(m)] for _ in range(n)]

def calculate_average_and_p90(arr):
    """
    此函数用于计算给定float类型数组的平均值和P90值
    :param arr: 输入的float类型数组
    :return: 包含平均值和P90值的元组
    """
    # 计算平均值
    average = sum(arr) / len(arr)
    # 对数组进行排序
    sorted_arr = sorted(arr)
    # 计算P90值的索引
    index = int(len(sorted_arr) * 0.9)
    # 获取P90值
    p90 = sorted_arr[index]
    return average, p90

