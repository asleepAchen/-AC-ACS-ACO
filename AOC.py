import numpy as np
import plotly.graph_objects as go
from scipy.io import loadmat
data = loadmat('citys_data.mat')  # 请替换为实际文件路径
citys = data['citys']
D = data['D']
m = 50  # 蚂蚁数量
alpha = 1  # 信息素重要程度因子
beta = 5  # 启发函数重要程度因子
rho = 0.2  # 信息素挥发因子
Q = 1  # 常系数
Eta = 1. / D  # 启发函数（适应性函数的倒数）
Tau = np.ones((n, n))  # 信息素矩阵
Table = np.zeros((m, n), dtype=int)  # 路径记录表
iter_max = 150  # 最大迭代次数
Route_best = np.zeros((iter_max, n), dtype=int)  # 各代最佳路径
Length_best = np.zeros(iter_max)  # 各代最佳路径的长度
Length_ave = np.zeros(iter_max)  # 各代路径的平均长度
np.random.seed(42)  # 为了结果的可重复性
shortest_paths_every_30_iters = []  # 保存每30次迭代的最短路径信息
iter_intervals = 30  # 每30次迭代绘制一次

for iter in range(iter_max):
    # 随机产生各个蚂蚁的起点城市
    start = np.random.choice(n, m, replace=True)
    Table[:, 0] = start
    for i in range(m):
        for j in range(1, n):
            tabu = Table[i, :j]  # 已访问的城市集合
            allow = np.setdiff1d(np.arange(n), tabu)  # 待访问的城市集合
            P = (Tau[tabu[-1], allow] ** alpha) * (Eta[tabu[-1], allow] ** beta)
            P /= np.sum(P)
            next_city = np.random.choice(allow, p=P)
            Table[i, j] = next_city

    # 计算路径长度
    Length = np.zeros(m)
    for i in range(m):
        route = Table[i]
        Length[i] = np.sum(D[route[:-1], route[1:]]) + D[route[-1], route[0]]

    # 更新最短路径和平均路径长度
    if iter == 0 or np.min(Length) < Length_best[iter - 1]:
        Length_best[iter] = np.min(Length)
        Route_best[iter] = Table[np.argmin(Length)]
    else:
        Length_best[iter] = Length_best[iter - 1]
        Route_best[iter] = Route_best[iter - 1]
    Length_ave[iter] = np.mean(Length)

    # 更新信息素
    Delta_Tau = np.zeros((n, n))
    for i in range(m):
        route = Table[i]
        for j in range(1, n):
            Delta_Tau[route[j-1], route[j]] += Q / Length[i]
        Delta_Tau[route[-1], route[0]] += Q / Length[i]  # 封闭路径
    Tau = (1 - rho) * Tau + Delta_Tau

    # 每30次迭代后保存当前最短路径和迭代次数
    if (iter + 1) % iter_intervals == 0 or iter == iter_max - 1:
        shortest_paths_every_30_iters.append((iter + 1, Route_best[iter], Length_best[iter]))

# 这里我们假设之后会有代码来绘制保存的路径
for iter, route, length in shortest_paths_every_30_iters:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=citys[route, 0], y=citys[route, 1], mode='markers+lines', name=f'Iter: {iter} Length: {length}'))
    fig.update_layout(title=f'蚁群算法优化路径 {iter} 迭代',
                      xaxis_title='城市位置横坐标',
                      yaxis_title='城市位置纵坐标')
    fig.show()  # 在本地环境中解除注释以显示图形
shortest_length = np.min(Length_best)
shortest_route = Route_best[np.argmin(Length_best)]

print(f'最短距离: {shortest_length}')
print(f'最短路径: {shortest_route + 1}')  # 转换为1-based索引以匹配MATLAB输出


shortest_route_1based = shortest_route + 1

# 创建图形对象
fig = go.Figure()

# 添加城市点
fig.add_trace(go.Scatter(x=citys[shortest_route, 0], y=citys[shortest_route, 1],
                         mode='markers+text', name='Cities',
                         text=shortest_route_1based, textposition="bottom center",
                         marker=dict(size=8, color='blue')))

# 添加路径线
fig.add_trace(go.Scatter(x=np.append(citys[shortest_route, 0], citys[shortest_route[0], 0]),
                         y=np.append(citys[shortest_route, 1], citys[shortest_route[0], 1]),
                         mode='lines', name='Path',
                         line=dict(color='red')))

# 更新布局
fig.update_layout(title=f'蚁群算法优化路径(最短距离: {shortest_length:.2f})',
                  xaxis_title='城市位置横坐标',
                  yaxis_title='城市位置纵坐标',
                  showlegend=False)

# 显示图形
fig.show()

# 创建折线图对象
fig = go.Figure()

# 添加最短路径长度的折线图
fig.add_trace(go.Scatter(x=np.arange(1, iter_max + 1), y=Length_best,
                         mode='lines+markers', name='最短路径长度',
                         line=dict(color='blue')))

# 添加平均路径长度的折线图
fig.add_trace(go.Scatter(x=np.arange(1, iter_max + 1), y=Length_ave,
                         mode='lines+markers', name='平均路径长度',
                         line=dict(color='red')))

# 更新布局
fig.update_layout(title='每次迭代最短路径和平均路径的折线图',
                  xaxis_title='迭代次数',
                  yaxis_title='路径长度',
                  legend_title='路径类型')

# 显示图形
fig.show()