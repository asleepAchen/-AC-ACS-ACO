import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
import random


def read_tsp_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    name = None
    dimension = None
    node_coord_start = False
    coordinates = []

    for line in lines:
        if 'NAME' in line:
            name = line.split(':')[1].strip()
        elif 'DIMENSION' in line:
            dimension = int(line.split(':')[1].strip())
        elif 'NODE_COORD_SECTION' in line:
            node_coord_start = True
        elif node_coord_start:
            if 'EOF' in line:
                break
            coords = line.strip().split()
            coordinates.append((float(coords[1]), float(coords[2])))

    return dimension, np.array(coordinates), name
dimension, coordinates, name = read_tsp_file('kroA100.tsp')
n = dimension
m = 50  # 蚂蚁数量
alpha = 1.5
beta = 2.5
rho = 0.1
Q = 1000
maxgen = 100
x = coordinates[:, 0]
y = coordinates[:, 1]
D = cdist(coordinates, coordinates, metric='euclidean')
np.fill_diagonal(D, np.finfo(float).eps)

eta = 1.0 / D  # 启发因子
tau = np.ones(D.shape)  # 信息素矩阵
path = np.zeros((m, n), dtype=int)  # 记录路径
Lbest = []  # 记录每次迭代的最短路径长度
# 蚁群算法主循环
for iter in range(maxgen):
    # 放置蚂蚁
    path[:, 0] = np.random.choice(n, m, replace=True)

    # 选择城市
    for i in range(1, n):
        # 选择蚂蚁
        for j in range(m):
            visited = path[j, :i]
            unvisited = list(set(range(n)) - set(visited))
            # 计算概率
            P = [(tau[visited[-1], k] ** alpha) * (eta[visited[-1], k] ** beta) for k in unvisited]
            P = P / np.sum(P)
            P = np.cumsum(P)
            r = random.random()
            index = np.where(P >= r)[0][0]
            next_city = unvisited[index]
            path[j, i] = next_city

    # 计算路径长度
    L = np.zeros(m)
    for i in range(m):
        L[i] = np.sum(D[path[i, :], path[i, np.roll(range(n), -1)]])

    # 更新信息素
    shortest_route_length = np.min(L)
    Lbest.append(shortest_route_length)  # 记录最短路径长度
    shortest_route_index = np.argmin(L)
    best_path = path[shortest_route_index]
    delta_tau = np.zeros_like(tau)

    for i, route in enumerate(path):
        for j in range(n - 1):
            delta_tau[route[j], route[j + 1]] += Q / L[i]
        delta_tau[route[-1], route[0]] += Q / L[i]

    tau = (1 - rho) * tau + delta_tau
    path = np.zeros((m, n), dtype=int)

print('最短距离: ', shortest_route_length)
# 创建绘图
fig = go.Figure()

# 添加城市点
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers+text',
    name='Cities',
    text=[str(i) for i in range(1, dimension + 1)],
    textposition='top center',
    marker=dict(size=8, color='blue')
))

# 根据最短路径绘制路径线
# 这里假设 best_path 是 ACO 算法计算出的最短路径
path_coords = coordinates[best_path, :]
fig.add_trace(go.Scatter(
    x=path_coords[:, 0],
    y=path_coords[:, 1],
    mode='lines+markers',
    name='Path',
    line=dict(color='red')
))

# 标记起点和终点
fig.add_trace(go.Scatter(
    x=[path_coords[0, 0], path_coords[-1, 0]],
    y=[path_coords[0, 1], path_coords[-1, 1]],
    mode='markers',
    marker=dict(size=10, color='green'),
    showlegend=False
))

# 更新图表布局
fig.update_layout(
    title='蚁群算法优化TSP路径',
    xaxis_title='X坐标',
    yaxis_title='Y坐标'
)

# 显示图表
fig.show()
# 绘制最短路径长度随迭代次数变化的图
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=list(range(1, maxgen+1)),
    y=Lbest,
    mode='lines+markers',
    name='最短路径长度',
    line=dict(color='blue')
))
fig2.update_layout(
    title='最短路径长度随迭代次数变化',
    xaxis_title='迭代次数',
    yaxis_title='路径长度'
)

# 显示图表
fig2.show()