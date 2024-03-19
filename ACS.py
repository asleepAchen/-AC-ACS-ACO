#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm  
import random


# In[2]:


def read_tsp_file(filename):
    with open(filename, 'r') as file:
        lines = file.read().splitlines()
    
    name = None
    dimension = None
    coordinates = []

    node_coord_start = False
    for line in lines:
        if line.startswith('NAME'):
            name = line.split(':')[1].strip()
        elif line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
        elif line.startswith('NODE_COORD_SECTION'):
            node_coord_start = True
            continue  # 跳过这一行，直接开始读取坐标
        elif node_coord_start:
            if line.strip() == 'EOF':  # 检测到文件结束标记
                break
            parts = line.strip().split()
            if len(parts) >= 3:  # 确保行中有足够的数据
                _, x, y = parts[:3]
                coordinates.append((float(x), float(y)))

    return dimension, np.array(coordinates), name


# In[3]:


# 定义蚁群系统函数
def acs(dimension, coordinates, m=50, alpha=1, beta=5, rho=0.5, iterations=100):
    # 计算所有城市对之间的距离
    distances = np.sqrt(((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2).sum(axis=2))
    # 初始化城市间路径上的信息素水平
    pheromones = np.ones_like(distances)
    # 将初始最佳路径长度设置为无穷大
    best_length = np.inf
    # 初始化最佳路径变量
    best_path = None
    # 初始化列表以记录每次迭代中找到的最佳路径长度
    length_history = []

    # 对指定的迭代次数进行主循环
    for iteration in range(iterations):
        # 为所有蚂蚁初始化路径
        paths = np.zeros((m, dimension), dtype=int)
        lengths = np.zeros(m)
        for ant in range(m):
            # 随机选择起始城市
            paths[ant, 0] = random.randint(0, dimension - 1)
            for i in range(1, dimension):
                current = paths[ant, i - 1]
                # 将当前城市到自身的距离设置为一个非零的大数
                temp_distances = np.copy(distances[current])
                temp_distances[temp_distances == 0] = np.inf

                # 计算转移到下一个城市的概率
                probabilities = pheromones[current] ** alpha * ((1.0 / temp_distances) ** beta)
                # 已访问的城市概率设置为0，避免重复访问
                probabilities[paths[ant, :i]] = 0
                next_city = random.choices(range(dimension), weights=probabilities)[0]
                paths[ant, i] = next_city

            # 计算并记录当前路径的长度
            length = np.sum(distances[paths[ant], np.roll(paths[ant], shift=-1)])
            # 更新最佳路径和长度
            if length < best_length:
                best_length = length
                best_path = paths[ant]
            lengths[ant] = length
        # 记录此次迭代的最佳长度
        length_history.append(best_length)
        # 更新信息素
        pheromones *= (1 - rho)
        Q = 1000.0  # 定义常数Q
        for ant in range(m):
            for i in range(dimension - 1):
                # 在路径上增加信息素
                pheromones[paths[ant, i], paths[ant, i + 1]] += Q / lengths[ant]
            # 确保路径是闭环的，即最后一个城市回到第一个城市
            pheromones[paths[ant, -1], paths[ant, 0]] += Q / lengths[ant]
    # 返回最佳路径长度、最佳路径和历史长度列表
    return best_length, best_path, length_history


# In[4]:


def ad(dimension, coordinates, m=50, alpha=1, beta=5, rho=0.01, iterations=100):
    # 计算所有城市对之间的距离
    distances = np.sqrt(((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2).sum(axis=2))
    # 初始化城市间路径上的信息素水平
    pheromones = np.ones_like(distances)
    # 将初始最佳路径长度设置为无穷大
    best_length = np.inf
    # 初始化最佳路径变量
    best_path = None
    # 初始化列表以记录每次迭代中找到的最佳路径长度
    length_history = []

    # 对指定的迭代次数进行主循环
    for iteration in range(iterations):
        # 为所有蚂蚁初始化路径
        paths = np.zeros((m, dimension), dtype=int)
        lengths = np.zeros(m)
        for ant in range(m):
            # 随机选择起始城市
            paths[ant, 0] = random.randint(0, dimension - 1)
            for i in range(1, dimension):
                current = paths[ant, i - 1]
                # 将当前城市到自身的距离设置为一个非零的大数
                temp_distances = np.copy(distances[current])
                temp_distances[temp_distances == 0] = np.inf

                # 计算转移到下一个城市的概率
                probabilities = pheromones[current] ** alpha * ((1.0 / temp_distances) ** beta)
                # 已访问的城市概率设置为0，避免重复访问
                probabilities[paths[ant, :i]] = 0
                next_city = random.choices(range(dimension), weights=probabilities)[0]
                paths[ant, i] = next_city

            # 计算并记录当前路径的长度
            length = np.sum(distances[paths[ant], np.roll(paths[ant], shift=-1)])
            # 更新最佳路径和长度
            if length < best_length:
                best_length = length
                best_path = paths[ant]
            lengths[ant] = length
        # 记录此次迭代的最佳长度
        length_history.append(best_length)
        # 更新信息素
        pheromones *= (1 - rho)
        Q = 1000.0  # 定义常数Q
        for ant in range(m):
            for i in range(dimension - 1):
                # 在路径上增加固定量的信息素
                pheromones[paths[ant, i], paths[ant, i + 1]] += Q
            # 确保路径是闭环的，即最后一个城市回到第一个城市
            pheromones[paths[ant, -1], paths[ant, 0]] += Q
    # 返回最佳路径长度、最佳路径和历史长度列表
    return best_length, best_path, length_history


# In[5]:


def aq(dimension, coordinates, m=50, alpha=1, beta=5, rho=0.01, iterations=100):
    # 计算所有城市对之间的距离
    distances = np.sqrt(((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2).sum(axis=2))
    # 初始化城市间路径上的信息素水平
    pheromones = np.ones_like(distances)
    # 将初始最佳路径长度设置为无穷大
    best_length = np.inf
    # 初始化最佳路径变量
    best_path = None
    # 初始化列表以记录每次迭代中找到的最佳路径长度
    length_history = []

    # 对指定的迭代次数进行主循环
    for iteration in range(iterations):
        # 为所有蚂蚁初始化路径
        paths = np.zeros((m, dimension), dtype=int)
        lengths = np.zeros(m)
        for ant in range(m):
            # 随机选择起始城市
            paths[ant, 0] = random.randint(0, dimension - 1)
            for i in range(1, dimension):
                current = paths[ant, i - 1]
                # 将当前城市到自身的距离设置为一个非零的大数
                temp_distances = np.copy(distances[current])
                temp_distances[temp_distances == 0] = np.inf

                # 计算转移到下一个城市的概率
                probabilities = pheromones[current] ** alpha * ((1.0 / temp_distances) ** beta)
                # 已访问的城市概率设置为0，避免重复访问
                probabilities[paths[ant, :i]] = 0
                next_city = random.choices(range(dimension), weights=probabilities)[0]
                paths[ant, i] = next_city

            # 计算并记录当前路径的长度
            length = np.sum(distances[paths[ant], np.roll(paths[ant], shift=-1)])
            # 更新最佳路径和长度
            if length < best_length:
                best_length = length
                best_path = paths[ant]
            lengths[ant] = length
        # 记录此次迭代的最佳长度
        length_history.append(best_length)
        # 更新信息素
        pheromones *= (1 - rho)
        Q = 1000.0  # 定义常数Q
        for ant in range(m):
            for i in range(dimension - 1):
                # 在路径上增加与路径长度成反比的信息素
                pheromones[paths[ant, i], paths[ant, i + 1]] += Q / distances[paths[ant, i], paths[ant, i + 1]]
            # 确保路径是闭环的，即最后一个城市回到第一个城市
            pheromones[paths[ant, -1], paths[ant, 0]] += Q / distances[paths[ant, -1], paths[ant, 0]]
    # 返回最佳路径长度、最佳路径和历史长度列表
    return best_length, best_path, length_history


# In[6]:


#寻找 最佳rho
def optimize_acs_for_rho(acs_func, dimension, coordinates, m=50, alpha=1, beta=5, iterations=100):
    best_overall_length = float('inf')
    best_overall_path = []
    best_overall_history = []
    best_rho = 0.01

    for rho in tqdm([x * 0.01 for x in range(1, 100)], desc="Optimizing"):
        best_length, best_path, length_history = acs_func(dimension, coordinates, m, alpha, beta, rho, iterations)
        
        if best_length < best_overall_length:
            best_overall_length = best_length
            best_overall_path = best_path
            best_overall_history = length_history
            best_rho = rho

    return best_rho, best_overall_length, best_overall_path, best_overall_history


# In[7]:


# 读取TSP数据
dimension, coordinates, name = read_tsp_file('kroA100.tsp')

print(f"TSP Name: {name}, Dimension: {dimension}")


# In[8]:


len(coordinates)


# In[9]:


name


# In[10]:


best_rho,best_length, best_path, length_history = optimize_acs_for_rho(acs,dimension, coordinates)
AQ_best_rho, AQ_best_length, AQ_best_path, AQ_best_history =  optimize_acs_for_rho(aq,dimension, coordinates)
AD_best_rho, AD_best_length, AD_best_path, AD_best_history =  optimize_acs_for_rho(aq,dimension, coordinates)


# In[11]:


# 运行ACS,AQ,AD
#best_length, best_path, length_history = acs(dimension, coordinates)
#AQ_best_length, AQ_best_path, AQ_length_history = aq(dimension, coordinates)
#AD_best_length, AD_best_path, AD_length_history = ad(dimension, coordinates)


# In[22]:


print(best_rho,AQ_best_rho,AD_best_rho)


# In[23]:


print(best_length,AQ_best_length,AD_best_length)


# In[13]:


#ACS
path_coordinates = coordinates[best_path]

# 创建一个图形对象
fig = go.Figure()

# 添加城市节点为散点图层
fig.add_trace(go.Scatter(x=coordinates[:, 0], y=coordinates[:, 1], 
                         mode='markers', name='Cities',
                         marker=dict(color='blue', size=8),
                         text=[f'City {i}' for i in range(len(coordinates))]))

# 添加路径为线图层
fig.add_trace(go.Scatter(x=path_coordinates[:, 0], y=path_coordinates[:, 1],
                         mode='lines+markers', name='Path', 
                         line=dict(color='red', width=2),
                         marker=dict(size=8, color='red')))

# 更新布局设置
fig.update_layout(title='最短路径', 
                  xaxis_title='X 横坐标',
                  yaxis_title='Y 纵坐标',
                  showlegend=True)

# 显示图形
fig.show()


# In[14]:


#ACS
# 使用Plotly绘制最佳路径长度随迭代次数变化的图
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, len(length_history) + 1)), y=length_history,
                         mode='lines+markers', name='Best Path Length',
                         line=dict(color='blue')))

fig.update_layout(title='Best Path Length Over Iterations',
                  xaxis_title='Iteration',
                  yaxis_title='Path Length',
                  showlegend=True)

fig.show()


# In[15]:


#AQ
path_coordinates = coordinates[AQ_best_path]

# 创建一个图形对象
fig = go.Figure()

# 添加城市节点为散点图层
fig.add_trace(go.Scatter(x=coordinates[:, 0], y=coordinates[:, 1], 
                         mode='markers', name='Cities',
                         marker=dict(color='blue', size=8),
                         text=[f'City {i}' for i in range(len(coordinates))]))

# 添加路径为线图层
fig.add_trace(go.Scatter(x=path_coordinates[:, 0], y=path_coordinates[:, 1],
                         mode='lines+markers', name='Path', 
                         line=dict(color='red', width=2),
                         marker=dict(size=8, color='red')))

# 更新布局设置
fig.update_layout(title='最短路径', 
                  xaxis_title='X 横坐标',
                  yaxis_title='Y 纵坐标',
                  showlegend=True)

# 显示图形
fig.show()


# In[18]:


#AQ
# 使用Plotly绘制最佳路径长度随迭代次数变化的图
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, len(AQ_best_history) + 1)), y=AQ_best_history,
                         mode='lines+markers', name='Best Path Length',
                         line=dict(color='blue')))

fig.update_layout(title='Best Path Length Over Iterations',
                  xaxis_title='Iteration',
                  yaxis_title='AQ_Path Length',
                  showlegend=True)

fig.show()


# In[19]:


#AD
path_coordinates = coordinates[AD_best_path]

# 创建一个图形对象
fig = go.Figure()

# 添加城市节点为散点图层
fig.add_trace(go.Scatter(x=coordinates[:, 0], y=coordinates[:, 1], 
                         mode='markers', name='Cities',
                         marker=dict(color='blue', size=8),
                         text=[f'City {i}' for i in range(len(coordinates))]))

# 添加路径为线图层
fig.add_trace(go.Scatter(x=path_coordinates[:, 0], y=path_coordinates[:, 1],
                         mode='lines+markers', name='Path', 
                         line=dict(color='red', width=2),
                         marker=dict(size=8, color='red')))

# 更新布局设置
fig.update_layout(title='最短路径', 
                  xaxis_title='X 横坐标',
                  yaxis_title='Y 纵坐标',
                  showlegend=True)

# 显示图形
fig.show()


# In[20]:


#AD
# 使用Plotly绘制最佳路径长度随迭代次数变化的图
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, len(AD_best_history) + 1)), y=AD_best_history,
                         mode='lines+markers', name='Best Path Length',
                         line=dict(color='blue')))

fig.update_layout(title='Best Path Length Over Iterations',
                  xaxis_title='Iteration',
                  yaxis_title='AQ_Path Length',
                  showlegend=True)

fig.show()


# In[ ]:




