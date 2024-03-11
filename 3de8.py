import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义方程组
def equation1(h, y):
    return 3*h+2*y+45

def equation2(h, y):
    return 5*h+3*y


def equation3(h, y):
    return y-2*h-45

# 生成数据
h_vals = np.linspace(0, 300)
y_vals = np.linspace(0, 300)
h, y = np.meshgrid(h_vals, y_vals)
X1 = equation1(h, y)
X2 = equation2(h, y)
X3 = equation3(h, y)

# 绘制图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
#ax.plot_surface(h, y, X1, alpha=0.5, rstride=100, cstride=100, color='b', label='X - 3h + 2y = 45')
#ax.plot_surface(h, y, X2, alpha=0.5, rstride=100, cstride=100, color='r', label='X - 5h + 3y = 0')
ax.plot_surface(h, y, X3, alpha=0.5, rstride=100, cstride=100, color='r', label='X - 5h + 3y = 0')


# 设置坐标轴标签
ax.set_xlabel('h')
ax.set_ylabel('y')
ax.set_zlabel('X')

# 显示图形
plt.show()