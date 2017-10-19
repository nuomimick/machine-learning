# optimize库中的fsolve函数可以用来对非线性方程组进行求解
from scipy.optimize import fsolve
from math import sin,cos

def f(x):
    x0 = float(x[0])
    x1 = float(x[1])
    x2 = float(x[2])
    return [
    5*x1+3,
    4*x0*x0 - 2*sin(x1*x2),
    x1*x2 - 1.5
    ]

result = fsolve(f, [1,1,1])

print(result)
print(f(result))
'''
由于fsolve函数在调用函数f时，传递的参数为数组，因此如果直接使用数组中的元素计算的话，计算
速度将会有所降低，因此这里先用float函数将数组中的元素转换为Python中的标准浮点数，然后调用
标准math库中的函数进行运算。
在对方程组进行求解时，fsolve会自动计算方程组的雅可比矩阵，如果方程组中的未知数很多，而与每
个方程有关的未知数较少时，即雅可比矩阵比较稀疏时，传递一个计算雅可比矩阵的函数将能大幅度
提高运算速度。
'''

from scipy.optimize import fsolve
from math import sin,cos
def f(x):
    x0 = float(x[0])
    x1 = float(x[1])
    x2 = float(x[2])
    return [
    5*x1+3,
    4*x0*x0 - 2*sin(x1*x2),
    x1*x2 - 1.5
    ]

def j(x):
    x0 = float(x[0])
    x1 = float(x[1])
    x2 = float(x[2])
    return [
    [0, 5, 0],
    [8*x0, -2*x2*cos(x1*x2), -2*x1*cos(x1*x2)],
    [0, x2, x1]
    ]

result = fsolve(f, [1,1,1], fprime=j)
print(result)
print(f(result))