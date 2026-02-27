import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray

class Optimize_Target(ABC):
    def __init__(self) -> None:
        self.recommend_interval = {"x": None, "y": None}
        self.recommend_start_pos = None
        self.optimal_res = None
        self.func_expression = ""

    def check_start_pos(self, start_pos):
        x_c = start_pos[0]
        y_c = start_pos[1]

        if (x_c < self.recommend_interval["x"][0]
            or x_c > self.recommend_interval["x"][1]
            or y_c < self.recommend_interval["y"][0]
            or y_c > self.recommend_interval["y"][1]):
            print("Start position out of recommended interval, may lead to unstable optimization")
            print("Consider adjusting the start position to the recommended interval.")
            print(f"Recommended interval: x\in{self.recommend_interval['x']} y\in{self.recommend_interval['y']}")

    @abstractmethod
    def func(self, x) -> NDArray:
        pass

    @abstractmethod
    def grad(self, x) -> NDArray:
        pass


class Sphere(Optimize_Target):
    def __init__(self) -> None:
        self.recommend_interval = {"x": [-10, 10], "y": [-10, 10]}
        self.recommend_start_pos = np.array([20, 20])
        self.optimal_res = [0, 0, 0]
        self.func_expression = r"$ F(x) = x^2 + y^2 $"

    def func(self, x):
        return np.sum(x**2, axis=-1)

    def grad(self, x):
        return 2*x

class Rosenbrock(Optimize_Target):
    """
        Recommend interval: x\in[-3, 3] y\in[-3, 3]
    """
    def __init__(self) -> None:
        self.recommend_interval = {"x": [-5, 10], "y": [-5, 10]}
        self.recommend_start_pos = np.array([2, 2])
        self.optimal_res = [1, 1, 0]
        self.func_expression = r"$ F(x) = 100(y-x^2)^2 + (1-x)^2 $"

    def func(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        return 100*(y_c - x_c**2)**2 + (1 - x_c)**2

    def grad(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        x_grad = 4*100*x_c**3 - 4*100*y_c*x_c + 2*x_c - 2
        y_grad = 2*100*(y_c-x_c**2)
        return np.stack([x_grad, y_grad], axis=-1)


class Ackley(Optimize_Target):
    """
        Recommand interval: x\in[-4, 4] y\in[-4, 4]
    """
    def __init__(self) -> None:
        self.recommend_interval = {"x": [-32.768, 32.768], "y": [-32.768, 32.768]}
        self.recommend_start_pos = np.array([1.5, 1.5])
        self.optimal_res = [0, 0, 0]
        self.func_expression = r"$ F(x) = -20 exp(-0.2 \sqrt{\frac{1}{2} (x^2+y^2)}) - exp(\frac{1}{2} (cos(2 \pi x) + cos(2 \pi y))) + 20 + e $"

    def func(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        return -20*np.exp(-0.2*np.sqrt(0.5*(x_c**2+y_c**2))) - np.exp(0.5*(np.cos(2*np.pi*x_c) + np.cos(2*np.pi*y_c))) + np.e + 20

    def grad(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        sum_sqrt = np.sqrt(0.5*(x_c**2+y_c**2))
        sum_cos = np.cos(2*np.pi*x_c) + np.cos(2*np.pi*y_c)
        x_grad = (20*0.2*x_c)/(2*sum_sqrt + 1e-8)*np.exp(-0.2*sum_sqrt) + np.pi*np.sin(2*np.pi*x_c)*np.exp(sum_cos/2)
        y_grad = (20*0.2*y_c)/(2*sum_sqrt + 1e-8)*np.exp(-0.2*sum_sqrt) + np.pi*np.sin(2*np.pi*y_c)*np.exp(sum_cos/2)
        return np.stack([x_grad, y_grad], axis=-1)

class Rastrigine(Optimize_Target):
    def __init__(self) -> None:
        self.recommend_interval = {"x": [-5.12, 5.12], "y": [-5.12, 5.12]}
        self.recommend_start_pos = np.array([1.5, 1.5])
        self.optimal_res = [0, 0, 0]
        self.func_expression = r"$ 20 + \left[ x^2-10 cos(2\pi x)\right] + \left[ y^2-10 cos(2\pi y)\right] $"

    def func(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        return 20 + (x_c**2 - 10*np.cos(2*np.pi*x_c)) + (y_c**2 - 10*np.cos(2*np.pi*y_c))

    def grad(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        x_grad = 2*x_c + 40*np.pi*np.sin(2*np.pi*x_c)
        y_grad = 2*y_c + 40*np.pi*np.sin(2*np.pi*y_c)
        return np.stack([x_grad, y_grad], axis=-1)

class Beale(Optimize_Target):
    def __init__(self) -> None:
        self.recommend_interval = {"x": [-4.5, 4.5], "y": [-4.5, 4.5]}
        self.recommend_start_pos = np.array([1.5, 1.5])
        self.optimal_res = [3, 0.5, 0]
        self.func_expression = r"$ F(x) = (1.5-x+x*y)^2 + (2.25-x+x*y^2)^2 + (2.625-x+x*y^3)^2 $"

    def func(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        return (1.5-x_c+x_c*y_c)**2 + (2.25-x_c+x_c*y_c**2)**2 + (2.625-x_c+x_c*y_c**3)**2

    def grad(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        x_grad = 2*(1.5-x_c+x_c**2*y_c)*(2*y_c-1) + 2*(2.25-x_c+x_c**2*y_c**2)*(2*y_c**2-1) + 2*(2.625-x_c+x_c**2*y_c**3)*(2*y_c**3-1)
        y_grad = 2*(1.5-x_c+x_c**2*y_c)*x_c* + 2*(2.25-x_c+x_c**2*y_c**2)*2*x_c*y_c + 2*(2.625-x_c+x_c**2*y_c**3)*3*x_c*y_c**2
        return np.stack([x_grad, y_grad], axis=-1)
        
class Booth(Optimize_Target):
    def __init__(self) -> None:
        self.recommend_interval = {"x": [-10, 10], "y": [-10, 10]}
        self.recommend_start_pos = np.array([5, 5])
        self.optimal_res = [1, 3, 0]
        self.func_expression = r"$ F(x) = (x+2*y-7)^2 + (2*x+y-5)^2 $"

    def func(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        return (x_c+2*y_c-7)**2 + (2*x_c+y_c-5)**2

    def grad(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        x_grad = 2*(x_c+2*y_c-7) + 4*(2*x_c+y_c-5)
        y_grad = 4*(x_c+2*y_c-7) + 2*(2*x_c+y_c-5)
        return np.stack([x_grad, y_grad], axis=-1)
        
class Easom(Optimize_Target):
    def __init__(self) -> None:
        self.recommend_interval = {"x": [-10, 10], "y": [-10, 10]}
        self.recommend_start_pos = np.array([2, 4])
        self.optimal_res = [np.pi, np.pi, -1]
        self.func_expression = r"$ F(x) = -cos(x)cos(y)e^{-(x-\pi)^2-(y-\pi)^2} $"

    def func(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        return -np.cos(x_c)*np.cos(y_c)*np.exp(-(x_c-np.pi)**2-(y_c-np.pi)**2)

    def grad(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        x_grad = np.cos(y_c)*np.exp(-(x_c-np.pi)**2-(y_c-np.pi)**2) + 2*(x_c-np.pi)*np.cos(x_c)*np.cos(y_c)*np.exp(-(x_c-np.pi)**2-(y_c-np.pi)**2)
        y_grad = np.cos(x_c)*np.exp(-(x_c-np.pi)**2-(y_c-np.pi)**2) + 2*(y_c-np.pi)*np.cos(x_c)*np.cos(y_c)*np.exp(-(x_c-np.pi)**2-(y_c-np.pi)**2)
        return np.stack([x_grad, y_grad], axis=-1)
    
    
class Goldstein_Price(Optimize_Target):
    def __init__(self) -> None:
        self.recommend_interval = {"x": [-2, 2], "y": [-2, 2]}
        self.recommend_start_pos = np.array([1.5, 1.5])
        self.optimal_res = [0, -1, 3]
        self.func_expression = r"$ F(x) = \left[1+(x+y+1)^2(19-14x+3x^2-14y+6xy+3y^2)\right] \cdot \left[30+(2x-3y)^2(18-32x+12x^2+48y-36xy+27y^2)\right] $"

    def func(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        return (1+(x_c+y_c+1)**2*(19-14*x_c+3*x_c**2-14*y_c+6*x_c*y_c+3*y_c**2)) * (30+(2*x_c-3*y_c)**2*(18-32*x_c+12*x_c**2+48*y_c-36*x_c*y_c+27*y_c**2))

    def grad(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        f1 = 1+(x_c+y_c+1)**2*(19-14*x_c+3*x_c**2-14*y_c+6*x_c*y_c+3*y_c**2)
        f2 = 30+(2*x_c-3*y_c)**2*(18-32*x_c+12*x_c**2+48*y_c-36*x_c*y_c+27*y_c**2)
        f1_x_grad = 4*(x_c+y_c+1)*(19-14*x_c+3*x_c**2-14*y_c+6*x_c*y_c+3*y_c**2) + (x_c+y_c+1)**2*(-14 + 6*2*x_c + 6*y_c)
        f1_y_grad = 4*(x_c+y_c+1)*(19-14*x_c+3*x_c**2-14*y_c+6*x_c*y_c+3*y_c**2) + (x_c+y_c+1)**2*(-14 + 6*x_c + 6*2*y_c)
        f2_x_grad = 4*(2*x_c-3*y_c)*(18-32*x_c+12*x_c**2+48*y_c-36*x_c*y_c+27*y_c**2) + (2*x_c-3*y_c)**2*(-32 + 6*2*x_c - 36*y_c)
        f2_y_grad = 4*(2*x_c-3*y_c)*(18-32*x_c+12*x_c**2+48*y_c-36*x_c*y_c+27*y_c**2) + (2*x_c-3*y_c)**2*(48 - 36*x_c + 54*y_c)
        x_grad = f1_x_grad * f2 + f1 * f2_x_grad
        y_grad = f1_y_grad * f2 + f1 * f2_y_grad
        return np.stack([x_grad, y_grad], axis=-1)
    
    
class Griewank(Optimize_Target):
    def __init__(self) -> None:
        self.recommend_interval = {"x": [-600, 600], "y": [-600, 600]}
        self.recommend_start_pos = np.array([10, 10])
        self.optimal_res = [0, 0, 0]
        self.func_expression = r"$ F(x) = \frac{1}{4000} (x^2+y^2) - cos(x)cos(\frac{y}{\sqrt{2}}) + 1 $"

    def func(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        return (x_c**2+y_c**2)/4000 - np.cos(x_c)*np.cos(y_c/np.sqrt(2)) + 1

    def grad(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        x_grad = x_c/2000 + np.sin(x_c)*np.cos(y_c/np.sqrt(2))
        y_grad = y_c/2000 + np.cos(x_c)*np.sin(y_c/np.sqrt(2))/np.sqrt(2)
        return np.stack([x_grad, y_grad], axis=-1)
    
    
class Levi_N13(Optimize_Target):
    def __init__(self) -> None:
        self.recommend_interval = {"x": [-10, 10], "y": [-10, 10]}
        self.recommend_start_pos = np.array([3, 3])
        self.optimal_res = [1, 1, 0]
        self.func_expression = r"$ F(x) = sin^2(3\pi x) + (x-1)^2 \left[1+sin^2(3\pi y)\right] + (y-1)^2 \left[1+sin^2(2\pi y)\right] $"

    def func(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        return np.sin(3*np.pi*x_c)**2 + (x_c-1)**2 * (1+np.sin(3*np.pi*y_c)**2) + (y_c-1)**2 * (1+np.sin(2*np.pi*y_c)**2)

    def grad(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        x_grad = 6*np.pi*np.sin(3*np.pi*x_c)*np.cos(3*np.pi*x_c) + 2*(x_c-1)*(1+np.sin(3*np.pi*y_c)**2)
        y_grad = 6*(x_c-1)**2 * np.sin(3*np.pi*y_c)*np.cos(3*np.pi*y_c) + 4*(y_c-1)*(1+np.sin(2*np.pi*y_c)**2) + 8*np.pi*(y_c-1)**2 * np.sin(2*np.pi*y_c)*np.cos(2*np.pi*y_c)
        return np.stack([x_grad, y_grad], axis=-1)
    
    
class Matyas(Optimize_Target):
    def __init__(self) -> None:
        self.recommend_interval = {"x": [-10, 10], "y": [-10, 10]}
        self.recommend_start_pos = np.array([10, 10])
        self.optimal_res = [0, 0, 0]
        self.func_expression = r"$ F(x) = 0.26(x^2+y^2) - 0.48xy $"

    def func(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        return 0.26*(x_c**2+y_c**2) - 0.48*x_c*y_c

    def grad(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        x_grad = 0.52*x_c - 0.48*y_c
        y_grad = 0.52*y_c - 0.48*x_c
        return np.stack([x_grad, y_grad], axis=-1)
    
    
class Three_Hump_Camel(Optimize_Target):
    def __init__(self) -> None:
        self.recommend_interval = {"x": [-5, 5], "y": [-5, 5]}
        self.recommend_start_pos = np.array([2, 2])
        self.optimal_res = [0, 0, 0]
        self.func_expression = r"$ F(x) = 2x^2 - 1.05x^4 + \frac{x^6}{6} + xy + y^2 $"

    def func(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        return 2*x_c**2 - 1.05*x_c**4 + x_c**6/6 + x_c*y_c + y_c**2

    def grad(self, x):
        x_c = x[..., 0]
        y_c = x[..., 1]
        x_grad = 4*x_c - 4.2*x_c**3 + x_c**5 + y_c
        y_grad = x_c + 2*y_c
        return np.stack([x_grad, y_grad], axis=-1)