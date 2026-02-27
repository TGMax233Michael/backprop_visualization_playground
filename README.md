# Backpropagation Visualization Playground
This repository contains a simple implementation of a backpropagation visualization playground. It used several optimization functions to demonstrate how the optimization process and help user to understand how different optimization functions affect the backpropagation process.

# Environment
This project is implemented in Python 3.11 with numpy and matplotlib.
``` bash
$ pip install numpy matplotlib
```

# Usage
To run the backpropagation visualization playground, simply execute the `backprop_sim.py` file:
``` bash
$ python backprop_sim.py -f Sphere
```

| Argument | Description |
| --- | --- |
| `-f` | The optimization function to visualize. Options include: `Sphere`, `Rosenbrock`, `Rastrigin`, `Ackley`, `Griewank`, `Schwefel`. |
| `epochs` | The number of epochs to run the simulation for. |
| `-lr` | The learning rate for the optimization process. |
| `-start` | The starting point for the optimization process. |
| `-mp4` | Generate an mp4 video file of the optimization process. |

# Optimization Functions
The following optimization functions are implemented in this playground:
| Name | Expression | Recommended Interval | Default Starting Point | Optimal Point | Optimal Value |
| --- | --- | --- | --- | --- | --- |
| Sphere | $f(x, y) = x^2 + y^2$ | $x\in[-100, 100], y\in[-100, 100]$ | $(20, 20) $ | $(0, 0)$ | $0$ |
| Rosenbrock | $f(x, y) = 100(y-x^2)^2 + (1-x)^2$ | $x\in[-5, 10], y\in [-5, 10]$ | $(2, 2)$ | $(1, 1)$ | $0$ |
| Ackley | $f(x, y) = -20\exp\left(-0.2\sqrt{0.5(x^2+y^2)}\right) - \exp\left(0.5(\cos(2\pi x)+\cos(2\pi y))\right) + e + 20$ | $x\in[-32.768, 32.768], y\in[-32.768, 32.768]$ | $(1.5, 1.5)$ | $(0, 0)$ | $0$ |
| Rastrigin | $f(x, y) = 20 + x^2 - 10\cos(2\pi x) + y^2 - 10\cos(2\pi y)$ | $x\in[-5.12, 5.12], y\in[-5.12, 5.12]$ | $(1.5, 1.5)$ | $(0, 0)$ | $0$ |
| Beale | $f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2$ | $x\in[-4.5, 4.5], y\in[-4.5, 4.5]$ | $(1.5, 1.5)$ | $(3, 0.5)$ | $0$ |
| Booth | $f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2$ | $x\in[-10, 10], y\in[-10, 10]$ | $(5, 5)$ | $(1, 3)$ | $0$ |
| Easom | $f(x, y) = -\cos(x)\cos(y)\exp(-(x-\pi)^2-(y-\pi)^2)$ | $x\in[-10, 10], y\in[-10, 10]$ | $(2, 4)$ | $(\pi, \pi)$ | $-1$ |
| Goldstein-Price | $f(x, y) = (1 + (x + y + 1)^2(19 - 14x + 3x^2 - 14y + 6xy + 3y^2))(30 + (2x - 3y)^2(18 - 32x + 12x^2 + 48y - 36xy + 27y^2))$ | $x\in[-2, 2], y\in[-2, 2]$ | $(1.5, 1.5)$ | $(0, -1)$ | $3$ |
| Griewank | $\frac{1}{4000} (x^2+y^2) - cos(x)cos(\frac{y}{\sqrt{2}}) + 1$ | $x\in[-600, 600], y\in[-600, 600]$ | $(10, 10)$ | $(0, 0)$ | $0$ |
| Levi_N13 | $f(x, y) = \sin^2(3\pi x) + (x-1)^2(1+\sin^2(3\pi y)) + (y-1)^2(1+\sin^2(2\pi y))$ | $x\in[-10, 10], y\in[-10, 10]$ | $(3, 3)$ | $(1, 1)$ | $0$ |
| Matyas | $f(x, y) = 0.26(x^2+y^2) - 0.48xy$ | $x\in[-10, 10], y\in[-10, 10]$ | $(10, 10)$ | $(0, 0)$ | $0$ |
| Three-hump Camel | $f(x, y) = 2x^2 - 1.05x^4 + \frac{x^6}{6} + xy + y^2$ | $x\in[-5, 5], y\in[-5, 5]$ | $(2, 2)$ | $(0, 0)$ | $0$ |

## Reference:
1. [Optimization Test Functions](https://www.sfu.ca/~ssurjano/optimization.html)
2. [Wikipedia: Test functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization)

# Example
[[media/video.mp4]]