#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from elastic import Elastic
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

Nx = 128
Ny = 128
L = 6

tx = np.linspace(-L/2, L/2, Nx)
ty = np.linspace(-L/2, L/2, Ny)
(x, y) = np.meshgrid(tx, ty)

pressure = np.random.rand(Nx, Ny)
elastic = Elastic(Nx, Ny, path="/home/tom/work/coefficients/")
deformation = elastic.deformation(pressure)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(x, y, deformation)

plt.show(block=True)
