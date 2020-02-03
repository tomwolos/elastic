#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from elastic import Elastic
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

Nx = 128
Ny = 128
L = 4
path = '/home/tom/work/coefficients/'

tx = np.linspace(-L/2, L/2, Nx)
ty = np.linspace(-L/2, L/2, Ny)
scale = (tx[-1] - tx[0], ty[-1] - ty[0])
(x, y) = np.meshgrid(tx, ty, indexing='ij')

# Hertzian contact
stress = np.zeros((Nx, Ny))
z = x**2 + y**2
stress[z < 1] = np.sqrt(1 - z[z < 1])
elastic = Elastic(Nx, Ny, scale, path)
deformation = 2/(np.pi**2)*elastic.update(stress)
gap = x**2/2 + y**2/2 + deformation

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Stress')
ax.plot_wireframe(x, y, stress)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Elastic deformation')
ax.plot_wireframe(x, y, deformation)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Gap')
ax.plot_wireframe(x, y, gap)

plt.show(block=True)
