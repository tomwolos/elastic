import numpy as np
import pickle
import pyfftw as pf


class Elastic:
    """Deformation of an elastic half-space for a given normal stress

    Calculate the deformation of an elastic half-space for a given normal
    stress distribution using FFT. Perform FFTW planning and generate the
    influence coefficients. The coefficients are saved to a file and then
    reloaded as needed.
    Dependencies: numpy, pyfftw

    Attributes
    ----------
    nx, ny : int, int
        numbers of points used for the discretisation of stress/deformation
    scale : tuple(float, float)
        scaling factors for the x and y directions (default (1.0, 1.0));
    path : string
        path to the influence coefficients file (default '');
        set path=None if you do not want to load/save the coefficients
    verbose : bool
        print when generating new influence coefficients (default True)

    Methods
    ------
    update(stress : numpy.ndarray) -> numpy.ndarray
        calculate and return the deformation for the given normal stress
    """

    def __init__(self, nx: int,
                 ny: int,
                 scale: (1.0, 1.0),
                 path: str='',
                 verbose: bool=True):
        self.nx = nx
        self.ny = ny
        self.scale = scale
        ratio = self.scale[0]/self.scale[1]
        self.path = path
        self.verbose = verbose
        self.x = np.linspace(0, ratio, nx)
        self.y = np.linspace(0, 1, ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # FFTW planning
        self.real = pf.empty_aligned((2*nx, 2*ny), 'float64')
        self.complex = pf.empty_aligned((2*nx, ny+1), 'complex128')
        self.fft = pf.FFTW(self.real, self.complex, axes=(0, 1))
        self.ifft = pf.FFTW(self.complex, self.real, axes=(0, 1),
                            direction='FFTW_BACKWARD')

        # load or generate and save the influence coefficients and their FFT
        self.fftcoeffs = pf.empty_aligned((2*nx, ny+1), 'complex128')
        self.coeffs = np.zeros((nx, ny))
        if path is not None:
            filename = f'{path}coeffs-{nx}-{ny}.npz'
            try:
                temp = np.load(filename, allow_pickle=True)
                self.fftcoeffs[:] = temp[ratio][0]
                self.coeffs[:] = temp[ratio][1]
            except IOError:
                self.__generate()
                with open(filename, 'wb') as f:
                    pickle.dump({ratio: (self.fftcoeffs, self.coeffs)}, f)
            except KeyError:
                self.__generate()
                temp[ratio] = (self.fftcoeffs, self.coeffs)
                with open(filename, 'wb') as f:
                    pickle.dump(temp, f)
        else:
            self.__generate()
        self.coeffs[:] *= scale[1]
        self.fftcoeffs[:] *= scale[1]


    def __generate(self):
        if self.verbose:
            print(f'Generating influence coefficients '
                  f'({self.nx} x {self.ny})', flush=True)
        for jx in range(self.nx):
            for jy in range(self.ny):
                x1 = self.x[jx] - self.x[-1] + self.dx/2
                x2 = self.x[jx] - self.x[-1] - self.dx/2
                y1 = self.y[jy] - self.y[-1] + self.dy/2
                y2 = self.y[jy] - self.y[-1] - self.dy/2
                self.real[jx][jy] = (
                    np.abs(x1)*(np.arcsinh(y1/x1) - np.arcsinh(y2/x1))
                    + np.abs(x2)*(np.arcsinh(y2/x2) - np.arcsinh(y1/x2))
                    + np.abs(y1)*(np.arcsinh(x1/y1) - np.arcsinh(x2/y1))
                    + np.abs(y2)*(np.arcsinh(x2/y2) - np.arcsinh(x1/y2)))
                self.real[-2-jx][jy] = self.real[jx][jy]
                self.real[jx][-2-jy] = self.real[jx][jy]
                self.real[-2-jx][-2-jy] = self.real[jx][jy]
        self.coeffs[:] = self.real[:self.nx, :self.ny]
        self.fft.execute()
        self.real[:] = 0.0
        self.fftcoeffs[:] = self.complex
        if self.verbose:
            print('done.\n', flush=True)

    def update(self, stress: np.ndarray) -> np.ndarray:
        self.real[:] = 0.0
        self.real[:self.nx, :self.ny] = stress 
        self.fft.execute()
        self.complex[:] *= self.fftcoeffs
        self.ifft.execute()
        self.real[-1-self.nx:-1, -1-self.ny:-1] /= 2*self.nx*2*self.ny
        return self.real[-1-self.nx:-1, -1-self.ny:-1]
