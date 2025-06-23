from scipy.signal import convolve2d
import numpy as np

def resize(A, nElx, nEly): # This should be done by using the shape functions, this is just a quick shortcut!
    import cv2
    A_T = np.reshape(A, (nEly, nElx)).T
    B_T = cv2.resize(A_T, dsize=(nEly+1, nElx+1), interpolation=cv2.INTER_LINEAR)
    return B_T.T

def conv2(x, y, mode='same'):
    """
    Python analogue to the Matlab conv2(A,B) function. Returns the two-dimensional convolution of matrices A and B.
    @ x: input matrix 1
    @ y: input matrix 2
    """
    return convolve2d(x,  y, mode=mode)
