import numpy as np
from numba import jit

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def calc_index_constraint(indexes_design_region, indexes_heating):
  indexes_constraint = np.zeros(len(indexes_heating[0,:]))
  for i in range(len(indexes_design_region[0,:])):
    for j in range(len(indexes_heating[0,:])):
      if np.abs(indexes_design_region[0,i] - indexes_heating[0,j]) < 1e-2 and np.abs(indexes_design_region[1,i] - indexes_heating[1,j]) < 1e-2:
        indexes_constraint[j] =  i
  return indexes_constraint