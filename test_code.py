import numpy as np
from scipy.spatial.distance import euclidean

aa = [[]]*2

aa[0].append([1, 2])
print(aa)

bb = np.array([[1],[2]])
cc = np.array([1,2,3])
print(bb+cc)

dd = np.array([[1,2], [3,4]])
ee = np.array([1,1])
print(np.linalg.norm(dd-ee, axis=-1))


