import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',')

nf = plt.figure()
plt.imshow(data)
plt.show()