import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',', usecols=range(32))

nf = plt.figure()
plt.imshow(data)
plt.show()