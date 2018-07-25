import math
import numpy as np

def sobel(image):
	Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
	Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

	rows = len(image)
	cols = len(image[0])
	temp = np.zeros((len(image), len(image[0])))

	for i in range(1, rows-2):
		for j in range(1, cols-2):
			S1 = sum(sum(Gx*image[i:i+3, j:j+3]))
			S2 = sum(sum(Gy*image[i:i+3, j:j+3]))

			temp[i+1, j+1] = math.sqrt(S1**2 + S2**2)

	return temp
