from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from os import listdir, path
import json

target_files = []
for name in listdir('.'):
	if 'P_' in name and '_vers' in name and '_proc' in name:
		target_files.append(name)
target_files = sorted(target_files)
print(target_files)

x = []
y = []
p = []

for name in target_files:
	f = open(name, 'r')
	text = f.read()
	f.close()
	values = json.loads(text)
	x.extend(values[0])
	y.extend(values[1])
	p.extend(values[2])

def plot():
	fig = plt.figure()
	fig.patch.set_facecolor('white')
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_trisurf(x, y, p, cmap=cm.coolwarm)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('P')
	plt.show()
	plt.close()

plot()
