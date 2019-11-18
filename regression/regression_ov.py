import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data1.txt',header=None)
x = data.iloc[:,0]
y = data.iloc[:,1]
m = len(y)

plt.figure()
plt.scatter(x,y,color='#ff0000',marker='o')
plt.title('Data Points',fontsize=24)
plt.xlabel('$x$',fontsize=24)
plt.ylabel('$y$',fontsize=24)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show() 