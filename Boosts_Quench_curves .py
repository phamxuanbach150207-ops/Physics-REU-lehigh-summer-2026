import matplotlib.pyplot as plt
import numpy as np

field_strengths = [0,40,100,200,400,600,800,1000,
                   1500,2000,3000,4000,5000,6000,7000,
                   8000,9000,10000]
integral_ratios = [1,]

plt.scatter(field_strengths,integral_ratios)
plt.xlabel('Field Strength [Gauss]')
plt.ylabel('Integral Ratio')
plt.axhline(y=1,color='black',linewidth=1)
plt.grid(alpha=0.3)
plt.ylim(0.8,1.4)
plt.show()
