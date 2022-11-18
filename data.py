import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("csiro_alt_gmsl_mo_2015_csv_edited.csv")
df['id'] = df.index
df.dropna(subset=["GMSL"], inplace=True)
print(df)

x = np.array(df.index)
y = np.array(df.GMSL)

plt.bar(x, y, color='b')
plt.plot(x, y, color='b')
plt.xlabel("1993-January to 2014-December")
plt.ylabel("Sea Level in mm")
plt.title("Sea Level Rise")
plt.grid(False)
plt.show()
