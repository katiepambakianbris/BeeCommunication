import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("sensor_output.csv")
colors = [
   '#4a2377',
   '#f55f74',
   '#8cc5e3',
   '#0d7d87'
]

plt.figure(figsize=(6,4))
index = 0
# Plot each (f,h) pair
for (f, h), group in df.groupby(["f", "h"]):
    print(f)
    print(h)
    plt.plot(group["distance"], group["sensor"], label=f"f={f}, h={h}", color=colors[index])
    index += 1

plt.rcParams.update({
    "font.size": 12
})
plt.xlabel("Distance")
plt.ylabel("Sensor Value")
# plt.title("Sensor Response vs Distance")
plt.xlim(0, 3)
plt.ylim(0, 1)
plt.legend()
plt.grid()

plt.show()