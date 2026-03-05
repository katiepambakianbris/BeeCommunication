import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/katiepambakian/Documents/BSc Computer Science/Y3/Dissertation/BeeCommunication/src_evol/test/sensor_output.csv")

plt.plot(data["distance"], data["sensor"])
plt.xlabel("Distance")
plt.ylabel("Other Sensor Value")
plt.grid(True)
plt.title("Sensor Response Curve")
plt.show()