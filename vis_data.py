
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import os
k = 4
# load the data
data = []
test_files = os.listdir("val_real")

for i in range(k):
    with open(f"val_real/{test_files[i]}", "rb") as f:
        data.append(pickle.load(f))

# plot the camera view of current step for the k examples
fig, axis = plt.subplots(2, k, figsize=(4*k, 4*2))
for i in range(k):
    axis[0,i].imshow(data[i]["camera"])
    axis[0,i].axis("off")

    axis[1, i].plot(data[i]["sdc_history_feature"][:, 0], data[i]["sdc_history_feature"][:, 1], "o-", color="gold", label="Past")
    axis[1, i].plot(data[i]["sdc_future_feature"][:, 0], data[i]["sdc_future_feature"][:, 1], "o-", color="green", label="Future")
    axis[1, i].legend()
    axis[1, i].axis("equal")
fig.savefig("data_vis.png", dpi=300, bbox_inches='tight')
plt.show()