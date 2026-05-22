import numpy as np
import json

# Load with allow_pickle in the same environment used to create the file
data = np.load("mean_std_new_state_dict.npz", allow_pickle=True)

stats_dict = {}

for key in data.files:
    item = data[key].item()
    stats_dict[key] = {"mu": float(item["mu"]), "std": float(item["std"])}

# Save as JSON for safe transport and future compatibility
with open("mean_std_recovered.json", "w") as f:
    json.dump(stats_dict, f, indent=2)

print("✅ Saved recovered values to mean_std_recovered.json")

