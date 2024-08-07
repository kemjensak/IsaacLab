import torch

file_path = "/home/irol/IsaacLab/logs/rsl_rl/Shelf_sweep/2024-06-18_23-08-43/exported/policy.pt"
loaded_data = torch.jit.load(file_path, map_location=torch.device("cpu"))


print(loaded_data)
# Check the type of the loaded data
if isinstance(loaded_data, dict):
    print("The file contains a state dictionary.")
    print("Keys in the state dictionary:")
    for key in loaded_data.keys():
        print(key)
        
    # Optionally, print shapes of the tensors
    for key, value in loaded_data.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")