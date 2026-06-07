import h5py
import matplotlib.pyplot as plt
def print_hdf5_structure(name, obj):    
    #for reference , dataset : arrays and group : folders in hdf5 file
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")

session_path = "/home/pranay23/VPR_model_tests/dino/chickenHAND/data/session_data_20260405_092950"

#view stuctuer
# with h5py.File(f"{session_path}/annotation.hdf5", 'r') as f:
#     f.visititems(print_hdf5_structure)


def read_hdf5_dataset(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        if dataset_name in f:
            data = f[dataset_name]
            return data
        else:
            print(f"Dataset '{dataset_name}' not found in the HDF5 file.")
            return None


with h5py.File(f"{session_path}/annotation.hdf5", 'r') as f:
    depth_frame = f['depth/frames'][0]  
    plt.imshow(depth_frame)
    plt.colorbar()
    plt.savefig("depth_frame.png")
    print("\n[✓] Saved depth frame visualization to 'depth_frame.png'")

