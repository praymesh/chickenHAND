import h5py

def print_hdf5_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")

session_path = "/home/pranay23/VPR_model_tests/dino/chickenHAND/data/session_data_20260405_092950"
with h5py.File(f"{session_path}/annotation.hdf5", 'r') as f:
    f.visititems(print_hdf5_structure)
