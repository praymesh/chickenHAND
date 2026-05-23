from huggingface_hub import snapshot_download

repo_id = "fpvlabs/stera-10m"  
folder_name = "session_data_20260328_234041" 
'''folder used as a part of train dataset : 
session_data_20260328_234041/
session_data_20260405_092950/

''' 

# i have only considered kitchen videos to make the train data of only one type  
local_dir = "/home/pranay23/VPR_model_tests/dino/chickenHAND/data"  

# Adding `*/` before the folder name ensures it catches it wherever it sits
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=[f"*{folder_name}/*", f"{folder_name}/*"],
    local_dir=local_dir
)

print(f"Download sequence finished for target: {folder_name}")
