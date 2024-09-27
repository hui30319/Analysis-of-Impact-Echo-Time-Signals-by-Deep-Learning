# Import Packages
import sys
import os
import random

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data import EchoDataset
from utils.process import log, training
from models.tcn import TCN

def same_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
same_seeds(1126)

# Model
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(device)

model = TCN([25]*5, 8).to(device)
model_name = "TCN([25]x5, 8)"

train_cfg = {
    'mode': "train3",
    'vels': [3998],
    'xyzs': ["80x80x30"],
    'cracks': ["32x32"],
    'bcs': ["HS"],
    'depths': [6,8,10,12,14,16,18,20,22,24,26],
    'balls': [],
    'xyz_set': [30,50,30,50],
    'seq_len': 320,             # unit (us)
    'step_len': 1.6,              # unit (us)
    'slide_len': 30,             # unit (us)
    'max_noise': 3,            # unit (%)
    'max_scale': 0,             # unit (%)
    'max_magW': 0,              # unit (%)
    'max_timeW': 0,             # unit (%)
    'batch_size': 32, 
    'n_epochs': 300,
    'lr': 1e-3,
    'patience': 300,
    'split_ratio': 0.9,
    'expansion': 1,
    'seed': 1126,
}

dir_cfg = {
    'save_dir': "../outputs/noise+slide/",
    'dataset_root': f"../data/",
    'training_name': f"{model_name}_{train_cfg['mode']}_noise{train_cfg['max_noise']}_slide{train_cfg['slide_len']}_batch{train_cfg['batch_size']}_lr{train_cfg['lr']}_epoch{train_cfg['n_epochs']}_expansion{train_cfg['expansion']}",
}

save_path = os.path.join(dir_cfg['save_dir'], dir_cfg['training_name']) # create saving directory
os.makedirs(save_path, exist_ok=True)

log_fw = open(f"{save_path}/training_log.txt", 'w') # open log file to save log outputs

# log(train_cfg, log_fw)

for key in list(train_cfg.keys()):
    log(f"{key}: {train_cfg[key]}", log_fw)
# log(dir_cfg, log_fw)  # log your configs to the log file
log(f"\n{model}", log_fw)

# Train Dataset
data_set = EchoDataset(root=dir_cfg['dataset_root'], data_cfg=train_cfg)

log(f"\nTraining number: {len(data_set)}", log_fw)
train_set, valid_set = random_split(data_set,
                                    [int(len(data_set)*train_cfg['split_ratio']),
                                    len(data_set) - int(len(data_set)*train_cfg['split_ratio'])],
                                    generator=torch.Generator().manual_seed(train_cfg['seed']))

train_loader = DataLoader(train_set, batch_size=train_cfg['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=train_cfg['batch_size'], shuffle=True, pin_memory=True)

train_loss_list, valid_loss_list,train_iter,valid_iter = training(train_loader,
                                            valid_loader,
                                            model,
                                            device,
                                            n_epochs=train_cfg['n_epochs'],
                                            lr=train_cfg['lr'],
                                            patience=train_cfg['patience'],
                                            save_path=save_path, log_fw=log_fw)

np.savetxt(f"{save_path}/training_loss.txt", np.column_stack([train_loss_list, valid_loss_list]), fmt=['%.5e','%.5e'])
np.savetxt(f"{save_path}/training_iter.txt", train_iter, fmt=['%.5e'])
np.savetxt(f"{save_path}/validing_iter.txt", valid_iter, fmt=['%.5e'])

plt.plot(train_loss_list, label="train_loss")
plt.plot(valid_loss_list, label="valid_loss")
plt.legend()
plt.title(f"{dir_cfg['training_name']}")
# plt.ylim(0,0.1)
plt.savefig(f"{save_path}/training_loss.jpg", bbox_inches='tight')
