# Import Packages
import random
import os
import sys

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data import EchoDataset
from utils.process import testing

def same_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
same_seeds(1126)

device = "cuda:1" if torch.cuda.is_available() else "cpu"
# "FCN", "InceptionTime", "LSTM_FCN", "LSTM", "LSTMAttention", "MLP", "ResCNN", "ResNet", "TCN", "xresnet1d18""TCN([25]x8, 7)"
max_seq = 320
max_slide = 30
max_noise = 3
max_scale = 0
max_magW = 0
max_timeW = 0

for model_name in [
                    # "MLP(200, [500,500,500], [0.1,0.2,0.2,0.3])",
                    # "FCN([128,256,128], [7,5,3])",
                    # "ResNet([64,128,128], [7,5,3])",            
                    # "TCN([25]x8, 7)",
                    # "TCN([25]x7, 7)",
                    # "TCN([25]x6, 7)",
                    # "TCN([25]x5, 7)",
                    # "TCN([25]x4, 7)",
                    # "TCN([25]x5, 3)",
                    # "TCN([25]x5, 4)",
                    # "TCN([25]x5, 5)",
                    # "TCN([25]x5, 6)",
                    # "TCN([25]x5, 8)",
                    # "TCN([25]x5, 9)",
                    # "TCN([25]x5, 5)",
                    # "TCN([55]x5, 8)",
                    # "TCN([25]x5, 9)",
                    # "TCN([25]x5, 10)",
                    # "TCN([25]x5, 11)",
                    # "TCN([25]x5, 12)",
                    "TCN([25]x5, 8)",
                   ]:
    # nn = int(model_name.split("noise")[-1].split("_")[0])
    # model_name = f"tsai_{t}"
    train_cfg = {
        'mode': "train1",
        'vels': [3998],
        'xyzs': ["80x80x30"],
        'cracks': ["32x32"],
        'bcs': ["HS"],
        'depths': [6,8,10,12,14,16,18,20,22,24,26],
        'balls': [],
        'xyz_set': [30,50,30,50],
        'seq_len': max_seq,             # unit (us)
        'step_len': 1.6,                # unit (us)
        'slide_len':max_slide,          # unit (us)
        'max_noise': max_noise,         # unit (%)
        'max_scale': max_scale,         # unit (%)
        'max_magW': max_magW,
        'max_timeW': max_timeW,
        'batch_size': 32, 
        'n_epochs': 300,
        'lr': 1e-3,
        'patience': 300,
        'split_ratio': 0.9,
        'expansion': 1,
        'seed': 1126,
    }
#_magW{train_cfg['max_magW']}
    dir_cfg = {
        'save_dir': "../outputs/noise+slide/",
        'dataset_root': f"../data/",
        # 'training_name': f"{model_name}_batch{train_cfg['batch_size']}_lr{train_cfg['lr']}_epoch{train_cfg['n_epochs']}_expansion{train_cfg['expansion']}",
        'training_name': f"{model_name}_{train_cfg['mode']}_seq{max_seq}_noise{max_noise}_slide{max_slide}_batch{train_cfg['batch_size']}_lr{train_cfg['lr']}_epoch{train_cfg['n_epochs']}_expansion{train_cfg['expansion']}",
    }

    exp_cfg = {
        'mode': "exp1",
        'vels': [],
        'xyzs': [],
        'cracks': [],
        'bcs': [],
        'depths': [],
        'balls': [],
        'xyz_set': [],
        'seq_len': train_cfg['seq_len'],    # unit (us)
        'step_len': train_cfg['step_len'],  # unit (us)
        'slide_len': 0,     # unit (us)
        'max_noise': 0,     # unit (%)
        'batch_size': 32,
    }

    save_path = os.path.join(dir_cfg['save_dir'], dir_cfg['training_name'])
    print(dir_cfg['training_name'])
    model_best = torch.load(f"{save_path}/best_model.ckpt", map_location=torch.device('cpu')).to(device)

    # Test Dataset
    test_set = EchoDataset(root=dir_cfg['dataset_root'], data_cfg=exp_cfg)

    print(f"Testing number: {len(test_set)}")
    test_loader = DataLoader(test_set, batch_size=exp_cfg['batch_size'], shuffle=False, pin_memory=True)

    truth, pred, fdir_list = testing(test_loader, model_best, device)
    d = dict(truth=truth, pred=pred, fdir=fdir_list)
    df = pd.DataFrame.from_dict(d, orient='index').transpose().astype({'truth': float, 'pred':float})
    df.to_csv(f"{save_path}/testing_noise{exp_cfg['max_noise']}_slide{exp_cfg['slide_len']}_exp.txt", index=False, header=False, float_format="%.5E", sep=" ")

# # #######################################################################################################
    test_cfg = exp_cfg
    test_cfg['mode'] = 'test1'
    test_cfg['max_noise'] = 0
    test_cfg['slide_len'] = 0
    # test_cfg['depths'] = [6,8,10,12,14,16,18,20,22,24,26]
    test_cfg['vels']= [3998]
    test_cfg['balls'] = [6, 10]
    test_cfg['xyz_set'] = [32,48,32,48]
    # Test Dataset
    test_set = EchoDataset(root=dir_cfg['dataset_root'], data_cfg=test_cfg)

    print(f"Testing number: {len(test_set)}")
    test_loader = DataLoader(test_set, batch_size=test_cfg['batch_size'], shuffle=False, pin_memory=True)

    truth, pred, fdir_list = testing(test_loader, model_best, device)
    # for i, fdir in enumerate(fdir_list):
    #     *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
    #     pred[i] = pred[i] * int(vel) / 3998
    d = dict(truth=truth, pred=pred, fdir=fdir_list)
    df = pd.DataFrame.from_dict(d, orient='index').transpose().astype({'truth': float, 'pred':float})
    df.to_csv(f"{save_path}/testing_noise{test_cfg['max_noise']}_slide{test_cfg['slide_len']}.txt", index=False, header=False, float_format="%.5E", sep=" ")
# # # #######################################################################################################
    test_cfg = exp_cfg
    test_cfg['mode'] = 'test1'
    test_cfg['max_noise'] = 5
    test_cfg['slide_len'] = 30
    # test_cfg['depths'] = [6,8,10,12,14,16,18,20,22,24,26]
    test_cfg['vels']= [3998]
    test_cfg['balls'] = [6, 10]
    test_cfg['xyz_set'] = [32,48,32,48]
    # Test Dataset
    test_set = EchoDataset(root=dir_cfg['dataset_root'], data_cfg=test_cfg)

    print(f"Testing number: {len(test_set)}")
    test_loader = DataLoader(test_set, batch_size=test_cfg['batch_size'], shuffle=False, pin_memory=True)

    truth, pred, fdir_list = testing(test_loader, model_best, device)
    # for i, fdir in enumerate(fdir_list):
    #     *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
    #     pred[i] = pred[i] * int(vel) / 3998
    d = dict(truth=truth, pred=pred, fdir=fdir_list)
    df = pd.DataFrame.from_dict(d, orient='index').transpose().astype({'truth': float, 'pred':float})
    df.to_csv(f"{save_path}/testing_noise{test_cfg['max_noise']}_slide{test_cfg['slide_len']}.txt", index=False, header=False, float_format="%.5E", sep=" ")
# # # #######################################################################################################
#     test_cfg = exp_cfg
#     test_cfg['mode'] = 'test1'
#     test_cfg['max_noise'] = 0
#     test_cfg['slide_len'] = 30
#     # test_cfg['depths'] = [6,8,10,12,14,16,18,20,22,24,26]
#     test_cfg['vels']= [3998]
#     test_cfg['balls'] = [6, 10]
#     test_cfg['xyz_set'] = [32,48,32,48]
#     # Test Dataset
#     test_set = EchoDataset(root=dir_cfg['dataset_root'], data_cfg=test_cfg)

#     print(f"Testing number: {len(test_set)}")
#     test_loader = DataLoader(test_set, batch_size=test_cfg['batch_size'], shuffle=False, pin_memory=True)

#     truth, pred, fdir_list = testing(test_loader, model_best, device)
#     # for i, fdir in enumerate(fdir_list):
#     #     *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
#     #     pred[i] = pred[i] * int(vel) / 3998
#     d = dict(truth=truth, pred=pred, fdir=fdir_list)
#     df = pd.DataFrame.from_dict(d, orient='index').transpose().astype({'truth': float, 'pred':float})
#     df.to_csv(f"{save_path}/testing_noise{test_cfg['max_noise']}_slide{test_cfg['slide_len']}.txt", index=False, header=False, float_format="%.5E", sep=" ")
# # # #######################################################################################################
#     test_cfg = exp_cfg
#     test_cfg['mode'] = 'test1'
#     test_cfg['max_noise'] = 0
#     test_cfg['slide_len'] = 50
#     # test_cfg['depths'] = [6,8,10,12,14,16,18,20,22,24,26]
#     test_cfg['vels']= [3998]
#     test_cfg['balls'] = [6, 10]
#     test_cfg['xyz_set'] = [32,48,32,48]
#     # Test Dataset
#     test_set = EchoDataset(root=dir_cfg['dataset_root'], data_cfg=test_cfg)

#     print(f"Testing number: {len(test_set)}")
#     test_loader = DataLoader(test_set, batch_size=test_cfg['batch_size'], shuffle=False, pin_memory=True)

#     truth, pred, fdir_list = testing(test_loader, model_best, device)
#     # for i, fdir in enumerate(fdir_list):
#     #     *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
#     #     pred[i] = pred[i] * int(vel) / 3998
#     d = dict(truth=truth, pred=pred, fdir=fdir_list)
#     df = pd.DataFrame.from_dict(d, orient='index').transpose().astype({'truth': float, 'pred':float})
#     df.to_csv(f"{save_path}/testing_noise{test_cfg['max_noise']}_slide{test_cfg['slide_len']}.txt", index=False, header=False, float_format="%.5E", sep=" ")
# # #######################################################################################################
    # test_cfg = exp_cfg
    # test_cfg['mode'] = 'test1'
    # test_cfg['max_noise'] = 10
    # test_cfg['slide_len'] = 0
    # # test_cfg['depths'] = [6,8,10,12,14,16,18,20,22,24,26]
    # test_cfg['vels']= [3998]
    # test_cfg['balls'] = [6, 10]
    # test_cfg['xyz_set'] = [32,48,32,48]
    # # Test Dataset
    # test_set = EchoDataset(root=dir_cfg['dataset_root'], data_cfg=test_cfg)

    # print(f"Testing number: {len(test_set)}")
    # test_loader = DataLoader(test_set, batch_size=test_cfg['batch_size'], shuffle=False, pin_memory=True)

    # truth, pred, fdir_list = testing(test_loader, model_best, device)
    # # for i, fdir in enumerate(fdir_list):
    # #     *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
    # #     pred[i] = pred[i] * int(vel) / 3998
    # d = dict(truth=truth, pred=pred, fdir=fdir_list)
    # df = pd.DataFrame.from_dict(d, orient='index').transpose().astype({'truth': float, 'pred':float})
    # df.to_csv(f"{save_path}/testing_noise{test_cfg['max_noise']}_slide{test_cfg['slide_len']}.txt", index=False, header=False, float_format="%.5E", sep=" ")
# #####################################################################################################
#     test_cfg = exp_cfg
#     test_cfg['mode'] = 'test1'
#     test_cfg['max_noise'] = 1
#     test_cfg['slide_len'] = 30
#     # test_cfg['depths'] = [6,8,10,12,14,16,18,20,22,24,26]
#     test_cfg['vels']= [3998]
#     test_cfg['balls'] = [6, 10]
#     test_cfg['xyz_set'] = [32,48,32,48]
#     # Test Dataset
#     test_set = EchoDataset(root=dir_cfg['dataset_root'], data_cfg=test_cfg)

#     print(f"Testing number: {len(test_set)}")
#     test_loader = DataLoader(test_set, batch_size=test_cfg['batch_size'], shuffle=False, pin_memory=True)

#     truth, pred, fdir_list = testing(test_loader, model_best, device)
#     # for i, fdir in enumerate(fdir_list):
#     #     *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
#     #     pred[i] = pred[i] * int(vel) / 3998
#     d = dict(truth=truth, pred=pred, fdir=fdir_list)
#     df = pd.DataFrame.from_dict(d, orient='index').transpose().astype({'truth': float, 'pred':float})
#     df.to_csv(f"{save_path}/testing_noise{test_cfg['max_noise']}_slide{test_cfg['slide_len']}.txt", index=False, header=False, float_format="%.5E", sep=" ")
# #####################################################################################################
#     test_cfg = exp_cfg
#     test_cfg['mode'] = 'test1'
#     test_cfg['max_noise'] = 1
#     test_cfg['slide_len'] = 50
#     # test_cfg['depths'] = [6,8,10,12,14,16,18,20,22,24,26]
#     test_cfg['vels']= [3998]
#     test_cfg['balls'] = [6, 10]
#     test_cfg['xyz_set'] = [32,48,32,48]
#     # Test Dataset
#     test_set = EchoDataset(root=dir_cfg['dataset_root'], data_cfg=test_cfg)

#     print(f"Testing number: {len(test_set)}")
#     test_loader = DataLoader(test_set, batch_size=test_cfg['batch_size'], shuffle=False, pin_memory=True)

#     truth, pred, fdir_list = testing(test_loader, model_best, device)
#     # for i, fdir in enumerate(fdir_list):
#     #     *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
#     #     pred[i] = pred[i] * int(vel) / 3998
#     d = dict(truth=truth, pred=pred, fdir=fdir_list)
#     df = pd.DataFrame.from_dict(d, orient='index').transpose().astype({'truth': float, 'pred':float})
#     df.to_csv(f"{save_path}/testing_noise{test_cfg['max_noise']}_slide{test_cfg['slide_len']}.txt", index=False, header=False, float_format="%.5E", sep=" ")

# #####################################################################################################
# #     test_cfg = exp_cfg
# #     test_cfg['mode'] = 'test1'
# #     test_cfg['max_noise'] = 1
# #     test_cfg['slide_len'] = 30
# #     # test_cfg['depths'] = [6,8,10,12,14,16,18,20,22,24,26]
# #     test_cfg['vels']= [3998]
# #     test_cfg['balls'] = [6, 10]
# #     test_cfg['xyz_set'] = [32,48,32,48]
# #     # Test Dataset
# #     test_set = EchoDataset(root=dir_cfg['dataset_root'], data_cfg=test_cfg)

#     print(f"Testing number: {len(test_set)}")
#     test_loader = DataLoader(test_set, batch_size=test_cfg['batch_size'], shuffle=False, pin_memory=True)

#     truth, pred, fdir_list = testing(test_loader, model_best, device)
#     # for i, fdir in enumerate(fdir_list):
#     #     *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
#     #     pred[i] = pred[i] * int(vel) / 3998
#     d = dict(truth=truth, pred=pred, fdir=fdir_list)
#     df = pd.DataFrame.from_dict(d, orient='index').transpose().astype({'truth': float, 'pred':float})
#     df.to_csv(f"{save_path}/testing_noise{test_cfg['max_noise']}_slide{test_cfg['slide_len']}.txt", index=False, header=False, float_format="%.5E", sep=" ")

# #####################################################################################################
#     test_cfg = exp_cfg
#     test_cfg['mode'] = 'test1'
#     test_cfg['max_noise'] = 1
#     test_cfg['slide_len'] = 50
#     # test_cfg['depths'] = [6,8,10,12,14,16,18,20,22,24,26]
#     # test_cfg['vels']= [3998]
#     test_cfg['balls'] = [6, 10]
#     test_cfg['xyz_set'] = [32,48,32,48]
#     # Test Dataset
#     test_set = EchoDataset(root=dir_cfg['dataset_root'], data_cfg=test_cfg)

#     print(f"Testing number: {len(test_set)}")
#     test_loader = DataLoader(test_set, batch_size=test_cfg['batch_size'], shuffle=False, pin_memory=True)

#     truth, pred, fdir_list = testing(test_loader, model_best, device)
#     # for i, fdir in enumerate(fdir_list):
#     #     *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
#     #     pred[i] = pred[i] * int(vel) / 3998
#     d = dict(truth=truth, pred=pred, fdir=fdir_list)
#     df = pd.DataFrame.from_dict(d, orient='index').transpose().astype({'truth': float, 'pred':float})
#     df.to_csv(f"{save_path}/testing_noise{test_cfg['max_noise']}_slide{test_cfg['slide_len']}.txt", index=False, header=False, float_format="%.5E", sep=" ")

# ######################################################################################################
#     test_cfg = exp_cfg
#     test_cfg['mode'] = 'test1'
#     test_cfg['max_noise'] = 3
#     test_cfg['slide_len'] = 30
#     # test_cfg['depths'] = [6,8,10,12,14,16,18,20,22,24,26]
#     # test_cfg['vels']= [3998]
#     # test_cfg['balls'] = [6, 10]
#     test_cfg['xyz_set'] = [32,48,32,48]
#     # # Test Dataset
#     test_set = EchoDataset(root=dir_cfg['dataset_root'], data_cfg=test_cfg)

#     print(f"Testing number: {len(test_set)}")
#     test_loader = DataLoader(test_set, batch_size=test_cfg['batch_size'], shuffle=False, pin_memory=True)

#     truth, pred, fdir_list = testing(test_loader, model_best, device)
#     # for i, fdir in enumerate(fdir_list):
#     #     *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
#     #     pred[i] = pred[i] * int(vel) / 3998
#     d = dict(truth=truth, pred=pred, fdir=fdir_list)
#     df = pd.DataFrame.from_dict(d, orient='index').transpose().astype({'truth': float, 'pred':float})
#     df.to_csv(f"{save_path}/testing_noise{test_cfg['max_noise']}_slide{test_cfg['slide_len']}.txt", index=False, header=False, float_format="%.5E", sep=" ")

############################################################################################################
    test_cfg = exp_cfg
    test_cfg['vels']= []
    test_cfg['mode'] = 'overallexp1'
    dir_cfg['dataset_root'] = f"./data/overallEXP/"
    test_cfg['vels']= []
    test_cfg['max_noise'] = 0
    test_cfg['slide_len'] = 0
    print(test_cfg)
    # Test Dataset
    test_set = EchoDataset(root=dir_cfg['dataset_root'], data_cfg=test_cfg)

    print(f"Testing number: {len(test_set)}")
    test_loader = DataLoader(test_set, batch_size=test_cfg['batch_size'], shuffle=False, pin_memory=True)

    truth, pred, fdir_list = testing(test_loader, model_best, device)
    # for i, fdir in enumerate(fdir_list):
    #     *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
    #     pred[i] = pred[i] * int(vel) / 3998
    d = dict(truth=truth, pred=pred, fdir=fdir_list)
    df = pd.DataFrame.from_dict(d, orient='index').transpose().astype({'truth': float, 'pred':float})
    df.to_csv(f"{save_path}/testing_noise{test_cfg['max_noise']}_slide{test_cfg['slide_len']}_overallexp.txt", index=False, header=False, float_format="%.5E", sep=" ")

################################################################################################################################
    test_cfg = exp_cfg
    test_cfg['mode'] = 'overall1'
    dir_cfg['dataset_root'] = f"./data/overall/"
    test_cfg['max_noise'] = 0
    test_cfg['slide_len'] = 0
    # Test Dataset
    test_set = EchoDataset(root=dir_cfg['dataset_root'], data_cfg=test_cfg)

    print(f"Testing number: {len(test_set)}")
    test_loader = DataLoader(test_set, batch_size=test_cfg['batch_size'], shuffle=False, pin_memory=True)

    truth, pred, fdir_list = testing(test_loader, model_best, device)
    # for i, fdir in enumerate(fdir_list):
    #     *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
    #     pred[i] = pred[i] * int(vel) / 3998
    d = dict(truth=truth, pred=pred, fdir=fdir_list)
    df = pd.DataFrame.from_dict(d, orient='index').transpose().astype({'truth': float, 'pred':float})
    df.to_csv(f"{save_path}/testing_noise{test_cfg['max_noise']}_slide{test_cfg['slide_len']}_overall.txt", index=False, header=False, float_format="%.5E", sep=" ")
# # ################################################################################################################################
#     test_cfg = exp_cfg
#     test_cfg['mode'] = 'overall1'
#     dir_cfg['dataset_root'] = f"./data/overall/"
#     test_cfg['max_noise'] = 3
#     test_cfg['slide_len'] = 30
#     # Test Dataset
#     test_set = EchoDataset(root=dir_cfg['dataset_root'], data_cfg=test_cfg)

#     print(f"Testing number: {len(test_set)}")
#     test_loader = DataLoader(test_set, batch_size=test_cfg['batch_size'], shuffle=False, pin_memory=True)

#     truth, pred, fdir_list = testing(test_loader, model_best, device)
#     # for i, fdir in enumerate(fdir_list):
#     #     *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
#     #     pred[i] = pred[i] * int(vel) / 3998
#     d = dict(truth=truth, pred=pred, fdir=fdir_list)
#     df = pd.DataFrame.from_dict(d, orient='index').transpose().astype({'truth': float, 'pred':float})
#     df.to_csv(f"{save_path}/testing_noise{test_cfg['max_noise']}_slide{test_cfg['slide_len']}_overall.txt", index=False, header=False, float_format="%.5E", sep=" ")

# #     ################################################################################################################################