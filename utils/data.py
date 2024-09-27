import numpy as np
import torch
import random
import os

from torch.utils.data import Dataset

from .augmentation import jittering, scaling, magnitude_warping, time_warping

# from .trainer import log
# save_path = "."
# os.makedirs(save_path, exist_ok=True)
# log_fw = open(f"{save_path}/log_training.txt", 'w')

cfgs = {
    "train1": [[6], [10]],
    'train2': [[4,6], [6,10]],
    'train3': [[4,6,10], [4,6,10]]
}

def overlap_signal(signal, input_dim):
    signal = signal.detach().cpu().numpy()
    overlapped_signals = []

    for i in range(0, len(signal) - input_dim +1, input_dim // 2):
        overlapped_signals.append(signal[i:i+input_dim])

    return torch.from_numpy(np.array(overlapped_signals))
    
class EchoDataset(Dataset):
    def __init__(self, root, data_cfg, slope=False):
        super().__init__()
        self.mode = data_cfg['mode'][:-1]

        vels = data_cfg['vels']
        print(vels)
        xyzs = data_cfg['xyzs']
        cracks = data_cfg['cracks']
        bcs = data_cfg['bcs']
        depths = data_cfg['depths']
        balls = data_cfg['balls']
        xyz_set = data_cfg['xyz_set']
        

        self.seq_len = data_cfg['seq_len']
        self.step_len = data_cfg['step_len']
        self.slide_len = data_cfg['slide_len']
        self.max_noise = data_cfg['max_noise']
        if self.mode != "test" and self.mode != "exp" and self.mode != "overall" and self.mode != "overallexp":
            self.max_scale = data_cfg['max_scale']
            self.max_magW = data_cfg['max_magW']
            self.max_timeW = data_cfg['max_timeW']

        try:
            if depths:
                a = float(depths[0])
            print("SLOPE FALSE")
            self.slope = False
        except:
            print("SLOPE TRUE")
            self.slope = True
        
        files = []

        if self.mode == "train":
            subroot = os.path.join(root, f"train/")
            for path, _, subdirs in os.walk(subroot):
                for fname in subdirs:
                    if fname.endswith("txt"):
                        fdir = os.path.join(path, fname).replace("\\", "/")
                        *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
                        vel = int(vel)
                        depth = float(depth.split("cm")[0])
                        ball = float(ball.split("mm")[0])

                        if depth <= 16:
                            balls = cfgs[data_cfg['mode']][0]
                        else:
                            balls = cfgs[data_cfg['mode']][1]
                        
                        if (vel in vels) and (xyz in xyzs) and (crack in cracks) and (bc in bcs) and (depth in depths) and (ball in balls):
                            x = float(fname.split("y")[0].split("x")[-1])
                            y = float(fname.split("--")[0].split("y")[-1])
                            if not xyz_set:
                                files.append(fdir)
                            elif xyz_set[0] <= x <= xyz_set[1] and xyz_set[2] <= y <= xyz_set[3]:
                                files.append(fdir)

        elif self.mode == "test":
            subroot = os.path.join(root, f"test/")
            for path, _, subdirs in os.walk(subroot):
                for fname in subdirs:
                    if fname.endswith("txt"):
                        fdir = os.path.join(path, fname).replace("\\", "/")
                        *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
                        vel = int(vel)
                        depth = float(depth.split("cm")[0])
                        ball = float(ball.split("mm")[0])

                        if (not vels or vel in vels) and (not xyzs or xyz in xyzs) and (not cracks or crack in cracks) and (not bcs or bc in bcs) and (not depths or depth in depths) and (not balls or ball in balls):
                            x = float(fname.split("y")[0].split("x")[-1])
                            y = float(fname.split("--")[0].split("y")[-1])
                            if not xyz_set:
                                files.append(fdir)
                            elif xyz_set[0] <= x <= xyz_set[1] and xyz_set[2] <= y <= xyz_set[3]:
                                files.append(fdir)

        elif self.mode == "slab":
            subroot = os.path.join(root, f"slab/")
            for path, _, subdirs in os.walk(subroot):
                for fname in subdirs:
                    # print(fname)
                    if fname.endswith("txt"):
                        fdir = os.path.join(path, fname).replace("\\", "/")
                        *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
                        vel = int(vel)
                        depth = float(depth.split("cm")[0])
                        ball = float(ball.split("mm")[0])

                        if (not vels or vel in vels) and (not xyzs or xyz in xyzs) and (not cracks or crack in cracks) and (not bcs or bc in bcs) and (not depths or depth in depths) and (not balls or ball in balls):
                            x = float(fname.split("y")[0].split("x")[-1])
                            y = float(fname.split("--")[0].split("y")[-1])
                            if not xyz_set:
                                files.append(fdir)
                            elif xyz_set[0] <= x <= xyz_set[1] and xyz_set[2] <= y <= xyz_set[3]:
                                files.append(fdir)

        elif self.mode == "exp":
            subroot = os.path.join(root, f"exp/")
            for path, _, subdirs in os.walk(subroot):
                for fname in subdirs:
                    if fname.endswith("txt"):
                        fdir = os.path.join(path, fname).replace("\\", "/")
                        *_, vel, depth, ball, fname = fdir.split("/")
                        depth = depth.split("cm")[0]
                        try:
                            depth = float(depth)
                            vel = int(vel)
                        except:
                            vel = int(vel)
                            pass
                        ball = float(ball.split("mm")[0])
                        if (not vels or vel in vels) and (not depths or depth in depths) and (not balls or ball in balls):
                            files.append(fdir)

        else:
            subroot = os.path.join(root)
            # print(subroot)
            for path, _, subdirs in os.walk(subroot):

                for fname in subdirs:
                    if fname.endswith("txt"):
                        fdir = os.path.join(path, fname).replace("\\", "/")
                        *_, c, depth, ball, fname = fdir.split("/")
                        try:
                            vel = int(c)
                        except:
                            bc = c
                        # print(vel)
                        depth = depth.split("cm")[0]
                        ball = float(ball.split("mm")[0])
                        try:
                            depth = float(depth)
                        except:
                            pass
                        # print(depth)
                        if (not vels or vel in vels) and (not depths or depth in depths) and (not bcs or bc in bcs):
                            files.append(fdir)

        if self.mode == "train":
            expansion = data_cfg['expansion']
            self.files = np.reshape(files*expansion, (-1,))
        else:
            self.files = np.reshape(files, (-1,))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fdir = self.files[idx]
        seq_time = self.seq_len#*1e-6
        slide_time = self.slide_len#*1e-6
        step_time = self.step_len#*1e-6
        max_noise = self.max_noise/100

        if self.mode == "exp" or self.mode == "overallexp":
            *_, vel, depth, ball, fname = fdir.split("/")
            bc = "exp"
            data = np.loadtxt(fdir, skiprows=0)

        else:
            *_, vel, xyz, crack, bc, depth, ball, fname = fdir.split("/")
            data = np.loadtxt(fdir, skiprows=1)
        
        try:
            a = float(depth.split("cm")[0])
            self.slope = False
        except:
            self.slope = True

        # normalized and sliding
        time = data[:, 0]
        disp = data[:, 1]#/np.abs(np.min(data[:, 1]))

        # fixed rigid body motion
        if bc == "FP":
            p_1 = (disp[0]-disp[-1])/(time[0]-time[-1]) 
            disp = disp - p_1 * time
        
        # normalized and sliding
        time += np.random.uniform(-slide_time, slide_time)*1e-6
        disp /= np.abs(np.min(disp))
        # resample
        
        re_time = np.arange(0, seq_time, step_time)*1e-6
        re_disp = np.interp(re_time, time*int(vel)/3998, disp)


        re_disp = jittering(re_disp, np.random.uniform(0, max_noise))

        if self.mode != "test" and self.mode != "exp" and self.mode != "overall" and self.mode != "overallexp":
            max_scale = self.max_scale/100
            max_magW = self.max_magW/100
            max_timeW = self.max_timeW/100

            re_disp = scaling(re_disp, np.random.uniform(0, max_scale))
            re_disp = magnitude_warping(re_disp, np.random.uniform(0, max_magW))
            re_disp = time_warping(re_disp, np.random.uniform(0, max_timeW))

        if self.mode == "exp":
            try:
                label = float(depth.split("cm")[0])
            except:
                x = float(fname.split("y")[0].split("x")[-1])
                y = float(fname.split(".txt")[0].split("--")[0].split("y")[-1])

                if (6 <= x <= 14 and 6 <= y <= 14):
                    if self.slope:
                        d_min=float(depth.split("-")[0])
                        d_max=float(depth.split("-")[1].split("cm")[0])
                        label = d_min + ((d_max-d_min)/8)*(x-6)
                else:
                    label = float(depth.split("cm")[0])

        elif self.mode == "overallexp":
            x = float(fname.split("y")[0].split("x")[-1])
            y = float(fname.split(".txt")[0].split("--")[0].split("y")[-1])
            
            if depth == '12-8cm':
                if (6 <= x <= 14):
                    d_min=float(depth.split("-")[0])
                    d_max=float(depth.split("-")[1].split("cm")[0])
                    label = d_min + ((d_max-d_min)/8)*(x-6)
                else:
                    label=20

            elif (6 <= x <= 14 and 6 <= y <= 14):
                if self.slope:
                    d_min=float(depth.split("-")[0])
                    d_max=float(depth.split("-")[1].split("cm")[0])
                    label = d_min + ((d_max-d_min)/8)*(x-6)

                else:
                    label = float(depth.split("cm")[0])
            else:
                label = 20


        else:   
            try:
                cs = float(depth.split("cm")[0])
                self.slope = False
            except:
                self.slope = True
                
            x = float(fname.split("y")[0].split("x")[-1])
            y = float(fname.split("--")[0].split("y")[-1])
            z_max = float(xyz.split("x")[2])
            xc_min = (float(xyz.split("x")[0]) - float(crack.split("x")[0])) / 2
            xc_max = xc_min + float(crack.split("x")[0])
            yc_min = (float(xyz.split("x")[1]) - float(crack.split("x")[1])) / 2
            yc_max = yc_min + float(crack.split("x")[1])

            if (xc_min <= x <= xc_max and yc_min <= y <= yc_max):
                if self.slope:
                    d_min=float(depth.split("-")[0])
                    d_max=float(depth.split("-")[1].split("cm")[0])
                    label = d_min + ((d_max-d_min)/(xc_max-xc_min))*(x-xc_min)
                else:
                    label = float(depth.split("cm")[0])
            else:
                label = z_max
        # print(re_disp.size)
        return re_disp, label, fdir
    