import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def log(text, log_fw):     # define a logging function to trace the training process
    print(text)
    log_fw.write(str(text) + '\n')
    log_fw.flush()

def training(train_loader, valid_loader, model, device, n_epochs, lr, patience, save_path, log_fw):
    writer = SummaryWriter(log_dir=save_path)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Start training, parameter total: {total}, trainable: {trainable}", log_fw)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss_list = []
    valid_loss_list = []

    stale = 0
    best_loss = np.inf

    for epoch in range(n_epochs):
        # ---------- Training ----------
        model.train()
        train_loss = []
        
        for inputs, labels, _ in tqdm(train_loader):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        train_loss = sum(train_loss)/len(train_loss)
        train_loss_list.append(train_loss)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.flush()
        log(f"[ Train | {epoch+1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}", log_fw)

        # ---------- Validation ----------
        model.eval()
        valid_loss = [] 
        for inputs, labels, _ in tqdm(valid_loader):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            with torch.no_grad():
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss.append(loss.item())
            
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_loss_list.append(valid_loss)
        writer.add_scalar("Loss/valid", valid_loss, epoch)
        writer.flush()

        if valid_loss < best_loss:
            log(f"[ Valid | {epoch+1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f} -> best", log_fw)
            torch.save(model, f"{save_path}/best_model.ckpt")
            torch.save(model.state_dict(), f"{save_path}/best_model_dict.ckpt")
            log(f"Best model found at epoch {epoch+1}, saving model", log_fw)
            best_loss = valid_loss
            stale = 0
        else:
            log(f"[ Valid | {epoch+1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}", log_fw)
            stale += 1
            if stale > patience:
                log(f"No improvment {patience} consecutive epochs, early stopping", log_fw)
                break         

    log("Finish training", log_fw)
    log_fw.close()
    writer.close()
    return train_loss_list, valid_loss_list

def testing(test_loader, model, device):
    model.eval()
    truth = []
    pred = []
    fdir_list = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs, labels, fdirs = batch
            inputs = inputs.to(device, dtype=torch.float)#torch.as_tensor(inputs).float()

            labels = labels.to(device, dtype=torch.float)#torch.as_tensor(labels).float()
            outputs = model(inputs)
            truth.append(labels.cpu())#.detach().cpu())
            pred.append(outputs.cpu())
            fdir_list.append(fdirs)
        truth = torch.cat(truth, dim=0)
        pred = torch.cat(pred, dim=0)
        fdir_list = np.concatenate(fdir_list, axis=0)

    print("Finish testing")
    return np.array(truth), np.array(pred), np.array(fdir_list)
