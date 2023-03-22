from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

class train_losses():
    def __init__(self, model, device, train_loader, train_stats, optimizer, total_epochs):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.stats = train_stats
        self.optimizer = optimizer
        self.total_epochs = total_epochs


    def s10_train(self, current_epoch, tb_writer):
        self.model.train()
        pbar = tqdm(self.train_loader)
        clip_norm = True
        scaler = torch.cuda.amp.GradScaler()
        criterion = nn.CrossEntropyLoss()
        train_loss, correct, processed = 0, 0, 0
        torch.autograd.set_detect_anomaly(True)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            with torch.cuda.amp.autocast():
                y_pred = self.model(data)
                train_loss = criterion(y_pred, target)

            scaler.scale(train_loss).backward()
            if clip_norm:
                scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()            

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc=f'Loss={train_loss.item() :0.4f} Batch={batch_idx} Train Acc={100 * correct / processed:0.2f}')

            train_iter = current_epoch * len(self.train_loader) + (batch_idx + 1)  # To find the iteration at which training is

            tb_writer.add_scalar('Train loss', round(train_loss.item(), 4), global_step=train_iter)

            self.stats(round(train_loss.item(), 6), 'train_loss')

        train_acc = round((100. * correct / len(self.train_loader.dataset)), 2)
        self.stats(train_acc, 'train_acc')
        tb_writer.add_scalar('Acc/Train', train_acc, global_step=current_epoch)

        print(f'Train set: Epoch : {current_epoch + 1}/{self.total_epochs} Average loss: {train_loss :.4f}, Train Accuracy: {train_acc}')
