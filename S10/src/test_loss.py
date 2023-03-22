import torch
import torch.nn as nn
import torch.nn.functional as F

class test_losses():
  def __init__(self, model, device, test_loader, test_stats, total_epochs):
      self.model       = model
      self.device      = device
      self.test_loader = test_loader
      self.stats       = test_stats
      self.total_epochs = total_epochs

  def s10_test(self, current_epoch, tb_writer):
      self.model.eval()
      test_loss, correct, count_wrong = 0, 0, 0
      criterion = nn.CrossEntropyLoss(reduction='sum')
      torch.autograd.set_detect_anomaly(True)
      with torch.no_grad():
          for data, target in self.test_loader:
              data, target = data.to(self.device), target.to(self.device)
              with torch.cuda.amp.autocast():
                  output = self.model(data)
              test_loss += criterion(output, target).item()  # sum up batch loss
              pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              correct += pred.eq(target.view_as(pred)).sum().item()

              if current_epoch == (self.total_epochs - 1):
                  compare = pred.eq(target.view_as(pred))
                  misclass_idx = (compare == False).nonzero(as_tuple=True)[0].tolist()
                  for i in misclass_idx:
                      self.stats(data[i],'mis_img')
                      self.stats(pred[i].item(),'mis_pred')
                      self.stats(target[i].item(),'mis_lbl')

      test_loss /= len(self.test_loader.dataset)
      self.stats(round(test_loss,6), 'test_loss')
      test_acc = round((100. * correct / len(self.test_loader.dataset)), 2)
      self.stats(test_acc, 'test_acc')

      tb_writer.add_scalar('Test loss', round(test_loss,4), global_step=current_epoch)
      tb_writer.add_scalar('Acc/Test', test_acc, global_step=current_epoch)

      print(f'Test set: Epoch : {current_epoch+1}/{self.total_epochs} Average loss: {test_loss :.4f}, Test Accuracy: {test_acc}')

