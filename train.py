import shutil
import shutil
import time
import numpy as np
np.random.seed(0)
from sklearn import metrics
from torch.autograd import Variable

from data import *


def train(train_loader, model, criterion, optimizer, epoch, normalizer, cuda=True, print_epoch=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if cuda:
            input_var = [Variable(i).cuda(non_blocking=True) for i in input]
        else:
            input_var = [Variable(i) for i in input]

        target_normed = normalizer.norm(target)

        if cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        output = model(input_var)
        loss = criterion(output, target_var)

        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if print_epoch and i % 25 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, mae_errors=mae_errors)
            )


def validate(val_loader, model, criterion, normalizer, test=False, return_pred=False, cuda=True, return_target=False, return_id=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if cuda:
            with torch.no_grad():
                input_var = [Variable(i).cuda(non_blocking=True) for i in input]
        else:
            with torch.no_grad():
                input_var = [Variable(i) for i in input]
        target_normed = normalizer.norm(target)

        if cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        output = model(input_var)
        loss = criterion(output, target_var)

        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 100 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                mae_errors=mae_errors))

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                    mae_errors=mae_errors))

    if return_pred and return_target:
        if return_id:
            return test_preds, test_targets, test_cif_ids
        return test_preds, test_targets
    elif return_pred:
        return test_preds
    return mae_errors.avg


class Normalizer(object):

    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        

class SigmoidNormalizer(object):
    
    def __init__(self, tensor) -> None:
        self.min = torch.min(tensor)
        self.max = torch.max(tensor)

    def norm(self, tensor) -> torch.Tensor:
        return (tensor - self.min) / (self.max - self.min)
    
    def denorm(self, normed_tensor) -> torch.Tensor:
        return ((self.max - self.min) * normed_tensor) + self.min

    def state_dict(self):
        return {'min': self.min,
                'max': self.max}

    def load_state_dict(self, state_dict):
        self.min = state_dict['min']
        self.max = state_dict['max']
    
    
def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))


def mse(prediction, target):
    return torch.mean((target - prediction) ** 2)


def rmse(prediction, target):
    return torch.sqrt(mse(prediction, target))


def mape(prediction, target):
    return torch.mean(torch.abs((target - prediction) / target))


def mad(target):
    return torch.mean(torch.abs(target - torch.mean(target)))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_encoder_checkpoint(state, is_best, filename='encoder_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'encoder_best.pth.tar')

