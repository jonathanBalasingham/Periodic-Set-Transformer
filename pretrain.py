import time
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from data import *
from matbench_parameters import p
from model import PeriodicSetTransformer, PeSTEncoder, CompositionDecoder, ElementMasker, FineTuner, NeighborDecoder, \
    DistanceDecoder
from train import AverageMeter, mae, save_checkpoint, validate, train, save_encoder_checkpoint, Normalizer
from matbench.bench import MatbenchBenchmark
import pandas as pd


def pretrain(train_loader, encoder, comp_decoder, masker, neighbor_decoder, criterion, optimizer, epoch, cuda=True, print_freq=100):
    """
    Pre-train task 1:
    Masked atom property prediction
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to train mode
    encoder.train()
    comp_decoder.train()
    neighbor_decoder.train()

    end = time.time()
    cycles = 1
    for i, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if cuda:
            input_var = [Variable(i).cuda(non_blocking=True) for i in input]
            coords = Variable(target[1]).cuda(non_blocking=True)
        else:
            input_var = Variable(input)
            coords = Variable(target[1])

        for _ in range(cycles):
            masked = [random.randint(0, m - 1) for m in target[0]]
            masked_ids = Variable(torch.LongTensor(masked)).cuda(non_blocking=True)
            masked_input_comp = masker(input_var[1], masked_ids, mask_type="composition")
            masked_input_str = masker(input_var[0], masked_ids, mask_type="structure")

            _, embedded_inp = encoder((masked_input_str, masked_input_comp, input_var[2]))
            gn, pn, en, cr, ve, fi, ea, bl, av = comp_decoder(embedded_inp, masked_ids)
            # pred_coords = neighbor_decoder(embedded_inp)
            dist_pred = neighbor_decoder(embedded_inp, masked_ids)

            target_var = encoder.af(input_var[1])[torch.arange(input_var[1].shape[0]), masked_ids]
            target_var2 = input_var[0][torch.arange(input_var[0].shape[0]), masked_ids][:, 1:]
            loss1 = criterion[0](gn, target_var[:, :19])
            loss2 = criterion[0](pn, target_var[:, 19:26])
            loss3 = criterion[0](en, target_var[:, 26:36])
            loss4 = criterion[0](cr, target_var[:, 36:46])
            loss5 = criterion[0](ve, target_var[:, 46:58])
            loss6 = criterion[0](fi, target_var[:, 58:68])
            loss7 = criterion[0](ea, target_var[:, 68:78])
            loss8 = criterion[0](bl, target_var[:, 78:82])
            loss9 = criterion[0](av, target_var[:, 82:92])

            comp_loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9) / 9
            structure_loss = criterion[1](target_var2, dist_pred)
            loss = comp_loss + structure_loss
            # measure accuracy and record loss
            # mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target[0].size(0))
            mae_errors.update(loss, target[0].size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                  'Composition Loss: {loss1:.4f}\t'
                  'Structural Loss: {loss2:.4f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, mae_errors=mae_errors,
                loss1=comp_loss.data.cpu(), loss2=structure_loss.data.cpu())
            )


def pretrain_validate(val_loader, encoder, comp_decoder, masker, neighbor_decoder, criterion, optimizer, test=False, cuda=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to evaluate mode
    encoder.eval()
    comp_decoder.eval()
    neighbor_decoder.eval()

    end = time.time()
    best_mae_error = 1e10
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if cuda:
            with torch.no_grad():
                input_var = [Variable(i).cuda(non_blocking=True) for i in input]
                coords = Variable(target[1]).cuda(non_blocking=True)
        else:
            with torch.no_grad():
                input_var = Variable(input)

        masked = [random.randint(0, m - 1) for m in target[0]]
        masked_ids = Variable(torch.LongTensor(masked)).cuda(non_blocking=True)
        masked_input_comp = masker(input_var[1], masked_ids, mask_type="composition")
        masked_input_str = masker(input_var[0], masked_ids, mask_type="structure")

        _, embedded_inp = encoder((masked_input_str, masked_input_comp, input_var[2]))
        gn, pn, en, cr, ve, fi, ea, bl, av = comp_decoder(embedded_inp, masked_ids)
        #pred_coords = neighbor_decoder(embedded_inp)
        dist_pred = neighbor_decoder(embedded_inp, masked_ids)
        target_var = encoder.af(input_var[1])[torch.arange(input_var[1].shape[0]), masked_ids]
        target_var2 = input_var[0][torch.arange(input_var[0].shape[0]), masked_ids][:, 1:]
        loss1 = criterion[0](gn, target_var[:, :19])
        loss2 = criterion[0](pn, target_var[:, 19:26])
        loss3 = criterion[0](en, target_var[:, 26:36])
        loss4 = criterion[0](cr, target_var[:, 36:46])
        loss5 = criterion[0](ve, target_var[:, 46:58])
        loss6 = criterion[0](fi, target_var[:, 58:68])
        loss7 = criterion[0](ea, target_var[:, 68:78])
        loss8 = criterion[0](bl, target_var[:, 78:82])
        loss9 = criterion[0](av, target_var[:, 82:92])

        comp_loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9) / 9
        structure_loss = criterion[1](target_var2, dist_pred)
        loss = comp_loss + structure_loss
        #loss = comp_loss
        # measure accuracy and record loss
        # mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target[0].size(0))
        mae_errors.update(loss, target[0].size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 50 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                  'Composition Loss: {loss1:.4f}\t'
                  'Structural Loss: {loss2:.4f}'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                mae_errors=mae_errors, loss1=comp_loss.data.cpu(), loss2=structure_loss.data.cpu()))


    if test:
        star_label = '**'
    else:
        star_label = '*'
    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                    mae_errors=mae_errors))
    return mae_errors.avg


def finetune(train_loader, encoder, decoder, criterion, optimizer, epoch, normalizer, cuda=True, print_freq=100):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # don't train the encoder anymore
    encoder.eval()
    decoder.train()

    end = time.time()

    for i, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if cuda:
            input_var = [Variable(i).cuda(non_blocking=True) for i in input]
        else:
            input_var = Variable(input)

        target_normed = normalizer.norm(target)

        if cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        weights, embedding = encoder(input_var, pool=False)
        output = decoder(embedding, weights)

        loss = criterion(target_var, output)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, mae_errors=mae_errors)
            )


def finetune_validate(val_loader, encoder, decoder, criterion, normalizer, test=False, return_pred=False, cuda=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    decoder.eval()
    encoder.eval()
    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if cuda:
            with torch.no_grad():
                #input_var = Variable(input).cuda(non_blocking=True)
                input_var = [Variable(i).cuda(non_blocking=True) for i in input]
        else:
            with torch.no_grad():
                input_var = Variable(input)
        target_normed = normalizer.norm(target)

        if cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output

        weights, embedding = encoder(input_var, pool=False)
        output = decoder(embedding, weights)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
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
        if i % 50 == 0:
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
    if return_pred:
        return test_preds
    return mae_errors.avg


def main_pretrain():
    mb = MatbenchBenchmark(autoload=False)
    task = mb.matbench_mp_gap
    #task = mb.matbench_dielectric
    pset = p["pretrain"]
    training_options = pset["training_options"]
    hp = pset["hp"]
    data_options = pset["data_options"]
    task.load()
    fold = 0
    train_inputs, train_outputs = task.get_train_and_val_data(fold)
    test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
    test_size = test_inputs.shape[0]
    val_size = int(len(train_outputs) * training_options["val_ratio"])
    train_size = len(train_outputs) - val_size
    dataset = PretrainData(pd.concat([train_inputs, test_inputs]),
                           k=data_options["k"],
                           collapse_tol=data_options["tol"])
    collate_fn = collate_pretrain_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=training_options["batch_size"],
        train_ratio=None,
        pin_memory=training_options["cuda"],
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        return_test=True)
    orig_atom_fea_len = dataset[0][0].shape[-1]
    encoder = PeSTEncoder(orig_atom_fea_len,
                          hp["fea_len"],
                          num_heads=hp["num_heads"],
                          n_encoders=hp["num_encoders"])
    encoder.cuda()
    encoder.eval()
    comp_decoder = CompositionDecoder(hp["fea_len"])
    comp_decoder.cuda()
    masker = ElementMasker()
    masker.cuda()
    distance_decoder = DistanceDecoder(hp["fea_len"], data_options["k"])
    #neighbor_decoder = NeighborDecoder(hp["fea_len"], data_options["k"] * 3)
    #neighbor_decoder.cuda()
    distance_decoder.cuda()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.L1Loss()
    optimizer = optim.Adam(encoder.parameters(), training_options["lr"],
                           weight_decay=training_options["wd"])
    scheduler = MultiStepLR(optimizer, milestones=training_options["lr_milestones"],
                            gamma=0.1)
    best_mae_error = 1e10
    for epoch in range(training_options["epochs"]):
        pretrain(train_loader, encoder, comp_decoder, masker, distance_decoder, [criterion1, criterion2], optimizer, epoch)
        mae_error = pretrain_validate(val_loader, encoder, comp_decoder, masker, distance_decoder, [criterion1, criterion2], optimizer)
        if mae_error != mae_error:
            print('Exit due to NaN')
            exit(1)
        scheduler.step()
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_encoder_checkpoint({
            'epoch': epoch + 1,
            'state_dict': encoder.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            #  'normalizer': normalizer.state_dict(),
        }, is_best)

    best_checkpoint = torch.load('encoder_best.pth.tar')
    encoder.load_state_dict(best_checkpoint['state_dict'])


def main():
    mb = MatbenchBenchmark(autoload=False)
    tasks = [
        #mb.matbench_dielectric,
        #mb.matbench_log_gvrh,
        #mb.matbench_log_kvrh,
        mb.matbench_mp_gap,
        #mb.matbench_phonons,
    ]
    pset = p["pretrain"]
    hp_encoder = pset["hp"]

    for task in tasks:
        pset = p[task.dataset_name]
        training_options = pset["training_options"]
        hp = pset["hp"]
        data_options = pset["data_options"]
        task.load()
        for fold in task.folds:
            best_mae_error = 1e10
            # Get all the data
            train_inputs, train_outputs = task.get_train_and_val_data(fold)
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            test_size = test_inputs.shape[0]
            val_size = int(len(train_outputs) * training_options["val_ratio"])
            train_size = len(train_outputs) - val_size
            dataset = PDDDataPymatgen(pd.concat([train_inputs, test_inputs]),
                                      pd.concat([train_outputs, test_outputs]),
                                      k=data_options["k"],
                                      collapse_tol=data_options["tol"])

            collate_fn = collate_pool
            train_loader, val_loader, test_loader = get_train_val_test_loader(
                dataset=dataset,
                collate_fn=collate_fn,
                batch_size=training_options["batch_size"],
                train_ratio=None,
                pin_memory=training_options["cuda"],
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                return_test=True)
            orig_atom_fea_len = dataset[0][0].shape[-1]
            encoder = PeSTEncoder(orig_atom_fea_len,
                                  hp_encoder["fea_len"],
                                  num_heads=hp_encoder["num_heads"],
                                  n_encoders=hp_encoder["num_encoders"])
            best_checkpoint = torch.load('encoder_best.pth.tar')
            encoder.load_state_dict(best_checkpoint['state_dict'])
            encoder.cuda()
            sample_data_list = [dataset[i] for i in
                                sample(range(train_outputs.shape[0]), 500)]
            _, sample_target, _ = collate_pool(sample_data_list)
            normalizer = Normalizer(sample_target)
            model = FineTuner(hp_encoder["fea_len"])
            model.cuda()
            criterion = nn.L1Loss()
            optimizer = optim.Adam(model.parameters(), training_options["lr"],
                                   weight_decay=training_options["wd"])
            scheduler = MultiStepLR(optimizer, milestones=training_options["lr_milestones"],
                                    gamma=0.1)
            for epoch in range(training_options["epochs"]):
                finetune(train_loader, encoder, model, criterion, optimizer, epoch, normalizer)
                mae_error = finetune_validate(val_loader, encoder, model, criterion, normalizer)
                if mae_error != mae_error:
                    print('Exit due to NaN')
                    exit(1)
                scheduler.step()
                is_best = mae_error < best_mae_error
                best_mae_error = min(mae_error, best_mae_error)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_mae_error': best_mae_error,
                    'optimizer': optimizer.state_dict(),
                    'normalizer': normalizer.state_dict(),
                }, is_best)
            print('---------Evaluate Model on Test Set---------------')
            best_checkpoint = torch.load('model_best.pth.tar')
            model.load_state_dict(best_checkpoint['state_dict'])
            predictions = finetune_validate(test_loader, encoder, model, criterion, normalizer, test=True, return_pred=True)
            task.record(fold, predictions)
        print(task.scores)


if __name__ == "__main__":
    #main_pretrain()
    main()
