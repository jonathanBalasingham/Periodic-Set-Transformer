import pickle
import random
import time
from random import sample

random.seed(0)
from matbench.bench import MatbenchBenchmark
from model import PeriodicSetTransformer
from data import PDDDataPymatgen, collate_pool, get_train_val_test_loader
import torch.nn as nn
import torch.optim as optim
import torch

from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from train import train, validate, save_checkpoint, Normalizer
import pandas as pd
from matbench_parameters import p
import numpy as np


def get_model(orig_atom_fea_len, hp, cuda=True):
    model = PeriodicSetTransformer(orig_atom_fea_len,
                                   hp["fea_len"],
                                   num_heads=hp["num_heads"],
                                   n_encoders=hp["num_encoders"],
                                   decoder_layers=hp["num_decoder"],
                                   dropout=hp["dropout"],
                                   components=["composition", "pdd"],
                                   attention_dropout=hp["attention_dropout"],
                                   use_cuda=cuda,
                                   use_weighted_pooling=True,
                                   use_weighted_attention=True)
    if cuda:
        model.cuda()
    return model


def get_data(train_inputs, train_outputs, test_inputs, test_outputs, data_options):
    dataset = PDDDataPymatgen(pd.concat([train_inputs, test_inputs]),
                              pd.concat([train_outputs, test_outputs]),
                              k=data_options["k"],
                              collapse_tol=data_options["tol"],
                              collapse=True)
    return dataset


def run_fold(fold, task, training_options, data_options, hp, use_cuda=True, suffix=""):
    best_mae_error = 1e10
    train_inputs, train_outputs = task.get_train_and_val_data(fold)
    test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
    test_size = test_inputs.shape[0]
    val_size = int(len(train_outputs) * training_options["val_ratio"])
    train_size = len(train_outputs) - val_size
    dataset = get_data(train_inputs, train_outputs, test_inputs, test_outputs * 0, data_options)
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
        num_workers=0,
        return_test=True)
    orig_atom_fea_len = dataset[0][0].shape[-1]
    # sample_size = min(train_outputs.shape[0], 5000)
    # sample_data_list = [dataset[i] for i in sample(range(train_outputs.shape[0]), sample_size)]
    sample_data_list = [dataset[i] for i in range(train_outputs.shape[0])]
    _, sample_target, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)
    model = get_model(orig_atom_fea_len, hp, cuda=use_cuda)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), training_options["lr"],
                           weight_decay=training_options["wd"])
    scheduler = MultiStepLR(optimizer, milestones=training_options["lr_milestones"],
                            gamma=0.1)
    start_time = time.time()
    for epoch in range(training_options["epochs"]):
        train(train_loader, model, criterion, optimizer, epoch, normalizer, cuda=use_cuda, print_epoch=epoch % 50 == 0)
        # mae_error = validate(val_loader, model, criterion, normalizer)
        # if mae_error != mae_error:
        #    print('Exit due to NaN')
        #    exit(1)
        scheduler.step()
        # is_best = mae_error < best_mae_error
        # best_mae_error = min(mae_error, best_mae_error)
    end_time = time.time()
    print(f"Fold took {(end_time - start_time) / 60} minutes")
    print('---------Evaluate Model on Test Set---------------')
    start_time_pred = time.time()
    predictions = validate(test_loader, model, criterion, normalizer, test=True, return_pred=True, cuda=use_cuda)
    end_time_pred = time.time()
    print(f"Prediction time: {end_time_pred - start_time_pred} seconds")

    with open(f"{task.dataset_name}_fold{fold}_predictions_compute_times_{suffix}.txt", "w") as f:
        f.write(f"Training time: {(end_time - start_time) / 60} minutes\n")
        f.write(f"Prediction time: {end_time_pred - start_time_pred} seconds\n")

    return predictions


def main(verbose=True, suffix=""):
    mb = MatbenchBenchmark(autoload=False)

    tasks = [
        mb.matbench_phonons,
        mb.matbench_jdft2d,
        mb.matbench_dielectric,
        mb.matbench_log_gvrh,
        mb.matbench_log_kvrh,
        mb.matbench_perovskites,
        mb.matbench_mp_e_form,
        mb.matbench_mp_gap
    ]

    for task in tasks:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(3)
        
        pset = p[task.dataset_name]
        training_options = pset["training_options"]
        hp = pset["hp"]
        data_options = pset["data_options"]
        task.load()
        fold_times = []
        for fold in task.folds:
            st = time.time()
            predictions = run_fold(fold, task, training_options, data_options, hp, use_cuda=torch.cuda.is_available(), suffix=suffix)
            end = time.time()
            if verbose:
                print(f"Fold took {end-st} seconds")
                fold_times.append(end-st)
            
            with open(f"{task.dataset_name}_fold{fold}_predictions_{suffix}", "wb") as f:
                pickle.dump(predictions, f)
            
            task.record(fold, predictions)
        
        if verbose:
            print(task.scores)
        
        with open(f"{task.dataset_name}_mat2vec_v2_results_{suffix}.txt", "w") as f:
            f.write(str(task.scores))
            f.write("\n")
            f.write(str(fold_times))
            f.write("\n")
            f.write(str(np.mean(fold_times)))

    my_metadata = {
        "PeST": "v0.2",
        "configuration": p
    }
    mb.add_metadata(my_metadata)
    mb.to_file(f"results_v2_{'_'.join(t.dataset_name for t in tasks)}_{suffix}.json.gz")



if __name__ == "__main__":
    main()
