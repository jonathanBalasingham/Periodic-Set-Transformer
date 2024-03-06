from random import sample

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

mb = MatbenchBenchmark(autoload=False)

best_mae_error = 1e10

tasks = [
    #mb.matbench_dielectric,
    #mb.matbench_log_gvrh,
    #mb.matbench_log_kvrh,
    #mb.matbench_mp_gap,
    mb.matbench_phonons,
    #mb.matbench_jdft2d
]

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
        print(orig_atom_fea_len)
        print(dataset[0][1].shape[-1])
        sample_data_list = [dataset[i] for i in
                            sample(range(train_outputs.shape[0]), train_outputs.shape[0])]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)
        model = PeriodicSetTransformer(orig_atom_fea_len,
                                       hp["fea_len"],
                                       num_heads=hp["num_heads"],
                                       n_encoders=hp["num_encoders"],
                                       decoder_layers=hp["num_decoder"])
        if torch.cuda.is_available():
            model.cuda()
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), training_options["lr"],
                               weight_decay=training_options["wd"])
        scheduler = MultiStepLR(optimizer, milestones=training_options["lr_milestones"],
                                gamma=0.1)
        for epoch in range(training_options["epochs"]):
            train(train_loader, model, criterion, optimizer, epoch, normalizer)
            """mae_error = validate(val_loader, model, criterion, normalizer)
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
            }, is_best)"""
        print('---------Evaluate Model on Test Set---------------')
        #best_checkpoint = torch.load('model_best.pth.tar')
        #model.load_state_dict(best_checkpoint['state_dict'])
        predictions = validate(test_loader, model, criterion, normalizer, test=True, return_pred=True)
        task.record(fold, predictions)
    print(task.scores)


my_metadata = {
    "PeST": "v0.1",
    "configuration": p
}
mb.add_metadata(my_metadata)
mb.to_file("results_bgap.json.gz")