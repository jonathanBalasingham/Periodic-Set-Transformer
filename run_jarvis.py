import os.path

from model import PeriodicSetTransformer
from data import JarvisData, collate_pool, get_train_val_test_loader
import torch.nn as nn
import torch.optim as optim
import torch

from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from train import *
import pandas as pd
import pickle
from figures import *
from units import *
import time

param_set = {
    "hp": {
        "fea_len": 128,
        "num_heads": 4,
        "num_encoders": 4,
        "num_decoder": 2,
        "attention_dropout": 0.1,
        "dropout": 0.0
    },
    "training_options": {
        "lr": 0.0001,
        "wd": 1e-5,
        "lr_milestones": [150, 200],
        "epochs": 250,
        "batch_size": 64,
        "cuda": True
    },
    "data_options": {
        "k": 15,
        "tol": 1e-4
    }
}


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
                                   atom_encoding="mat2vec")
    if cuda:
        model.cuda()
    return model


def main(verbose=True):
    jarvis_datapath = "./jarvis_dft_3d_2021_pymatgen_structures"
    if os.path.exists(jarvis_datapath):
        _, properties, _ = pickle.load(open(jarvis_datapath, "rb"))
    use_cuda = torch.cuda.is_available()
    print(f"Using GPU: {use_cuda}")
    d = {"n": [], "MAE": [], "MSE": [], "RMSE": [], "MAPE": [], "MAD": [], "MAD:MAE": [], "Training time": [],
         "Prediction Time": []}
    props_to_run = [
        "exfoliation_energy",
        "dfpt_piezo_max_eij",
        "dfpt_piezo_max_dij",
        "bulk_modulus_kv",
        "shear_modulus_gv",
        "mbj_bandgap",
        "slme",
        "spillage",
        "ehull",
        "formation_energy_peratom",
        "optb88vdw_bandgap",
        "optb88vdw_total_energy",
    ]
    for prop_name in props_to_run:
        best_mae_error = 1e10
        print(f"Running property: {prop_name}")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(3)
        training_options = param_set["training_options"]
        hp = param_set["hp"]
        data_options = param_set["data_options"]
        dataset = JarvisData(jarvis_datapath,
                             prop_name, k=data_options["k"], collapse_tol=data_options["tol"])

        collate_fn = collate_pool
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=training_options["batch_size"],
            train_ratio=None,
            pin_memory=training_options["cuda"],
            val_ratio=0.01,
            test_ratio=0.1,
            train_size=None,
            test_size=None,
            val_size=None,
            return_test=True,
            num_workers=0)
        orig_atom_fea_len = dataset[0][0].shape[-1]

        sample_data_list = [dataset[i] for i in range(len(dataset))]
        d["n"].append(len(sample_data_list))
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

        model = PeriodicSetTransformer(orig_atom_fea_len,
                                       hp["fea_len"],
                                       num_heads=hp["num_heads"],
                                       n_encoders=hp["num_encoders"],
                                       decoder_layers=hp["num_decoder"],
                                       dropout=hp["dropout"],
                                       components=["composition", "pdd"],
                                       attention_dropout=hp["attention_dropout"],
                                       use_cuda=use_cuda,
                                       atom_encoding="mat2vec")
        model.cuda()
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), training_options["lr"],
                               weight_decay=training_options["wd"])
        scheduler = MultiStepLR(optimizer, milestones=training_options["lr_milestones"],
                                gamma=0.1)

        train_time_start = time.time()
        total_val_time = 0
        for epoch in range(training_options["epochs"]):
            train(train_loader, model, criterion, optimizer, epoch, normalizer, cuda=use_cuda,
                  print_epoch=epoch % 25 == 0)
            val_time_start = time.time()
            if epoch > 200:
                mae_error = validate(val_loader, model, criterion, normalizer)
                if mae_error != mae_error:
                    print('Exit due to NaN')
                    exit(1)
                is_best = mae_error < best_mae_error
                best_mae_error = min(mae_error, best_mae_error)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_mae_error': best_mae_error,
                    'optimizer': optimizer.state_dict(),
                    'normalizer': normalizer.state_dict(),
                    'args': param_set
                }, is_best)

            val_time_end = time.time()
            total_val_time += (val_time_end - val_time_start)
            scheduler.step()

        train_time_end = time.time()
        training_time = train_time_end - train_time_start - total_val_time

        print('---------Evaluate Model on Test Set---------------')
        #best_checkpoint = torch.load('model_best.pth.tar')
        #model.load_state_dict(best_checkpoint['state_dict'])
        pred_time_start = time.time()
        predictions, test_targets, ids = validate(test_loader, model, criterion, normalizer, test=True, return_pred=True,
                                             cuda=use_cuda, return_target=True, return_id=True)
        pred_time_end = time.time()
        prediction_time = pred_time_end - pred_time_start
        pickle.dump((predictions, test_targets), open(f"./jarvis_results/{prop_name}_predictions", "wb"))
        pd.DataFrame({"id": ids, "prediction": predictions, "target": test_targets}).to_csv(f"./jarvis_results/{prop_name}_results.csv", index=False)
        predictions = torch.Tensor(predictions)
        test_targets = torch.Tensor(test_targets)
        mean_absolute_error = mae(predictions, test_targets)
        mean_squared_error = mse(predictions, test_targets)
        mean_absolute_percentage_error = mape(predictions, test_targets)
        mean_absolute_deviation = mad(test_targets)
        root_mean_squared_error = rmse(predictions, test_targets)
        d["MAE"].append(mean_absolute_error)
        d["MAPE"].append(mean_absolute_percentage_error)
        d["MSE"].append(mean_squared_error)
        d["RMSE"].append(root_mean_squared_error)
        d["MAD"].append(mean_absolute_deviation)
        d["MAD:MAE"].append(mean_absolute_deviation / mean_absolute_error)
        d["Training time"].append(training_time)
        d["Prediction Time"].append(prediction_time)

        with open(f"./jarvis_results/jarvis_{prop_name}_results.txt", "w") as f:
            f.write(str(prop_name) + "\n")
            f.write(f"Samples: {d['n'][-1]}\n")
            f.write(f"MAE: {mean_absolute_error}\n")
            f.write(f"RMSE: {root_mean_squared_error}\n")
            f.write(f"MSE: {mean_squared_error}\n")
            f.write(f"MAPE: {mean_absolute_percentage_error}\n")
            f.write(f"MAD: {mean_absolute_deviation}\n")
            f.write(f"MAD:MAE  : {mean_absolute_deviation / mean_absolute_error}\n")
            f.write(f"Training time: {training_time}\n")
            f.write(f"Prediction time: {prediction_time}\n")

        #plot_truth_vs_prediction(predictions, test_targets,
        #                         title=property_names[prop_name],
        #                         filename=prop_name)

    df = pd.DataFrame(d)
    df.to_csv("jarvis_results_all.csv", index=False)


if __name__ == "__main__":
    main()
