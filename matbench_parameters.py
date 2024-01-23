
param_set_small = {
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
        "batch_size": 32,
        "val_ratio": 0.01,
        "cuda": True
    },
    "data_options": {
        "k": 15,
        "tol": 1e-4
    }
}


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
        "val_ratio": 0.01,
        "cuda": True
    },
    "data_options": {
        "k": 15,
        "tol": 1e-4
    }
}

param_set_large = {
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
        "batch_size": 128,
        "val_ratio": 0.01,
        "cuda": True
    },
    "data_options": {
        "k": 15,
        "tol": 1e-4
    }
}


p = {
    "matbench_phonons": param_set_small,
    "matbench_jdft2d": param_set_small,
    "matbench_dielectric": param_set_small,
    "matbench_log_gvrh": param_set,
    "matbench_log_kvrh": param_set,
    "matbench_mp_gap": param_set,
    "matbench_mp_e_form": param_set,
    "matbench_perovskites": param_set,
}