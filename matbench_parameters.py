

param_set0 = {
    "hp": {
        "fea_len": 128,
        "num_heads": 2,
        "num_encoders": 2,
        "num_decoder": 1
    },
    "training_options": {
        "lr": 0.0001,
        "wd": 1e-6,
        "lr_milestones": [150],
        "epochs": 200,
        "batch_size": 32,
        "val_ratio": 0.2,
        "cuda": True
    },
    "data_options": {
        "k": 15,
        "tol": 1e-4
    }
}


param_set_pretrain = {
    "hp": {
        "fea_len": 256,  # 64,
        "num_heads": 1,
        "num_encoders": 8,
        "num_decoder": 1
    },
    "training_options": {
        "lr": 1e-5,
        "wd": 0,
        "lr_milestones": [400],
        "epochs": 500,
        "batch_size": 32,
        "val_ratio": 0.1,
        "cuda": True
    },
    "data_options": {
        "k": 15,
        "tol": 1e-4
    }
}


param_set1 = {
    "hp": {
        "fea_len": 64,
        "num_heads": 4,
        "num_encoders": 4,
        "num_decoder": 1
    },
    "training_options": {
        "lr": 0.0001,
        "wd": 1e-5,
        "lr_milestones": [75, 150],
        "epochs": 200,
        "batch_size": 32,
        "val_ratio": 0.1,
        "cuda": True
    },
    "data_options": {
        "k": 15,
        "tol": 1e-4
    }
}


param_set2 = {
    "hp": {
        "fea_len": 64,
        "num_heads": 4,
        "num_encoders": 4,
        "num_decoder": 1
    },
    "training_options": {
        "lr": 0.0001,
        "wd": 1e-5,
        "lr_milestones": [150],
        "epochs": 200,
        "batch_size": 32,
        "val_ratio": 0.1,
        "cuda": True
    },
    "data_options": {
        "k": 15,
        "tol": 1e-4
    }
}


param_set3 = {
    "hp": {
        "fea_len": 256,
        "num_heads": 2,
        "num_encoders": 8,
        "num_decoder": 1
    },
    "training_options": {
        "lr": 0.00001,
        "wd": 0,
        "lr_milestones": [75, 150],
        "epochs": 200,
        "batch_size": 32,
        "val_ratio": 0.05,
        "cuda": True
    },
    "data_options": {
        "k": 15,
        "tol": 1e-4
    }
}

p = {
    "pretrain": param_set_pretrain,
    "matbench_phonons": param_set0,
    "matbench_dielectric": param_set1,
    "matbench_log_gvrh": param_set2,
    "matbench_log_kvrh": param_set2,
    "matbench_mp_gap": param_set3,
    "matbench_mp_e_form": param_set3
}
