import json
from matbench.bench import MatbenchBenchmark
import numpy as np
from torcheval.metrics.functional import binary_confusion_matrix
import torch
import gc

TASK_NAMES = ['matbench_mp_gap','matbench_phonons', 'matbench_jdft2d', 'matbench_dielectric', 'matbench_log_gvrh',
         'matbench_log_kvrh', 'matbench_perovskites']

mb = MatbenchBenchmark(autoload=False)

TASKS = [
        mb.matbench_mp_gap,
        mb.matbench_phonons,
        mb.matbench_jdft2d,
        mb.matbench_dielectric,
        mb.matbench_log_gvrh,
        mb.matbench_log_kvrh,
        mb.matbench_perovskites,
        #mb.matbench_mp_e_form,
        #mb.matbench_mp_gap
    ]

FOLDS = [f'fold_{i}' for i in range(5)]


def get_results(results_path):
    with open(results_path, "r") as f:
        data = json.load(f)
    return data


data = get_results("./results.json.gz")
threshold = 0.2

for task, task_name in zip(TASKS, TASK_NAMES):
    print(task_name)
    task.load()
    confusion_matrices = []
    for fold, fold_number in zip(FOLDS, list(range(5))):
        _, test_outputs = task.get_test_data(fold_number, include_target=True)
        property_threshold = np.percentile(test_outputs, threshold)
        target = np.where(test_outputs > property_threshold, 0, 1)
        
        predicted = list(data['tasks'][task_name]['results'][fold]['data'].values())
        predicted = np.array(predicted)
        predicted = np.where(predicted > np.percentile(predicted, threshold), 0, 1)
        confusion_matrices.append(binary_confusion_matrix(torch.Tensor(predicted), torch.Tensor(target).type(torch.int64)))
    confusion_matrices = torch.stack(confusion_matrices)
    cm = torch.sum(confusion_matrices, dim=0)
    print(cm)
    print(f"Accuracy: {(cm[0][0] + cm[1][1]) / torch.sum(cm)}")
    gc.collect()
