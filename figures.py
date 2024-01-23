import matplotlib.pyplot as plt
import json

TASKS = ['matbench_dielectric', 'matbench_jdft2d', 'matbench_log_gvrh', 'matbench_log_kvrh',
         'matbench_mp_e_form', 'matbench_mp_gap', 'matbench_perovskites', 'matbench_phonons']


def get_matbench_results(filepath, verbose=True):
    with open(filepath, 'r') as f:
        data = json.load(f)
    task_scores = {}
    for task in TASKS:
        folds = [f'fold_{i}' for i in range(5)]
        maes = [data['tasks'][task]['results'][fold]['scores']['mae'] for fold in folds]
        mapes = [data['tasks'][task]['results'][fold]['scores']['mape'] for fold in folds]
        rmses = [data['tasks'][task]['results'][fold]['scores']['rmse'] for fold in folds]
        merrs = [data['tasks'][task]['results'][fold]['scores']['max_error'] for fold in folds]
        task_scores[task] = {'mae': np.mean(maes), 'mape': np.mean(mapes),
                             'rmse': np.mean(rmses), 'merr': np.mean(merrs)}
    if verbose:
        for task in TASKS:
            print(f'{task_scores[task]}')
    return task_scores


def plot_truth_vs_prediction(predicted_value, true_value, title="", x_axis_label="Ground Truth", y_axis_label="Prediction", filename=""):
    plt.scatter(true_value, predicted_value, c='crimson', s=0.7)
    p1 = max(max(predicted_value), max(true_value))
    p2 = min(min(predicted_value), min(true_value))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.ylabel(y_axis_label)
    plt.xlabel(x_axis_label)
    plt.title(title)
    plt.savefig(f"./figures/{filename}.png")