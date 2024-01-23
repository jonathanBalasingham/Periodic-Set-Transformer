import amd
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn
import random

seaborn.set_style("darkgrid")


def plot_mds(crystals=None, samples=100, k=10):
    if not crystals:
        crystals = ["./data/T2_Predicted_Structures.cif",
                    "./data/S2_Predicted_Structures.cif",
                    "./data/P1_Predicted_Structures.cif"]

    periodic_sets = []

    for crystal in crystals:
        r = amd.CifReader(crystal)
        i = 0
        for ps in r:
            print(ps)
            periodic_sets.append(ps)
            i += 1
            if i > samples:
                break

    names = [p.name for p in periodic_sets]
    c = [n.split("_")[1] for n in names]
    energies = [float(n.split("_")[0]) for n in names]
    pdds = [amd.PDD(ps, k=k) for ps in periodic_sets]
    distances = amd.PDD_pdist(pdds)
    distances = squareform(distances)
    model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
    output = model.fit_transform(distances)
    plot = seaborn.scatterplot(x=output[:, 0], y=output[:, 1], hue=c, palette=seaborn.color_palette("husl", 3))
    plt.savefig("./figures/scatter2d.png")
    plt.show()
    return plot


def plot_truth_vs_prediction(predicted_value, true_value):
    plt.scatter(true_value, predicted_value, c='crimson', s=0.7)
    p1 = max(max(predicted_value), max(true_value))
    p2 = min(min(predicted_value), min(true_value))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.ylabel("Prediction")
    plt.xlabel("Ground Truth")
    plt.show()


def plot_energy_scatter(crystals="./data/T2_Predicted_Structures.cif", samples=100, k=10):
    periodic_sets = []
    from os.path import basename
    r = amd.CifReader(crystals)
    i = 0
    periodic_sets = list(r)
    periodic_sets = random.sample(periodic_sets, samples)
    print("Samples found: " + str(i))
    names = [p.name for p in periodic_sets]
    c = [n.split("_")[1] for n in names]
    energies = [float(n.split("_")[0]) for n in names]
    print(energies)
    pdds = [amd.PDD(ps, k=k) for ps in periodic_sets]
    distances = amd.PDD_pdist(pdds)
    distances = squareform(distances)
    model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
    output = model.fit_transform(distances)
    plot = seaborn.scatterplot(x=output[:, 0], y=output[:, 1], hue=energies,
                               palette=seaborn.color_palette("magma", as_cmap=True),
                               legend=None)
    plt.savefig("./figures/scatter2d" + basename(crystals).split(".cif")[-1] + ".png")
    plt.show()
    return plot


def plot_scatter3d(periodic_sets, samples):
    ps = random.sample(periodic_sets, samples)
    names = [p.name for p in ps]
    c = [n.split("_")[1] for n in names]
    energies = [float(n.split("_")[0]) for n in names]
    print(energies)
    pdds = [amd.PDD(p, k=10) for p in ps]
    distances = amd.PDD_pdist(pdds, metric='euclidean')
    distances = squareform(distances)
    model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
    output = model.fit_transform(distances)
    plot = seaborn.scatterplot(x=output[:, 0], y=output[:, 1], hue=energies,
                               palette=seaborn.cubehelix_palette(start=.5, rot=-.5, as_cmap=True),
                               legend=None)
    plt.show()
