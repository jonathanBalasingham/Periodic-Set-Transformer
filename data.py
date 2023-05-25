import pandas as pd
import os

def to_individual_cifs(path):
    name = os.path.basename(path).split(".")[0]
    with open(path) as f:
        cif = f.read()

    cifs = cif.split("#END")
    os.mkdir("./data/" + name)
    for i, c in enumerate(cifs):
        with open(os.path.join("./" + name, str(i) + ".cif"), "w") as fh:
            fh.write(c)