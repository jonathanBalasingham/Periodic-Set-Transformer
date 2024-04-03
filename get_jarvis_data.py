from jarvis.db.figshare import data
import pickle
from jarvis.core.atoms import Atoms
import pandas as pd
import json



def get_data(dataset_name="dft_3d_2021"):
    filename = f"jarvis_{dataset_name}_pymatgen_structures"
    d = data(dataset_name)
    jids = [i['jid'] for i in d]
    a = [Atoms.from_dict(i['atoms']).pymatgen_converter() for i in d]
    properties = ["formation_energy_peratom", "optb88vdw_bandgap", "optb88vdw_total_energy", 
                    "ehull", "mbj_bandgap", "bulk_modulus_kv", "shear_modulus_gv", 'magmom_oszicar', 
                    'magmom_outcar', "slme", "spillage", "kpoint_length_unit",
                    'epsx', 'epsy', 'epsz', 'mepsx', 'mepsy', 'mepsz',
                    'dfpt_piezo_max_dij', 'dfpt_piezo_max_eij', "exfoliation_energy", "max_efg",
                    "avg_elec_mass", "avg_hole_mass", "n-Seebeck", "n-powerfact",
                    "p-Seebeck", "p-powerfact"]
    
    df = pd.DataFrame(d)
    df = df[properties]
    pickle.dump((a, df, jids), open(filename, "wb"))
    return a, df, jids
    
    
def get_old_data():
    d = json.load(open("./jdft_3d-8-18-2021.json", "r"))
    filename = f"jarvis_dft_3d_pymatgen_structures_old"
    a = [Atoms.from_dict(i['atoms']).pymatgen_converter() for i in d]
    properties = ["formation_energy_peratom", "optb88vdw_bandgap", "optb88vdw_total_energy", 
                    "ehull", "mbj_bandgap", "bulk_modulus_kv", "shear_modulus_gv", 'magmom_oszicar', 
                    'magmom_outcar', "slme", "spillage", "kpoint_length_unit",
                    'epsx', 'epsy', 'epsz', 'mepsx', 'mepsy', 'mepsz',
                    'dfpt_piezo_max_dij', 'dfpt_piezo_max_eij', "exfoliation_energy", "max_efg",
                    "avg_elec_mass", "avg_hole_mass", "n-Seebeck", "n-powerfact",
                    "p-Seebeck", "p-powerfact"]
    
    df = pd.DataFrame(d)
    df = df[properties]
    pickle.dump((a, df), open(filename, "wb"))    