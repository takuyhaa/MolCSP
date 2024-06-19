from ccdc import io
from ccdc.crystal import PackingSimilarity
import pandas as pd
import numpy as np
from tqdm import tqdm


def get_csd_structure(refcode):
    entry_reader = io.EntryReader('CSD')
    structure = entry_reader.entry(refcode)
    return structure.crystal

def get_cif_structure(path_cif):
    structure = io.CrystalReader(path_cif, format='cif')
    return structure[0]

def get_pred_structures(path_cif):
    structures = io.CrystalReader(path_cif, format='cif')
    return structures

def calc_rmsd(exp_structure, pred_structures, cluster=15):
    similarity_engine = PackingSimilarity()
    similarity_engine.settings.allow_molecular_differences = False
    similarity_engine.settings.match_entire_packing_shell = False
    similarity_engine.settings.packing_shell_size = cluster

    data_dict = {}
    for i in tqdm(range(len(pred_structures))):
        h = similarity_engine.compare(exp_structure, pred_structures[i])
        identifier = pred_structures[i].identifier
        if h is None:
            data_dict[identifier] = {'Nmatch': np.nan, f'RMSD{cluster}': np.nan}
        else:
            data_dict[identifier] = {'Nmatch': h.nmatched_molecules, f'RMSD{cluster}': round(h.rmsd, 4)}
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    return df

if __name__ == '__main__':

    try:
        exp_structure = get_csd_structure(reference)
    except:
        exp_structure = get_cif_structure(reference)

    pred_structures = get_pred_structures(root_folder + 'opt_structures.cif')
    df_rmsd = calc_rmsd(exp_structure, pred_structures, cluster)
    df_result = pd.read_csv(root_folder + 'opt_results.csv', index_col=0)
    df_result_join = df_result.join(df_rmsd, how='left')
    df_result_join.to_csv(root_folder + f'opt_results_rmsd{cluster}_{reference}.csv')