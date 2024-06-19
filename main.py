import pandas as pd
from pyxtal import pyxtal
import numpy as np
from ase import Atoms
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
# from ase.constraints import FixSymmetry
from ase.io import write
from tqdm import tqdm
import time, os, glob, warnings
import re
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
# from input import smiles, num_conformers, num_structures, root_folder, sg_mode, model_name, list_numIons, conformer_mode
import autode
import concurrent.futures
from pymatgen.core import Structure
import yaml
warnings.simplefilter('ignore')

# Get the path to the config.yaml file from command-line arguments
import sys
if len(sys.argv) > 1:
    config_path = sys.argv[1]

# Load the config.yaml file
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Access the parameters
smiles = config['smiles']
num_conformers = config['num_conformers']
num_structures = config['num_structures']
root_folder = config['root_folder']
sg_mode = config['sg_mode']
model_name = config['model_name']
list_numIons = config['list_numIons']
conformer_mode = config['conformer_mode']

# Preset variables
sg99 = [2, 4, 5, 9, 14, 15, 19, 33, 61, 1, 29, 7, 13, 18, 20, 43, 56, 60, 76, 78, 88, 92, 96, 144, 145, 148, 169, 170]
sg95 = [2, 4, 5, 9, 14, 15, 19, 33, 61, 1, 29]
n_workers = 8

##################################################
##### Generate molecules class ###################
##################################################
def generate_conformers(smiles, num_conformers):
    conformers = []
    mol = autode.Molecule(smiles=smiles)
    mol.populate_conformers(n_confs=num_conformers)
    n = len(mol.conformers)
    print(f"{n} conformers found (n_confs:{num_conformers})")

    for i, ade_mol in enumerate(mol.conformers):
        ele = [atom.atomic_symbol for atom in ade_mol.atoms]
        pos = np.array([atom.coord for atom in ade_mol.atoms])
        atoms = Atoms(ele, pos)
        conformers.append(atoms)

    return conformers


def write_conformers_to_xyz(conformers, output_folder, prefix):
    for i, atoms in enumerate(conformers):
        output_file = f"{output_folder}{prefix}_conformer_{i + 1}.xyz"
        write(output_file, atoms)


##################################################
##### Generate crystals class #####
##################################################

def choice_sg(sg_mode):
    if sg_mode == 'all':
        sg = int(np.random.randint(1, 231, 1))
    elif sg_mode == 'sg99':
        sg = np.random.choice(sg99)
    elif sg_mode == 'sg95':
        sg = np.random.choice(sg95)
    elif type(sg_mode) is int:
        sg = sg_mode

    return sg


def choice_mol(mol_files):
    return str(np.random.choice(mol_files))


def choice_numIons(candidates):
    return np.random.choice(candidates)


def gen_one_structure_from_one_xyz(xyz_file, num_mol_primitive_cell, sg):
    crystal = pyxtal(molecular=True)
    crystal.from_random(
        dim=3,
        group=sg,
        species=[xyz_file],
        numIons=[num_mol_primitive_cell]
    )
    return crystal.to_ase(resort=False), sg, num_mol_primitive_cell


def gen_one_structure_from_multi_xyz(xyz_file, num_mol_primitive_cell, sg):
    crystal = pyxtal(molecular=True)
    crystal.from_random(
        dim=3,
        group=sg,
        species=xyz_file,
        numIons=num_mol_primitive_cell
    )
    return crystal.to_ase(resort=False), sg, num_mol_primitive_cell


def generate_crystal_structures(mol_files, num_structures, sg_mode='all', atoms_dict=None, conformer_mode='search'):
    if atoms_dict is None:
        structures = {}
    else:
        structures = atoms_dict
    if conformer_mode == 'search':
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            while len(structures) < num_structures:
                futures = [
                    executor.submit(
                        gen_one_structure_from_one_xyz,
                        choice_mol(mol_files),
                        choice_numIons(list_numIons),
                        choice_sg(sg_mode)
                    )
                    for _ in range(num_structures - len(structures))
                ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        crystal, sg, num_mol_primitive_cell = future.result()
                        id = len(structures)
                        structures[f'ID_{id}'] = crystal
                        print(f'ID_{id} (SG: {sg}, numIons: {num_mol_primitive_cell}) generated')
                    except:
                        continue
                    # except Exception as e:
                    #     print(f'Error generating structure: {e}')
    elif conformer_mode == 'predefined':
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            while len(structures) < num_structures:
                futures = [
                    executor.submit(
                        gen_one_structure_from_multi_xyz,
                        mol_files,
                        list_numIons,
                        choice_sg(sg_mode)
                    )
                    for _ in range(num_structures - len(structures))
                ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        crystal, sg, num_mol_primitive_cell = future.result()
                        id = len(structures)
                        structures[f'ID_{id}'] = crystal
                        print(f'ID_{id} (SG: {sg}, numIons: {num_mol_primitive_cell}) generated')
                    except:
                        continue

    return structures


def get_prev_atoms_dict(cif_path):
    from pymatgen.io.cif import CifParser
    cifparser = CifParser(cif_path)
    structures = cifparser.get_structures()
    structures_keys = list(cifparser.as_dict().keys())
    atoms_list = [AseAtomsAdaptor.get_atoms(structure) for structure in structures]
    atoms_dict = {}

    for i in range(len(atoms_list)):
        key = structures_keys[i]
        atoms = atoms_list[i]
        atoms_dict[f'{key}'] = atoms

    return atoms_dict


##################################################
##### Optimize crystals class #####
##################################################
def set_calculator(model_name):
    if model_name == 'CHGNet':
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        model = CHGNet().load()
        calculator = CHGNetCalculator(model=model, use_device='cuda:0')
    elif model_name == 'PFP':
        from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
        from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
        model = Estimator(calc_mode=EstimatorCalcMode.CRYSTAL_U0_PLUS_D3, model_version="v5.0.0")
        calculator = ASECalculator(model)
    return calculator


def optimize_one_structure(atoms_in, calculator, logfile=None, trajectory=None):
    atoms = atoms_in.copy()
    atoms.calc = calculator
    # atoms.set_constraint([FixSymmetry(atoms)])
    ecf = FrechetCellFilter(atoms)
    opt = LBFGS(ecf, logfile=logfile, trajectory=trajectory)
    opt.run(fmax=0.02, steps=2000)
    return atoms


def optimize_structure(key, atoms, model_name):
    calculator = set_calculator(model_name)
    opt_atoms = optimize_one_structure(atoms, calculator)
    struct = AseAtomsAdaptor.get_structure(opt_atoms)
    struct = struct.as_dict()
    struct['properties']['energy'] = opt_atoms.get_potential_energy()
    return key, opt_atoms.todict(), struct


def replace_data_block(cif_text, new_data_block):
    pattern = r'(?<=\n)data_.*?(?=\n)'
    replaced_text = re.sub(pattern, new_data_block, cif_text, count=1)
    return replaced_text


def write_cif(atoms_dict, output_file):
    with open(output_file, 'w') as f:
        for key, atoms in atoms_dict.items():
            struct = AseAtomsAdaptor.get_structure(atoms)
            cif = CifWriter(struct, symprec=0.1)
            new_data_block = f'data_{key}'
            modified_cif = replace_data_block(str(cif), new_data_block)
            f.write(modified_cif)


##################################################
# Bayesian optimization for structure generation & optimization
##################################################

def bayes_get_opt_structures(mol_files, num_structures, model_name):
    from bayes_opt import BayesianOptimization
    from bayes_opt.util import UtilityFunction

    def objective_function(mol_file_idx, num_mol_primitive_cell_idx, sg_idx):
        mol_file = mol_files[int(mol_file_idx)]
        num_mol_primitive_cell = list_numIons[int(num_mol_primitive_cell_idx)]
        sg = sg95[int(sg_idx)]
        print(sg, num_mol_primitive_cell, mol_file)

        max_attempts = 5
        attempt_count = 0

        while attempt_count < max_attempts:
            try:
                atoms, sg, num_mol = gen_one_structure_from_one_xyz(mol_file, num_mol_primitive_cell, sg)
                break
            except:
                attempt_count += 1
                continue

        if attempt_count == max_attempts:
            print(f"Failed to generate crystal structure after {max_attempts} attempts.")
            return 0

        key = f'ID_{len(optimized_structures)}'
        print(key)
        _, opt_atoms_dict, struct_dict = optimize_structure(key, atoms, model_name)
        optimized_structures[key] = (opt_atoms_dict, struct_dict, sg, num_mol)
        energy = struct_dict['properties']['energy']
        return -energy  # ベイズ最適化は最大化を目指すため、エネルギーの符号を反転

    optimized_structures = {}
    pbounds = {'mol_file_idx': (0, len(mol_files) - 1),
               'num_mol_primitive_cell_idx': (0, len(list_numIons) - 1),
               'sg_idx': (0, len(sg95) - 1)}

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=10
    )
    optimizer.maximize(init_points=10, n_iter=num_structures - 10)
    return optimized_structures


##################################################
# Press the green button in the gutter to run the script.
##################################################
if __name__ == '__main__':
    start = time.time()

    init_cif_path = root_folder + 'init_structures.cif'
    opt_cif_path = root_folder + 'opt_structures.cif'
    opt_result_path = root_folder + 'opt_results.csv'
    output_folder = root_folder + 'xyz/'

    # First structure generation & optimization
    if not os.path.exists(init_cif_path):

        # Generate conformers
        if conformer_mode == 'search':
            generated_conformers = generate_conformers(smiles, num_conformers)
            os.makedirs(output_folder, exist_ok=True)
            write_conformers_to_xyz(generated_conformers, output_folder, smiles)
        mol_files = glob.glob(output_folder + '*.xyz')

        # Generate crystal structures
        generated_structures = generate_crystal_structures(mol_files, num_structures, sg_mode=sg_mode,
                                                           conformer_mode=conformer_mode)

        # Save initial structures
        write_cif(generated_structures, init_cif_path)
        elapsed = time.time() - start
        print(f'{num_structures} have been generated. ({elapsed:.1f} sec)')

        # Optimize structures
        opt_atoms_dict, results = {}, {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(optimize_structure, key, atoms, model_name) for key, atoms in
                       generated_structures.items()]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                key, opt_atoms, struct_dict = future.result()
                opt_atoms = Atoms.fromdict(opt_atoms)
                opt_atoms.calc = set_calculator(model_name)
                struct = Structure.from_dict(struct_dict)
                opt_atoms_dict[key] = opt_atoms
                results[key] = {
                    'density': float(struct.density),
                    'energy_per_atom': opt_atoms.get_potential_energy() / opt_atoms.get_global_number_of_atoms(),
                    'sg_symbol': struct.get_space_group_info()[0],
                    'sg_number': struct.get_space_group_info()[1]
                }

        # Save locally optimized structures
        write_cif(opt_atoms_dict, opt_cif_path)
        df_results = pd.DataFrame(results).T
        df_results.to_csv(opt_result_path)
        elapsed = time.time() - start
        print(f'{num_structures} have been optimized. (Total: {elapsed:.1f} sec)')

    # First optimization after structure generation
    elif os.path.exists(init_cif_path) and not os.path.exists(opt_cif_path):
        # Load initial structures
        init_atoms_dict = get_prev_atoms_dict(init_cif_path)
        generated_structures = init_atoms_dict

        # Optimize structures
        opt_atoms_dict, results = {}, {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(optimize_structure, key, atoms, model_name) for key, atoms in
                       generated_structures.items()]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                key, opt_atoms, struct_dict = future.result()
                opt_atoms = Atoms.fromdict(opt_atoms)
                opt_atoms.calc = set_calculator(model_name)
                struct = Structure.from_dict(struct_dict)
                opt_atoms_dict[key] = opt_atoms
                results[key] = {
                    'density': float(struct.density),
                    'energy_per_atom': opt_atoms.get_potential_energy() / opt_atoms.get_global_number_of_atoms(),
                    'sg_symbol': struct.get_space_group_info()[0],
                    'sg_number': struct.get_space_group_info()[1]
                }

        # Save locally optimized structures
        write_cif(opt_atoms_dict, opt_cif_path)
        df_results = pd.DataFrame(results).T
        df_results.to_csv(opt_result_path)
        elapsed = time.time() - start
        print(f'{num_structures} have been optimized. (Total: {elapsed:.1f} sec)')

    # Additional run
    elif os.path.exists(init_cif_path) and os.path.exists(opt_cif_path):

        # Load initial & optimized structures
        init_atoms_dict = get_prev_atoms_dict(init_cif_path)
        opt_atoms_dict = get_prev_atoms_dict(opt_cif_path)

        # Additional structure generation & optimization
        if len(init_atoms_dict) == len(opt_atoms_dict):
            add_num_structures = num_structures - len(init_atoms_dict)
            n_already = len(init_atoms_dict)

            # Generate crystal structures
            mol_files = glob.glob(output_folder + '*.xyz')
            generated_structures = generate_crystal_structures(mol_files, num_structures, sg_mode=sg_mode,
                                                               atoms_dict=init_atoms_dict,
                                                               conformer_mode=conformer_mode)

            # Save additional initial structures
            write_cif(generated_structures, init_cif_path)
            elapsed = time.time() - start
            print(f'{n_already} already. New {add_num_structures} have been generated. ({elapsed:.1f} sec)')

            # Pick up added structures
            from itertools import islice

            add_generated_structures = dict(
                islice(generated_structures.items(), len(generated_structures) - add_num_structures,
                       len(generated_structures)))

            # Optimize structures
            results = pd.read_csv(opt_result_path, index_col=0).T.to_dict()
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(optimize_structure, key, atoms, model_name) for key, atoms in
                           add_generated_structures.items()]

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    key, opt_atoms, struct_dict = future.result()
                    opt_atoms = Atoms.fromdict(opt_atoms)
                    opt_atoms.calc = set_calculator(model_name)
                    struct = Structure.from_dict(struct_dict)
                    opt_atoms_dict[key] = opt_atoms
                    results[key] = {
                        'density': float(struct.density),
                        'energy_per_atom': opt_atoms.get_potential_energy() / opt_atoms.get_global_number_of_atoms(),
                        'sg_symbol': struct.get_space_group_info()[0],
                        'sg_number': struct.get_space_group_info()[1]
                    }

            # Save locally optimized structures
            write_cif(opt_atoms_dict, opt_cif_path)
            df_results = pd.DataFrame(results).T
            df_results.to_csv(opt_result_path)
            elapsed = time.time() - start
            print(f'{n_already} already. New {add_num_structures} have been optimized. (Total: {elapsed:.1f} sec)')


        # Additional optimization after structure generation
        else:
            # Pick up added structures
            from itertools import islice

            generated_structures = init_atoms_dict
            n_already = len(opt_atoms_dict)
            add_num_structures = len(init_atoms_dict) - n_already
            add_generated_structures = dict(
                islice(generated_structures.items(), len(generated_structures) - add_num_structures,
                       len(generated_structures)))

            # Optimize structures
            results = pd.read_csv(opt_result_path, index_col=0).T.to_dict()
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(optimize_structure, key, atoms, model_name) for key, atoms in
                           add_generated_structures.items()]

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    key, opt_atoms, struct_dict = future.result()
                    opt_atoms = Atoms.fromdict(opt_atoms)
                    opt_atoms.calc = set_calculator(model_name)
                    struct = Structure.from_dict(struct_dict)
                    opt_atoms_dict[key] = opt_atoms
                    results[key] = {
                        'density': float(struct.density),
                        'energy_per_atom': opt_atoms.get_potential_energy() / opt_atoms.get_global_number_of_atoms(),
                        'sg_symbol': struct.get_space_group_info()[0],
                        'sg_number': struct.get_space_group_info()[1]
                    }

            # Save locally optimized structures
            write_cif(opt_atoms_dict, opt_cif_path)
            df_results = pd.DataFrame(results).T
            df_results.to_csv(opt_result_path)
            elapsed = time.time() - start
            print(f'{n_already} already. New {add_num_structures} have been optimized. (Total: {elapsed:.1f} sec)')
