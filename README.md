# MolCSP
MolCSP is a Python program for crystal structure prediction of molecular crystals.


# Tutorial

## For MAPbI3
1. Navigate to the directory containing `main.py` in the terminal.
2. Enter `python main.py example/MAPbI3/config.yaml` in the terminal.
3. Generate N structures:
   - During the process, the generated IDs will be displayed.
   - Upon completion, `init_structures.cif` will be created in `example/MAPbI3/`.
4. Optimize N structures:
   - During the process, a progress bar will be displayed.
   - Upon completion, `opt_structures.cif` and `opt_results.csv` will be created in `example/MAPbI3/`.
5. For subsequent runs in the same folder, change `num_structures` in `config.yaml` (e.g., to M).
   - (M-N) structures will be generated and optimized.
   - The `.cif` and `.csv` files will be automatically updated.

## For benzene
1. Navigate to the directory containing `main.py` in the terminal.
2. Enter `python main.py example/benzene/config.yaml` in the terminal.
3. Generate N structures:
   - Conformer search will be performed, and `.xyz` files will be created in `example/benzene/`.
   - During the process, the generated IDs will be displayed.
   - Upon completion, `init_structures.cif` will be created in `example/benzene/`.
4. Optimize N structures:
   - During the process, a progress bar will be displayed.
   - Upon completion, `opt_structures.cif` and `opt_results.csv` will be created in `example/benzene/`.
5. For subsequent runs in the same folder, change `num_structures` in `config.yaml` (e.g., to M).
   - (M-N) structures will be generated and optimized.
   - The `.cif` and `.csv` files will be automatically updated.
