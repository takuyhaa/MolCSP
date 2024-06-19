conformer_mode = 'predefined' # 'predefined', 'search'

# For search mode
smiles = "CC(=O)c1ccc2cc3cccc(C(C)=O)c3cc2c1"  # SMILES
num_conformers = 5  # 生成するコンフォメーションの数

# For structure generation
num_structures = 10
list_numIons = [4,4] #[1, 2, 3, 4]
root_folder = f'../MAPbI3/'
sg_mode = 'all' #'sg99'
model_name = 'CHGNet'

# For evaluation
reference = 'ANTCEN21'
cluster = 15
