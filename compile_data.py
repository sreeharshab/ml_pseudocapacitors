import os
import re
import numpy as np
import pandas as pd
import hashlib
import logging
from pipelines import DOS
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
from matminer.featurizers.conversions import ASEAtomstoStructure, StrToComposition
from matminer.featurizers.structure import DensityFeatures, MaximumPackingEfficiency
from matminer.featurizers.composition import ElementProperty
from pymatgen.analysis.local_env import VoronoiNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class get_data:
    def __init__(self, material, addnl_folder_paths=None):
        self.material = material
        self.addnl_folder_paths = addnl_folder_paths
        if self.addnl_folder_paths is None:
            os.chdir(material)
        elif self.addnl_folder_paths is not None:
            pwd = os.getcwd()
            self.addnl_folder_paths = [f"{pwd}"] + self.addnl_folder_paths
            self.addnl_folder_paths = [path+"/" for path in self.addnl_folder_paths]
            for path in self.addnl_folder_paths:
                try:
                    os.chdir(f"{path}"+f"{material}")
                    break
                except FileNotFoundError:
                    pass
        self.formula = self.get_formula()
        self.elements = re.findall(r'([A-Z][a-z]?)\d*', self.formula)
        self.structure = self.get_PMG_structure()
        os.chdir("../")
    
    def get_metals(self):
        metals = [metal for metal in self.elements if metal not in ["O", "S", "C", "N", "Si", "P", "Li"]]
        return metals
    
    def get_bridging_elements(self):
        bridging_elements = [element for element in self.elements if element in ["O", "S", "C", "N", "Si", "P"]]
        return bridging_elements
    
    def get_atoms_and_energy(self, dir_name):
        os.chdir(dir_name)
        atoms = read("OUTCAR@-1")
        energy = atoms.get_potential_energy()
        os.chdir("../")
        return atoms, energy
    
    def get_formula(self, dir_name='Energy_calculation'):
        atoms,_ = self.get_atoms_and_energy(dir_name)
        return str(atoms.symbols)
    
    def get_PMG_structure(self, dir_name="Energy_calculation"):
        atoms,_ = self.get_atoms_and_energy(dir_name)
        data = {
            "ase atoms": [atoms]
        }
        df = pd.DataFrame(data)
        conv = ASEAtomstoStructure()
        df = conv.featurize_dataframe(df, "ase atoms", ignore_errors=True)
        structure = df.at[0, "PMG Structure from ASE Atoms"]
        return structure
    
    def get_dos_data(self):
        try:
            os.chdir("Electronic_calculation")
            dos = DOS()
            band_gap = dos.get_band_gap()
            band_centers = [dos.get_band_centers()[1], dos.get_band_centers()[2]]    
            os.chdir("../")
            return [band_gap] + band_centers
        except FileNotFoundError:
            logging.warning(f"Electronic calculation does not exist for {self.material}. Taking band gap and band centers as 0...")
            return [0,0,0]
    
    def get_max_void_radius(self):
        def distance(coord1, coord2):
            coord1 = np.array(coord1)
            coord2 = np.array(coord2)
            return np.linalg.norm(coord2 - coord1)
        structure = self.structure
        voronoi = VoronoiNN()
        radii = np.array([])
        for index in range(len(structure)):    
            voronoi_polyhedra, voronoi_object = voronoi.get_voronoi_polyhedra(structure, index)
            for poly_info in voronoi_polyhedra.values():
                vertex_indices = poly_info["verts"]
                vertices = voronoi_object.vertices[vertex_indices]
                for vertex in vertices:
                    radii = np.append(radii, round(distance(vertex, structure[index].coords),3))
        return max(radii)
    
    def get_Li_M_Li_O_distance(self, atoms):
        kwargs = {"Mn":2, "Co":2, "Fe":2.5, "Nb":2, "C":1.7, "N":1.7}   # Custom cutoffs
        nat_cut = natural_cutoffs(atoms, **kwargs)
        nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
        nl.update(atoms)
        Li_indices = [atom.index for atom in atoms if atom.symbol=="Li"]
        Li_M_distances = []
        Li_O_distances = []
        for Li_index in Li_indices:
            indices, offsets = nl.get_neighbors(Li_index)
            row = []
            for i, offset in zip(indices, offsets):
                if atoms[i].symbol in self.get_metals():
                    pos = atoms.positions[i] + offset @ atoms.get_cell()
                    dist = ((atoms[Li_index].x - pos[0])**2 + (atoms[Li_index].y - pos[1])**2 + (atoms[Li_index].z - pos[2])**2)**(1/2)
                    row.append(dist)
            row = np.array(row)
            Li_M_distances.append(np.mean(row))
            row = []
            for i, offset in zip(indices, offsets):
                if atoms[i].symbol in self.get_bridging_elements():
                    pos = atoms.positions[i] + offset @ atoms.get_cell()
                    dist = ((atoms[Li_index].x - pos[0])**2 + (atoms[Li_index].y - pos[1])**2 + (atoms[Li_index].z - pos[2])**2)**(1/2)
                    row.append(dist)
            row = np.array(row)
            Li_O_distances.append(np.mean(row))
        Li_M_distances = np.array(Li_M_distances)
        Li_O_distances = np.array(Li_O_distances)
        return round(np.mean(Li_M_distances, axis=0),3), round(np.mean(Li_O_distances, axis=0),3)
    
    def get_Li_charge(self):
        try:
            os.chdir("bader")
            try:
                atoms = read("with_charges.traj")
            except FileNotFoundError:
                Li_charge = 0
                os.chdir("../")
                return Li_charge
            Li_indices = [atom.index for atom in atoms if atom.symbol=="Li"]
            charges = atoms.get_initial_charges()
            Li_charges = []
            for Li_index in Li_indices:
                Li_charges.append(charges[Li_index])
            Li_charges = np.array(Li_charges)
            Li_charge = np.mean(Li_charges)
            os.chdir("../")
            return Li_charge
        except FileNotFoundError:
            Li_charge = 0
            return Li_charge
    
    def get_Li_M_icohp(self):
        try:
            os.chdir("cohp")
            try:
                icohp_ef = [
                    eval(line.split()[-1])
                    for line in open("./ICOHPLIST.lobster").readlines()[1:]
                ]
            except FileExistsError:
                Li_M_icohp = 0
                os.chdir("../")
                return Li_M_icohp
            icohp_ef = [i for index,i in enumerate(icohp_ef) if index%2==0]
            Li_M_icohp = round(sum(icohp_ef)/len(icohp_ef),3)
            os.chdir("../")
            return Li_M_icohp
        except FileNotFoundError:
            Li_M_icohp = 0
            return Li_M_icohp
    
    def get_intercalation_data(self, fhandle):
        atoms, energy = self.get_atoms_and_energy("Energy_calculation")
        volume = atoms.get_volume()
        n_M = len([atom.index for atom in atoms if atom.symbol in self.get_metals()])
        fhandle.write("\tLithium Intercalation:\n")
        os.chdir("Intercalation")
        oswalk = [i for i in os.walk(".")]
        nLifolders = sorted(oswalk[0][1])
        str_format = "{:^15} {:^28} {:^15} {:^23} {:^23} {:^23}\n"
        fhandle.write("\t")
        fhandle.write(str_format.format("Site", "Li Intercalation Energy (eV)", "Charge on Li", f"Average Li-{self.get_metals()} Distance", f"Average Li-{self.get_bridging_elements()} Distance", f"COHP of Li-{self.get_metals()}"))
        for nLifolder in nLifolders:
            match = re.match(r"(\d+)_Li", nLifolder)
            n_Li = int(match.group(1))
            if n_Li/n_M == 0.25 or n_Li/n_M == 0.5:
                os.chdir(nLifolder)
                oswalk = [i for i in os.walk(".")]
                sites = sorted(oswalk[0][1])
                Li_energies, volume_changes, final_Li_charges, Li_M_distances, Li_O_distances, Li_M_icohps = [], [], [], [], [], []
                for site in sites:
                    fhandle.write("\t")
                    os.chdir(f"{site}")
                    atoms_with_Li, energy_with_Li = self.get_atoms_and_energy("geo_opt")
                    volume_with_Li = atoms_with_Li.get_volume()
                    mu_Li = -33.22057791/16 # From DFT calculation.
                    Li_energy = round((energy_with_Li-energy-n_Li*mu_Li)/(n_Li),3)
                    volume_change = volume_with_Li - volume
                    Li_M_Li_O_distances = self.get_Li_M_Li_O_distance(atoms_with_Li)
                    Li_M_distance = Li_M_Li_O_distances[0]
                    Li_O_distance = Li_M_Li_O_distances[1]
                    Li_charge = self.get_Li_charge()
                    Li_M_icohp = self.get_Li_M_icohp()
                    fhandle.write(str_format.format(site, Li_energy, Li_charge, Li_M_distance, Li_O_distance, Li_M_icohp))
                    lists = [Li_energies, volume_changes, final_Li_charges, Li_M_distances, Li_O_distances, Li_M_icohps]
                    values = [Li_energy, volume_change, Li_charge, Li_M_distance, Li_O_distance, Li_M_icohp]
                    for lst, val in zip(lists, values):
                        lst.append(val)
                    os.chdir("../")
                mLei = Li_energies.index(min(Li_energies))  # mLei: minimum Li energy index
                if final_Li_charges[mLei]==0:
                    logging.warning(f"Li charge not available at {nLifolder}. Taking it as 0...")
                if n_Li/n_M == 0.25:
                    data_25 = [Li_energies[mLei], volume_changes[mLei], final_Li_charges[mLei], Li_M_distances[mLei], Li_O_distances[mLei], Li_M_icohps[mLei]]
                elif n_Li/n_M == 0.5:
                    data_50 = [Li_energies[mLei], volume_changes[mLei], final_Li_charges[mLei], Li_M_distances[mLei], Li_O_distances[mLei], Li_M_icohps[mLei]]
                os.chdir("../")
        os.chdir("../")
        return data_25+data_50

    def compile_data(self, fhandle):
        material = self.material
        formula = self.formula
        logging.info(f"Material: {material}, Formula: {formula}")
        fhandle.write(f"{material}\n")
        if self.addnl_folder_paths is None:
            os.chdir(material)
        elif self.addnl_folder_paths is not None:
            for path in self.addnl_folder_paths:
                try:
                    os.chdir(f"{path}"+f"{material}")
                    break
                except FileNotFoundError:
                    pass
        structure = self.structure
        max_void_radius = self.get_max_void_radius()
        intercalation_data = self.get_intercalation_data(fhandle)
        dos_data = self.get_dos_data()
        os.chdir("../")
        return [material, formula, structure] + intercalation_data + [max_void_radius] + dos_data

def compute_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def changes_in_code(hash_path):
    current_hash = compute_file_hash(__file__)
    try:
        with open(hash_path, "r") as f:
            cached_hash = f.read().strip()
    except FileNotFoundError:
        cached_hash = None
    return current_hash != cached_hash

if __name__=="__main__":
    materials = ['LiMn1.5Ni0.5O4', 'LiMn2O4', 'Mo3Nb2O14', 'MoO2', 'TiO2-B', 'TiO2-R', 'VO2-B', 'W3Nb2O14', 'Fe2CN6', 'LiCoO2', 'LiCoPO4', 'LiFeMgPO4', 'LiFePO4', 'LiFeSO4F', 'LiNiO2', 'LiVPO5', 'MoS2', 'NCM333', 'NCM811', 'Nb2O5-B', 'Nb2O5-H', 'Nb2O5-T', 'Nb2O5-TT', 'NbS2', 'TiO2-anatase', 'TiS2', 'V2O5', 'V6O13', 'VO2-M', 'VO2-R', 'Li2FeSiO4', 'Li3+xV2O5']
    if os.path.exists("cached_dataframe.pkl") and not changes_in_code("code_hash.txt"):
        df = pd.read_pickle("cached_dataframe.pkl")
        logging.info("Loaded cached dataframe!")
    else:
        pwd = os.getcwd()
        df = pd.DataFrame(columns=["material", "formula", "structure", "Li Intercalation Energy @ 0.25 Li/M", "Volume Change @ 0.25 Li/M", "Charge on Li @ 0.25 Li/M", f"Average Li-M Distance @ 0.25 Li/M", f"Average Li-O Distance @ 0.25 Li/M", f"COHP of Li-M @ 0.25 Li/M", "Li Intercalation Energy @ 0.50 Li/M", "Volume Change @ 0.50 Li/M", "Charge on Li @ 0.50 Li/M", f"Average Li-M Distance @ 0.50 Li/M", f"Average Li-O Distance @ 0.50 Li/M", f"COHP of Li-M @ 0.50 Li/M", "Maximum Void Radius", "p Band Center", "d Band Center", "Band Gap"])
        fhandle = open(f"data.txt","w")
        for material in materials:
            system = get_data(material, addnl_folder_paths=["/expanse/lustre/scratch/sdutta3/temp_project/DMREF/bulk_calculation-forML/matminer"])
            next_index = len(df)
            df.loc[next_index] = system.compile_data(fhandle)
        os.chdir(pwd)

        df = StrToComposition().featurize_dataframe(df, "formula")
    
        ep_feat = ElementProperty.from_preset(preset_name="magpie")
        df = ep_feat.featurize_dataframe(df, col_id="composition")
        
        df_feat = DensityFeatures()
        df = df_feat.featurize_dataframe(df, col_id="structure")

        mpe_feat = MaximumPackingEfficiency()
        df = mpe_feat.featurize_dataframe(df, col_id="structure")
        
        df.to_pickle("cached_df.pkl")
        f = open("code_hash.txt", "w")
        f.write(compute_file_hash(__file__))