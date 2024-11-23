import os
import re
import numpy as np
import pandas as pd
import hashlib
from pipelines import DOS
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
from matminer.featurizers.conversions import ASEAtomstoStructure, StrToComposition
from matminer.featurizers.structure import DensityFeatures, MaximumPackingEfficiency
from matminer.featurizers.composition import ElementProperty
from pymatgen.analysis.local_env import VoronoiNN

class get_data:
    def __init__(self, material):
        self.material = material
        self.elements = re.findall(r'([A-Z][a-z]?)\d*', self.material)
        os.chdir(material)
        self.formula = self.get_formula()
        self.structure = self.get_PMG_structure()
        os.chdir("../")
    
    def get_metals(self):
        metals = [metal for metal in self.elements if metal not in ["O", "S", "Li"]]
        return metals
    
    def get_bridging_element(self):
        elements = self.elements
        return elements[-1]
    
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
    
    def get_dos_info(self):
        os.chdir("Electronic_calculation")
        dos = DOS()
        band_centers = [dos.get_band_centers()[1], dos.get_band_centers()[2]]
        os.chdir("../")
        return band_centers
    
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
        if "Li" in self.elements:
            mult=1.5
        else:
            mult=1
        nat_cut = natural_cutoffs(atoms, mult=mult)
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
                if atoms[i].symbol==self.get_bridging_element():
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
        fhandle.write(str_format.format("Site", "Li Intercalation Energy (eV)", "Charge on Li", f"Average Li-{self.get_metals()} Distance", f"Average Li-{self.get_bridging_element()} Distance", f"COHP of Li-{self.get_metals()}"))
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
                    n_Li = len([atom for atom in atoms_with_Li if atom.symbol=="Li"])
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
        print(formula)
        fhandle.write(f"{material}\n")
        os.chdir(material)
        structure = self.structure
        max_void_radius = self.get_max_void_radius()
        intercalation_data = self.get_intercalation_data(fhandle)
        os.chdir("../")
        return [material, formula, structure] + intercalation_data + [max_void_radius]

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
    materials = ["LiMn2O4", "Mo3Nb2O14", "MoO2", "TiO2", "VO2", "W3Nb2O14", "LiMn1.5Ni0.5O4"]
    if os.path.exists("cached_dataframe.pkl") and not changes_in_code("code_hash.txt"):
        df = pd.read_pickle("cached_dataframe.pkl")
        print("Loaded cached dataframe!")
    else:
        df = pd.DataFrame(columns=["material", "formula", "structure", "Li Intercalation Energy @ 0.25 Li/M", "Volume Change @ 0.25 Li/M", "Charge on Li @ 0.25 Li/M", f"Average Li-M Distance @ 0.25 Li/M", f"Average Li-O Distance @ 0.25 Li/M", f"COHP of Li-M @ 0.25 Li/M", "Li Intercalation Energy @ 0.50 Li/M", "Volume Change @ 0.50 Li/M", "Charge on Li @ 0.50 Li/M", f"Average Li-M Distance @ 0.50 Li/M", f"Average Li-O Distance @ 0.50 Li/M", f"COHP of Li-M @ 0.50 Li/M", "Maximum Void Radius"])
        fhandle = open(f"data.txt","w")
        for material in materials:
            system = get_data(material)
            next_index = len(df)
            df.loc[next_index] = system.compile_data(fhandle)

        df = StrToComposition().featurize_dataframe(df, "formula")
    
        ep_feat = ElementProperty.from_preset(preset_name="magpie")
        df = ep_feat.featurize_dataframe(df, col_id="composition")
        
        df_feat = DensityFeatures()
        df = df_feat.featurize_dataframe(df, col_id="structure")

        mpe_feat = MaximumPackingEfficiency()
        df = mpe_feat.featurize_dataframe(df, col_id="structure")
        
        df.to_pickle("cached_dataframe.pkl")
        f = open("code_hash.txt", "w")
        f.write(compute_file_hash(__file__))

    pd.set_option("display.max_columns", 200)
    print(df)