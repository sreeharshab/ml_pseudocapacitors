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

"""
Li: Lithium
M: Metals
B: Bridging elements
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
root = os.getcwd()

class get_data:
    def __init__(self, material, fhandle, addnl_folder_paths=None):
        self.material = material
        if addnl_folder_paths is None:
            os.chdir(material)
        elif addnl_folder_paths is not None:
            addnl_folder_paths = [f"{root}"] + addnl_folder_paths
            addnl_folder_paths = [path+"/" for path in addnl_folder_paths]
            for path in addnl_folder_paths:
                try:
                    os.chdir(f"{path}"+f"{material}")
                    break
                except FileNotFoundError:
                    pass
        self.atoms, self.energy = self.get_atoms_and_energy()
        self.formula = str(self.atoms.symbols)
        logging.info(f"Material: {material}, Formula: {self.formula}")
        self.elements = re.findall(r'([A-Z][a-z]?)\d*', self.formula)
        self.metals = self.get_metals()
        self.bridging_elements = self.get_bridging_elements()
        self.structure = self.get_PMG_structure()
        self.max_void_radius = self.get_max_void_radius()
        self.intercalation_data = self.get_intercalation_data(fhandle)
        self.dos_data = self.get_dos_data()
        self.data = [material, self.formula, self.structure] + self.intercalation_data + [self.max_void_radius] + self.dos_data
        os.chdir("../")
    
    def get_metals(self):
        metals = [metal for metal in self.elements if metal not in ["O", "S", "C", "N", "Si", "P", "Li"]]
        return metals
    
    def get_bridging_elements(self):
        bridging_elements = [element for element in self.elements if element in ["O", "S", "C", "N", "Si", "P"]]
        return bridging_elements
    
    def get_atoms_and_energy(self, dir_name="Energy_calculation"):
        os.chdir(dir_name)
        atoms = read("OUTCAR@-1")
        energy = atoms.get_potential_energy()
        os.chdir("../")
        return atoms, energy
    
    def get_PMG_structure(self):
        data = {
            "ase atoms": [self.atoms]
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
            try:
                dos.parse_doscar()
                band_gap = dos.get_band_gap()
                p_dos_up, p_dos_down = dos.get_orbital_projected_dos("p")
                d_dos_up, d_dos_down = dos.get_orbital_projected_dos("d")
                p_band_center = dos.get_band_center(p_dos_up, p_dos_down)    
                d_band_center = dos.get_band_center(d_dos_up, d_dos_down)
                indices = [atom.index for atom in self.atoms if atom.symbol in self.metals]
                temp_dos_up, temp_dos_down = dos.get_select_atoms_orbital_projected_dos(indices,"p")
                metal_cond_p_band_center = dos.get_band_center(dos_up=temp_dos_up, dos_down=temp_dos_down, energy_range=[0,dos.energies_wrt_fermi[-1]])
                indices = [atom.index for atom in self.atoms if atom.symbol in self.bridging_elements]
                temp_dos_up, temp_dos_down = dos.get_select_atoms_orbital_projected_dos(indices,"p")
                brid_val_p_band_center = dos.get_band_center(dos_up=temp_dos_up, dos_down=temp_dos_down, energy_range=[dos.energies_wrt_fermi[0],0])
                os.chdir("../")
                return [band_gap, p_band_center, d_band_center, metal_cond_p_band_center, brid_val_p_band_center]
            except AssertionError:
                logging.warning(f"Electronic calculation is not completed for {self.material}. Taking band gap and band centers as 0...")
                os.chdir("../")
                return [0,0,0,0,0]
        except FileNotFoundError:
            logging.warning(f"Electronic calculation does not exist for {self.material}. Taking band gap and band centers as 0...")
            return [0,0,0,0,0]
    
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
    
    def get_Li_M_B_distances(self, atoms):
        kwargs = {"Mn":2, "Co":2, "Fe":2.5, "Nb":2, "C":1.7, "N":1.7}   # Custom cutoffs
        nat_cut = natural_cutoffs(atoms, **kwargs)
        nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
        nl.update(atoms)
        Li_M_distances = []
        Li_B_distances = []
        M_B_distances = []
        Li_indices = [atom.index for atom in atoms if atom.symbol=="Li"]
        for Li_index in Li_indices:
            indices, offsets = nl.get_neighbors(Li_index)
            distances = np.linalg.norm(atoms.positions[indices] + offsets @ atoms.cell - atoms.positions[Li_index], axis=1)
            Li_M_distances.append(distances[[atoms[i].symbol in self.metals for i in indices]].mean())
            Li_B_distances.append(distances[[atoms[i].symbol in self.bridging_elements for i in indices]].mean())
        M_indices = [atom.index for atom in atoms if atom.symbol in self.metals]
        for M_index in M_indices:
            indices, offsets = nl.get_neighbors(M_index)
            distances = np.linalg.norm(atoms.positions[indices] + offsets @ atoms.cell - atoms.positions[M_index], axis=1)
            M_B_distances.append(distances[[atoms[i].symbol in self.bridging_elements for i in indices]].mean())
        return [round(np.mean(Li_M_distances, axis=0),3), round(np.mean(Li_B_distances, axis=0),3), round(np.mean(M_B_distances, axis=0),3)]
    
    def get_Li_M_B_charges(self):
        try:
            os.chdir("bader")
            try:
                atoms = read("with_charges.traj")
            except FileNotFoundError:
                os.chdir("../")
                return [0,0,0]
            charges = atoms.get_initial_charges()
            Li_charges = charges[[atom.symbol=="Li" for atom in atoms]]
            Li_charge = round(np.mean(Li_charges),3)
            M_charges = charges[[atom.symbol in self.metals for atom in atoms]]
            M_charge = round(np.mean(M_charges),3)
            B_charges = charges[[atom.symbol in self.bridging_elements for atom in atoms]]
            B_charge = round(np.mean(B_charges),3)
            os.chdir("../")
            return [Li_charge, M_charge, B_charge]
        except FileNotFoundError:
            return [0,0,0]
    
    def get_intercalation_data(self, fhandle):
        atoms = self.atoms
        energy = self.energy
        volume = atoms.get_volume()
        n_M = len([atom.index for atom in atoms if atom.symbol in self.metals])
        fhandle.write(f"Material: {self.material}, Formula: {self.formula}:\n")
        os.chdir("Intercalation")
        oswalk = [i for i in os.walk(".")]
        nLifolders = sorted(oswalk[0][1])
        str_format = "{:^15} {:^28} {:^15} {:^15} {:^15} {:^23} {:^23} {:^23}\n"
        fhandle.write(str_format.format("Site", "Li Intercalation Energy (eV)", "Charge on Li", "Charge on M", "Charge on B", "Average Li-M Distance", "Average Li-B Distance", "Average M-B Distance"))
        for nLifolder in nLifolders:
            match = re.match(r"(\d+)_Li", nLifolder)
            n_Li = int(match.group(1))
            if n_Li/n_M == 0.25 or n_Li/n_M == 0.5:
                fhandle.write(f"\tNumber of Li: {n_Li}\n")
                os.chdir(nLifolder)
                oswalk = [i for i in os.walk(".")]
                sites = sorted(oswalk[0][1])
                Li_energies, volume_changes, Li_charges, M_charges, B_charges, Li_M_distances, Li_B_distances, M_B_distances = [], [], [], [], [], [], [], []
                for site in sites:
                    os.chdir(f"{site}")
                    atoms_with_Li, energy_with_Li = self.get_atoms_and_energy("geo_opt")
                    volume_with_Li = atoms_with_Li.get_volume()
                    mu_Li = -33.22057791/16 # From DFT calculation.
                    Li_energy = round((energy_with_Li-energy-n_Li*mu_Li)/(n_Li),3)
                    volume_change = volume_with_Li - volume
                    Li_M_B_distances = self.get_Li_M_B_distances(atoms_with_Li)
                    Li_M_B_charges = self.get_Li_M_B_charges()
                    fhandle.write(str_format.format(site, Li_energy, Li_M_B_charges[0], Li_M_B_charges[1], Li_M_B_charges[2], Li_M_B_distances[0], Li_M_B_distances[1], Li_M_B_distances[2]))
                    lists = [Li_energies, volume_changes, Li_charges, M_charges, B_charges, Li_M_distances, Li_B_distances, M_B_distances]
                    values = [Li_energy, volume_change]+Li_M_B_charges+Li_M_B_distances
                    for lst, val in zip(lists, values):
                        lst.append(val)
                    os.chdir("../")
                mLei = Li_energies.index(min(Li_energies))  # mLei: minimum Li energy index
                if Li_charges[mLei]==0:
                    logging.warning(f"Li charge not available at {nLifolder}. Taking it as 0...")
                if n_Li/n_M == 0.25:
                    data_25 = [Li_energies[mLei], volume_changes[mLei], Li_charges[mLei], M_charges[mLei], B_charges[mLei], Li_M_distances[mLei], Li_B_distances[mLei], M_B_distances[mLei]]
                elif n_Li/n_M == 0.5:
                    data_50 = [Li_energies[mLei], volume_changes[mLei], Li_charges[mLei], M_charges[mLei], B_charges[mLei], Li_M_distances[mLei], Li_B_distances[mLei], M_B_distances[mLei]]
                os.chdir("../")
        os.chdir("../")
        fhandle.write("\n")
        return data_25+data_50

if __name__=="__main__":
    materials = ['Li3VO4', 'LiMn1.5Ni0.5O4', 'LiMn2O4', 'Mo3Nb2O14', 'MoO2', 'TiO2-B', 'TiO2-R', 'VO2-B', 'W3Nb2O14', 'Fe2CN6', 'LiCoO2', 'LiCoPO4', 'LiFeMgPO4', 'LiFePO4', 'LiFeSO4F', 'LiNiO2', 'LiVPO5', 'MoS2', 'NCM333', 'NCM811', 'Nb2O5-R', 'Nb2O5-H', 'Nb2O5-T', 'Nb2O5-TT', 'NbS2', 'TiO2-anatase', 'TiS2', 'V2O5', 'V6O13', 'VO2-M', 'VO2-R', 'Li2FeSiO4', 'Li3+xV2O5']
    df = pd.DataFrame(columns=["material", "formula", "structure", "Li Intercalation Energy @ 0.25 Li/M", "Volume Change @ 0.25 Li/M", "Charge on Li @ 0.25 Li/M", "Charge on M @ 0.25 Li/M", "Charge on B @ 0.25 Li/M", f"Average Li-M Distance @ 0.25 Li/M", f"Average Li-B Distance @ 0.25 Li/M", f"Average M-B Distance @ 0.25 Li/M", "Li Intercalation Energy @ 0.50 Li/M", "Volume Change @ 0.50 Li/M", "Charge on Li @ 0.50 Li/M", "Charge on M @ 0.50 Li/M", "Charge on B @ 0.50 Li/M", f"Average Li-M Distance @ 0.50 Li/M", f"Average Li-B Distance @ 0.50 Li/M", f"Average M-B Distance @ 0.50 Li/M", "Maximum Void Radius", "Band Gap", "p Band Center", "d Band Center", "M Conduction p Band Center", "B Valence p Band Center"])
    fhandle = open(f"intercalation_data.txt","w")
    for material in materials:
        system = get_data(material, fhandle, addnl_folder_paths=["/expanse/lustre/scratch/sdutta3/temp_project/DMREF/bulk_calculation-forML/matminer"])
        next_index = len(df)
        df.loc[next_index] = system.data
    os.chdir(root)

    df = StrToComposition().featurize_dataframe(df, "formula")

    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df = ep_feat.featurize_dataframe(df, col_id="composition")
    
    df_feat = DensityFeatures()
    df = df_feat.featurize_dataframe(df, col_id="structure")

    mpe_feat = MaximumPackingEfficiency()
    df = mpe_feat.featurize_dataframe(df, col_id="structure")
    
    df.to_pickle("cached_df.pkl")