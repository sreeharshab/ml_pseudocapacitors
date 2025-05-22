import os
import re
import numpy as np
import pandas as pd
import logging
from pipelines import DOS
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.structure import DensityFeatures, MaximumPackingEfficiency
from matminer.featurizers.composition import ElementProperty
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.ase import AseAtomsAdaptor

"""
Li: Lithium
M: Metals
B: Bridging elements
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
root = os.getcwd()

class get_features:
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
        self.structure = AseAtomsAdaptor.get_structure(self.atoms)
        self.formula = str(self.atoms.symbols)
        logging.info(f"Material: {material}, Formula: {self.formula}")
        self.elements = re.findall(r'([A-Z][a-z]?)\d*', self.formula)
        self.metals = [metal for metal in self.elements if metal not in ["O", "S", "C", "N", "Si", "P", "F", "Li"]]
        self.bridging_elements = [element for element in self.elements if element in ["O", "S", "C", "N", "Si", "P", "F"]]
        self.lattice_parameters = list(self.atoms.cell.cellpar()[0:3]/self.atoms.get_volume())
        self.max_void_radius = self.get_max_void_radius()
        self.distances = self.get_Li_M_B_distances(self.atoms)
        self.charges = self.get_Li_M_B_charges()
        if self.charges==[0,0,0]:
            logging.warning(f"Bader_calculation does not exist/is not completed at {os.getcwd()}. Taking charges as 0...")
        self.dos_data = self.get_dos_data()
        if self.dos_data==[0]*19:
            logging.warning(f"Electronic_calculation does not exist/is not completed at {os.getcwd()}. Taking band gap and band centers as 0...")
        self.intercalation_data = self.get_intercalation_data(fhandle)
        self.data = [material, self.formula, self.structure] + self.lattice_parameters + [self.max_void_radius] + self.distances + self.charges + self.dos_data + self.intercalation_data
        os.chdir("../")
    
    def get_atoms_and_energy(self, dir_name="Energy_calculation"):
        if not os.path.exists(dir_name):
            raise FileNotFoundError(f"{dir_name} does not exist at {os.getcwd()}.")
        os.chdir(dir_name)
        try:
            atoms = read("OUTCAR@-1")
            energy = atoms.get_potential_energy()
        except:
            raise IOError(f"Failed to read OUTCAR from {dir_name} at {os.getcwd()}.")
        os.chdir("../")
        return atoms, energy
    
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
        Li_M_distances, Li_B_distances, M_B_distances = [], [], []
        Li_indices = [atom.index for atom in atoms if atom.symbol=="Li"]
        if len(Li_indices)!=0:
            for Li_index in Li_indices:
                indices, offsets = nl.get_neighbors(Li_index)
                distances = np.linalg.norm(atoms.positions[indices] + offsets @ atoms.cell - atoms.positions[Li_index], axis=1)
                Li_M_distances.append(distances[[atoms[i].symbol in self.metals for i in indices]].mean())
                Li_B_distances.append(distances[[atoms[i].symbol in self.bridging_elements for i in indices]].mean())
        else:            
            Li_M_distances, Li_B_distances = [0], [0]
        M_indices = [atom.index for atom in atoms if atom.symbol in self.metals]
        for M_index in M_indices:
            indices, offsets = nl.get_neighbors(M_index)
            distances = np.linalg.norm(atoms.positions[indices] + offsets @ atoms.cell - atoms.positions[M_index], axis=1)
            M_B_distances.append(distances[[atoms[i].symbol in self.bridging_elements for i in indices]].mean())
        return [round(np.mean(Li_M_distances, axis=0),3), round(np.mean(Li_B_distances, axis=0),3), round(np.mean(M_B_distances, axis=0),3)]
    
    def get_Li_M_B_charges(self, dir_name="Bader_calculation"):
        try:
            os.chdir(dir_name)
            try:
                atoms = read("with_charges.traj")
            except FileNotFoundError:
                os.chdir("../")
                return [0,0,0]
            charges = atoms.get_initial_charges()
            Li_charges = charges[[atom.symbol=="Li" for atom in atoms]]
            if len(Li_charges)!=0:
                Li_charge = round(np.mean(Li_charges),3)
            else:
                Li_charge = 0
            M_charges = charges[[atom.symbol in self.metals for atom in atoms]]
            M_charge = round(np.mean(M_charges),3)
            B_charges = charges[[atom.symbol in self.bridging_elements for atom in atoms]]
            B_charge = round(np.mean(B_charges),3)
            os.chdir("../")
            return [Li_charge, M_charge, B_charge]
        except FileNotFoundError:
            return [0,0,0]
    
    def get_dos_data(self, dir_name="Electronic_calculation"):
        def get_band_centers(dos_up, dos_down):
            band_center = dos.get_band_center(dos_up, dos_down)
            val_band_center = dos.get_band_center(dos_up=dos_up, dos_down=dos_down, energy_range=[dos.energies_wrt_fermi[0],0])
            cond_band_center = dos.get_band_center(dos_up=dos_up, dos_down=dos_down, energy_range=[0,dos.energies_wrt_fermi[-1]])
            return [band_center, val_band_center, cond_band_center]
        try:
            os.chdir(dir_name)
            dos = DOS()
            try:
                dos.parse_doscar()
                band_gap = dos.get_band_gap()
                temp_dos_up, temp_dos_down = dos.get_total_dos()
                band_centers = get_band_centers(temp_dos_up, temp_dos_down)
                temp_dos_up, temp_dos_down = dos.get_orbital_projected_dos("p")
                p_band_centers = get_band_centers(temp_dos_up, temp_dos_down)
                temp_dos_up, temp_dos_down = dos.get_orbital_projected_dos("d")
                d_band_centers = get_band_centers(temp_dos_up, temp_dos_down)
                indices = [atom.index for atom in self.atoms if atom.symbol in self.metals]
                temp_dos_up, temp_dos_down = dos.get_select_atoms_orbital_projected_dos(indices,"p")
                metal_p_band_centers = get_band_centers(temp_dos_up, temp_dos_down)
                temp_dos_up, temp_dos_down = dos.get_select_atoms_orbital_projected_dos(indices,"d")
                metal_d_band_centers = get_band_centers(temp_dos_up, temp_dos_down)
                indices = [atom.index for atom in self.atoms if atom.symbol in self.bridging_elements]
                temp_dos_up, temp_dos_down = dos.get_select_atoms_orbital_projected_dos(indices,"p")
                brid_p_band_centers = get_band_centers(temp_dos_up, temp_dos_down)
                os.chdir("../")
                return [band_gap] + band_centers + p_band_centers + d_band_centers + metal_p_band_centers + metal_d_band_centers + brid_p_band_centers
            except AssertionError:
                os.chdir("../")
                return [0]*19
        except FileNotFoundError:
            return [0]*19
    
    def get_intercalation_data(self, fhandle):
        atoms = self.atoms
        energy = self.energy
        volume = atoms.get_volume()
        n_M = sum(1 for atom in atoms if atom.symbol in self.metals)
        if self.material=="Li7NbS2":
            n_M = 8
        fhandle.write(f"Material: {self.material}, Formula: {self.formula}:\n")
        str_format = "{:^15} {:^28} {:^15} {:^15} {:^15} {:^23} {:^23} {:^23}\n"
        fhandle.write(str_format.format("Site", "Li Intercalation Energy (eV)", "Average Li-M Distance", "Average Li-B Distance", "Average M-B Distance", "Charge on Li", "Charge on M", "Charge on B"))
        data = {}
        nLifolders = [entry.name for entry in os.scandir("Intercalation") if entry.is_dir()]
        os.chdir("Intercalation")
        for nLifolder in nLifolders:
            match = re.match(r"(\d+)_Li", nLifolder)
            if match:
                n_Li = int(match.group(1))
            else:
                n_Li = 0
            if not (0.25<=n_Li/n_M<=0.3 or n_Li/n_M==0.5):
                continue
            fhandle.write(f"\tNumber of Li: {n_Li}\n")
            os.chdir(nLifolder)
            oswalk = [i for i in os.walk(".")]
            sites = sorted(oswalk[0][1])
            Li_energies, volume_changes, Li_M_distances, Li_B_distances, M_B_distances, Li_charges, M_charges, B_charges, B_val_band_centers, B_cond_band_centers = [], [], [], [], [], [], [], [], [], []
            for site in sites:
                os.chdir(f"{site}")
                try:
                    atoms_with_Li, energy_with_Li = self.get_atoms_and_energy("geo_opt")
                except FileNotFoundError:
                    continue
                volume_with_Li = atoms_with_Li.get_volume()
                mu_Li = -33.22057791/16 # From DFT calculation.
                Li_energy = round((energy_with_Li-energy-n_Li*mu_Li)/(n_Li),3)
                volume_change = (volume_with_Li - volume)/volume
                Li_M_B_distances = self.get_Li_M_B_distances(atoms_with_Li)
                try:
                    Li_M_B_charges = self.get_Li_M_B_charges("bader")
                except FileNotFoundError:
                    Li_M_B_charges = [0,0,0]
                B_val_cond_band_centers = self.get_dos_data(dir_name="dos")[17:19]
                fhandle.write(str_format.format(site, Li_energy, Li_M_B_distances[0], Li_M_B_distances[1], Li_M_B_distances[2], Li_M_B_charges[0], Li_M_B_charges[1], Li_M_B_charges[2]))
                lists = [Li_energies, volume_changes, Li_M_distances, Li_B_distances, M_B_distances, Li_charges, M_charges, B_charges, B_val_band_centers, B_cond_band_centers]
                values = [Li_energy, volume_change]+Li_M_B_distances+Li_M_B_charges+B_val_cond_band_centers
                for lst, val in zip(lists, values):
                    lst.append(val)
                os.chdir("../")
            mLei = Li_energies.index(min(Li_energies))  # mLei: minimum Li energy index
            if (Li_charges[mLei]==0 and M_charges[mLei]==0 and B_charges[mLei]==0):
                logging.warning(f"bader does not exist/is not completed at {os.getcwd()}/{sites[mLei]}. Taking charges as 0...")
            if (B_val_band_centers[mLei]==0 and B_cond_band_centers[mLei]==0):
                logging.warning(f"dos does not exist/is not completed at {os.getcwd()}/{sites[mLei]}. Taking band centers as 0...")
            data[round(n_Li/n_M,2)] = [Li_energies[mLei], volume_changes[mLei], Li_M_distances[mLei], Li_B_distances[mLei], M_B_distances[mLei], Li_charges[mLei], M_charges[mLei], B_charges[mLei], B_val_band_centers[mLei], B_cond_band_centers[mLei]]
            if 0.25<=n_Li/n_M<=0.3:
                data[0.25] = data[round(n_Li/n_M,2)]
            os.chdir("../")
        os.chdir("../")
        fhandle.write("\n")
        try:
            data[0.25]
        except KeyError:
            logging.warning(f"Intercalation data does not exist for 0.25 Li/M at {os.getcwd()}. Taking values as 0...")
            data[0.25] = [0]*10
        try:
            data[0.5]
        except KeyError:
            logging.warning(f"Intercalation data does not exist for 0.5 Li/M at {os.getcwd()}. Taking values as 0...")
            data[0.5] = [0]*10
        return data[0.25]+data[0.5]

def get_df(materials):
    df = pd.DataFrame(columns=["material", "formula", "structure", "Lattice Parameter a", "Lattice Parameter b", "Lattice Parameter c", "Maximum Void Radius", "Average Li-M Distance", "Average Li-B Distance", "Average M-B Distance", "Charge on Li", "Charge on M", "Charge on B", "Band Gap", "Band Center", "Valence Band Center", "Conduction Band Center", "p Band Center", "Valence p Band Center", "Conduction p Band Center", "d Band Center", "Valence d Band Center", "Conduction d Band Center", "M p Band Center", "M Valence p Band Center", "M Conduction p Band Center", "M d Band Center", "M Valence d Band Center", "M Conduction d Band Center", "B p Band Center", "B Valence p Band Center", "B Conduction p Band Center", "Li Intercalation Energy @ 0.25 Li/M", "Volume Change @ 0.25 Li/M", "Average Li-M Distance @ 0.25 Li/M", "Average Li-B Distance @ 0.25 Li/M", "Average M-B Distance @ 0.25 Li/M", "Charge on Li @ 0.25 Li/M", "Charge on M @ 0.25 Li/M", "Charge on B @ 0.25 Li/M", "B Valence p Band Center @ 0.25 Li/M", "B Conduction p Band Center @ 0.25 Li/M", "Li Intercalation Energy @ 0.50 Li/M", "Volume Change @ 0.50 Li/M", "Average Li-M Distance @ 0.50 Li/M", "Average Li-B Distance @ 0.50 Li/M", "Average M-B Distance @ 0.50 Li/M", "Charge on Li @ 0.50 Li/M", "Charge on M @ 0.50 Li/M", "Charge on B @ 0.50 Li/M", "B Valence p Band Center @ 0.50 Li/M", "B Conduction p Band Center @ 0.50 Li/M"])
    fhandle = open(f"intercalation_data.txt","w")
    for material in materials:
        features = get_features(material, fhandle)
        next_index = len(df)
        df.loc[next_index] = features.data
    os.chdir(root)
    df = StrToComposition().featurize_dataframe(df, "formula")
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df = ep_feat.featurize_dataframe(df, col_id="composition") 
    df_feat = DensityFeatures()
    df = df_feat.featurize_dataframe(df, col_id="structure")
    mpe_feat = MaximumPackingEfficiency()
    df = mpe_feat.featurize_dataframe(df, col_id="structure") 
    df.to_pickle("cached_df.pkl")
    return df