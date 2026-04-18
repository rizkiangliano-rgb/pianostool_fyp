"""importing relevant packages"""
import functools
from itertools import combinations
import pandas as pd
import numpy as np
import networkx as nx
import os
import re
from ase.io import read
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from collections import defaultdict, deque
from cosymlib.shape import Shape

"""set up variables and relevant data"""
tmQMg_data = pd.read_csv(r"C:\Users\Taufik\Downloads\tmQMg_properties_and_targets.csv")
tmQMg_xyz_folder = r"C:\Users\Taufik\Downloads\tmQMg_xyz\xyz"
tmQMg_gml_folder = r"C:\Users\Taufik\Downloads\u-NatQ_graphs\u-NatQ_graphs"
tmQMg_gml_2_folder = r"C:\Users\Taufik\Downloads\d-NatQ_graphs\d-NatQ_graphs"
results = [] #final classification dataframe
data_summary = [] #summary of amount of complexes based on os, charge, ring, metal
results_csd = [] #to be used for the main csv
results_further_info = [] #geomteric data of the complexes
results_aligned_array = [] #rotated complex position
results_csd_Os = []
results_Os = []
results_further_info_Os = []
results_aligned_array_Os = []
results_csd_Re = []
results_Re = []
results_further_info_Re = []
results_aligned_array_Re = []
data_summary_Os = []
data_summary_Re = []
target_metals = {"Ru", "Rh", "Os", "Ir", "Re"} #osmium will not be in the training data, for leave-one-metal-out purposes
ligands = {"C","H","B","N","O","F","Si","P","S","Cl","As","Se","Br","I"} #complex must not have atoms outside this set
charges = {0,1}
ring_size = {5,6}
oxidation = {0,1,2,3,4,5}
radii = {"H":(0.31,0.05), "B":(0.84,0.03), "C":(0.76,0.01), "N":(0.71,0.01), "O":(0.66,0.02), "F":(0.57,0.03), "Si":(1.11,0.02), "P":(1.07,0.03), "S":(1.05,0.03), "Cl":(1.02,0.04),
         "As":(1.19,0.04), "Se":(1.20,0.04), "Br":(1.20,0.03), "Ru":(1.46,0.07), "Rh":(1.42,0.07), "I":(1.39,0.03), "Os":(1.44,0.04), "Ir":(1.41,0.06), "Re":(1.51,0.07)} #radii and their s.d. from Cordero et al.
metal_electronegativity = {"Ru":1.54, "Os":1.65, "Rh":1.56, "Ir":1.68, "Re":1.60} #allen scale

"""initial filtering of metal centers,charge,and missing data; if any"""
tmQMg_Met = tmQMg_data[tmQMg_data["metal_center"].isin(target_metals)]
tmQMg_Clean = tmQMg_Met.dropna()
tmQMg_Clean = tmQMg_Clean[tmQMg_Clean["charge"] != -1]

"""functions"""
#defining the bond threshold
def bond_distances(sym,k=3.0): #identify maximum bond distance for rough guess, based on normal distribution assumption
    max_dist = {}
    max_dist_raw = {}
    for a,sym1 in enumerate(sym):
        for b,sym2 in enumerate(sym):
            if b < a:
                continue
            radius1, esd1 = radii[sym1]
            radius2, esd2 = radii[sym2]
            sum_radii = radius1 + radius2
            stddev = np.sqrt(esd1**2 + esd2**2)
            threshold = sum_radii + (k*stddev) + 0.225 #0.225 is an arbitrary buffer based on half of the buffer used in another study
            raw_threshold = sum_radii
            max_dist[(sym1,sym2)] = float(threshold)
            max_dist[(sym2,sym1)] = float(threshold)
            max_dist_raw[(sym1,sym2)] = float(raw_threshold)
            max_dist_raw[(sym2,sym1)] = float(raw_threshold)
    return max_dist, max_dist_raw

#find statistical summary of multiple values
def statistics(lengths):
    stddev = np.std(lengths)
    maximum = np.max(lengths)
    minimum = np.min(lengths)
    mean = np.mean(lengths)
    return stddev, maximum, minimum, mean

#smiles filter and identification
def ox_from_smiles(smiles_here,metal_center): #identify oxidation state based on existing smiles string
    pattern_1 = rf"\[{metal_center}([+-]?\d*)\]"
    match_a = re.search(pattern_1,smiles_here)
    if not match_a:
        return 0
    charge_str = match_a.group(1)
    if charge_str == "":
        return 0
    if charge_str == "+":
        return 1
    if charge_str == "-":
        return -1
    return int(charge_str)
def carboranes_from_smiles(smiles_borane): #removal of borane/carborane bonding
    borane = re.escape("[BH-]")
    count_b = len(re.findall(borane,smiles_borane))
    return count_b

#planarity functions
def verify_planar(graph,index): #verify if ring is all sp2
    is_planar = True
    for idx in index:
        total_degree = graph.degree(idx)
        external_degree = total_degree - 2
        if external_degree != 1:
           is_planar = False
    return is_planar
def best_fit_plane(coordinates): #find the rmsd of a ring
    if len(coordinates) < 3:
        return 1.5, [], [], []
    coord = np.array(coordinates)
    centroid_ring = np.mean(coord, axis=0)
    centered_coord = coord - centroid_ring
    u, st, vt = np.linalg.svd(centered_coord)
    normal_vector = vt[2]
    spread = vt[0]
    distance_to_plane = np.dot(centered_coord,normal_vector)
    rmsd = np.sqrt(np.mean(distance_to_plane**2))
    return rmsd, centroid_ring, normal_vector, spread

#establishing connectivity of atoms to roughly find the right coordination sphere
def components(indices_candidate,xyz,maxim,setter): #identify connected atoms from the coordination sphere within cutoff
    #indices: atom indices to be connected, maxim: bond threshold, setter: set of possible values
    pos_all = xyz.positions[indices_candidate]
    adjacency_here = None
    for a, b in combinations(indices_candidate,2):
        sym_a = xyz[a].symbol
        sym_b = xyz[b].symbol
        value = maxim[(sym_a,sym_b)]
        atoms_connect = pdist(pos_all)
        dist_matrix = squareform(atoms_connect)
        adj_here = (dist_matrix < value) & (dist_matrix != 0)
        adjacency_here = adj_here if adjacency_here is None else adjacency_here + adj_here
    #cluster the adjacent atoms
    num_constituents, labels = connected_components(adjacency_here, directed=False)
    constituents_here = defaultdict(list)
    for x, label in enumerate(labels):
        constituents_here[label].append(x)
    connectivity = [constituent for constituent in constituents_here.values() if len(constituent) in setter]
    connectivity = [item for sublist in connectivity for item in sublist]
    return connectivity

#establishing the connectivity of atoms beyond the coordination sphere
def ring_connection(molecule,center_index,global_positions,bond,sub_index,seat_indices,leg_index,unsubstituted_index):
    #establishing the whole graph with bond threshold cutoff to infer connectivity
    tree = KDTree(global_positions)
    candidates = tree.query_pairs(r=3.5) #use an arbitrary and generous initial cutoff
    adj_list = []
    for j, k in candidates:
        sym_j, sym_k = molecule[j].symbol, molecule[k].symbol
        thresh = bond[(sym_j,sym_k)]
        distance = np.linalg.norm(global_positions[j] - global_positions[k])
        if distance < thresh:
            adj_list.append((j,k))
    g = nx.Graph()
    g.add_edges_from(adj_list)
    #remove metal and ring indices for analysis of substituent
    search_graph = g.copy()
    indices_total = seat_indices + [center_index]
    search_graph.remove_nodes_from(indices_total)
    #acquire the whole ligand subgraph
    ligand_only_graph = g.copy()
    ligand_only_graph.remove_node(center_index)
    connectivity = {}
    #find all atoms connected to the substituent
    for n in sub_index:
        all_neigh = list(g.neighbors(n))
        substituent_start = [k for k in all_neigh if k not in seat_indices and k != center_index] #ensure it is purely the substituent
        connected_atoms = []
        #bond moving from the root of the substituent
        for root in substituent_start:
            connected_atom = nx.node_connected_component(search_graph, root)
            connected_atoms.append(list(connected_atom))
        connectivity[n] = connected_atoms
    ring_class = {idx: {"class":set(),"neighbors": []} for idx in seat_indices}
    ring_class_simplified = {idx: [] for idx in seat_indices}
    #find fused aromatic rings complexes
    all_cycles = nx.cycle_basis(ligand_only_graph)
    for cycle in all_cycles:
        intersect = set(cycle) & set(seat_indices)
        if len(intersect) == 2:
            if verify_planar(ligand_only_graph, cycle):
                for index in intersect:
                    ring_class[index]["class"].add("fused_aromatic") #flags indenyl, fluorenyl, or naphthalene
            else:
                for index in intersect:
                   ring_class[index]["class"].add("bulky_substituent") #consider non-planar rings as bulky substituent
    #consider non-fused cases
    for carbon, connect in connectivity.items():
        for con in connect:
            ring_class[carbon]["neighbors"].extend(list(con)) #list of carbon neighbors
            ring_class_simplified[carbon].extend(list(con))
            #detect bridged cases, which arises when a leg ligand is part of the substituent
            bridge = [x for x in sub_index if any(item in leg_index for item in con)]
            if bridge:
                ring_class[carbon]["class"].add("bridged")
            #detect unusual rings that may potentially be tetra haptic
            for n in sub_index:
                neighbor = list(ligand_only_graph.neighbors(n))
                cpd = any(molecule[node].symbol == "O" and ligand_only_graph.degree(node) == 1 for node in con if node in neighbor)
                cpd_sulfur = any(molecule[node].symbol == "S" and ligand_only_graph.degree(node) == 1 for node in con if node in neighbor)
                cpd_imine = any(molecule[node].symbol == "N" and ligand_only_graph.degree(node) == 2 for node in con if node in neighbor)
                if cpd or cpd_sulfur or cpd_imine:
                   ring_class[carbon]["class"].add("heteroring")
                fulvene = any(molecule[node].symbol == "C" and ligand_only_graph.degree(node) in {2, 3} and np.linalg.norm(global_positions[node] - global_positions[n]) < 1.425 for node in con if node in neighbor)
                if fulvene:
                   if "fused_aromatic" not in ring_class[carbon]["class"]:
                       ring_class[carbon]["class"].add("fulvene")
            #is bulky or simple substituent
            if not ring_class[carbon]["class"]:
                for cycle in all_cycles: #consider "bulky" as containing a ring fragment or 10 atoms above
                    if (set(cycle).issubset(set(con))) or (len(con) >= 10):
                        ring_class[carbon]["class"].add("bulky_substituent")
            if not ring_class[carbon]["class"]:
                ring_class[carbon]["class"].add("simple_substituent")
            #remove redundancies
            if "bridged" in ring_class[carbon]["class"] and "bulky_substituent" in ring_class[carbon]["class"]:
                ring_class[carbon]["class"].discard("bulky_substituent")
    #unsubstituted
    for carbon in seat_indices:
        if not ring_class[carbon]["class"]:
            ring_class[carbon]["class"].add("unsubstituted")
    #add hydrogen indices
    for idx in unsubstituted_index:
        neighbors = list(ligand_only_graph.neighbors(idx))
        for neigh in neighbors:
            if neigh in seat_indices:
                ring_class[neigh]["neighbors"].append(idx)
                ring_class_simplified[neigh].append(idx)
    return ring_class, ring_class_simplified,ligand_only_graph
def denticity(graph_ligand,leg_index,ring_index): #verify the denticity of ancillaries
    graph_ligand.remove_nodes_from(ring_index)
    ligands_connect = list(nx.connected_components(graph_ligand))
    coordination_data = []
    for ligand in ligands_connect:
        atoms_check = [n for n in ligand if n in leg_index]
        if not atoms_check:
            continue
        connectivity = len(atoms_check) #denticity
        tether = [] #how many atoms tether
        all_tether = None #account for other shortest path or non-shortest path
        tether_atoms = [] #which atoms
        if connectivity > 1:
            for atom1, atom2 in combinations(atoms_check,2):
                path = nx.shortest_path(graph_ligand,source=atom1,target=atom2)
                all_tether = list(nx.all_simple_paths(graph_ligand,source=atom1,target=atom2))
                tether.append(len(path)-2)
                tether_atoms.append(list(set(path) - set(atoms_check)))
        cleaned = []
        if all_tether:
           all_tether = [[item for item in sublist if item not in atoms_check] for sublist in all_tether]
           sets = [set(l) for l in all_tether]
           for ix, st in enumerate(sets):
               if not any(st > other for j,other in enumerate(sets) if ix != j):
                   cleaned.append(all_tether[ix])
        coordination_data.append({"denticity":connectivity, "num_tether":tether, "atoms":atoms_check, "tether":tether_atoms, "all_tether":cleaned})
    #identify complete amount of atoms bonded to the ancillary
    connectivity = {}
    for n in leg_index:
        ancillary_neigh = list(graph_ligand.neighbors(n))
        ancillary_root = [k for k in ancillary_neigh if k not in leg_index]
        connected_atoms = []
        search_graph = graph_ligand.copy()
        search_graph.remove_nodes_from(leg_index)
        #bond moving from the root of the ancillary
        for root in ancillary_root:
            connected_atom = nx.node_connected_component(search_graph, root)
            connected_atoms.append(list(connected_atom))
        connectivity[n] = connected_atoms
    ancillary_connection = {idx: [] for idx in leg_index}
    for ancillary, connect in connectivity.items():
        for con in connect:
            ancillary_connection[ancillary].extend(list(con))
    return coordination_data, ancillary_connection
def extract_features(classification_a,classification_b, target_a, target_b): #defining the complex as a whole under a binary basis and count
    complex_features = {f"is_{tag}": 0 for tag in target_a}
    n_bridge = 0
    n_bulky = 0
    n_simple = 0
    for idx, data in classification_a.items():
        for tag in data["class"]:
            key = f"is_{tag}"
            if key in complex_features:
                complex_features[key] = 1
            if "bridged" in tag:
                n_bridge += 1
            if "bulky_substituent" in tag:
                n_bulky += 1
            if "simple_substituent" in tag:
                n_simple += 1
    for j, k in enumerate(target_b):
        if classification_b["denticity"] == j + 1:
            complex_features[k] = 1
        else:
            complex_features[k] = 0
    complex_features["num_bridge"] = n_bridge
    complex_features["num_bulk"] = n_bulky
    complex_features["num_simple"] = n_simple
    return complex_features

#recursive function to find the priority based on CIP, using bond order data from quantum graph
def cahn_ingold_prelog(ancillaries,graph,metal,max_depth=10,wbo_tol=0.2):
    state = {ancillary: {"current_shell":[ancillary], "visited": {metal, ancillary}, "score_sequence":[[(float(graph.nodes[ancillary].get("feature_atomic_number",0)),1.0)]]} for ancillary in ancillaries}
    for depth in range(max_depth):
        for ancillary in ancillaries:
            shell = state[ancillary]["current_shell"]
            visited = state[ancillary]["visited"]
            if not shell:
                state[ancillary]["score_sequence"].append([])
                continue
            next_shell = []
            signature = []
            for nodes in shell:
                neighbors = [node for node in graph.neighbors(nodes) if node not in visited]
                for ng in neighbors:
                    visited.add(ng)
                    next_shell.append(ng)
                    edge_data = graph.get_edge_data(nodes,ng)[0]
                    wbo = float(edge_data.get("feature_wiberg_bond_order",1.0))
                    z_num = float(graph.nodes[ng].get("feature_atomic_number",0))
                    signature.append((z_num,wbo))
            signature.sort(key=lambda x:(x[0],x[1]), reverse=True)
            state[ancillary]["score_sequence"].append(signature)
            state[ancillary]["current_shell"] = next_shell
    symmetry = {}
    def compare_ancillary(ancillary_a,ancillary_b): #compare each leg per shell to assign priority
        sequence_a = state[ancillary_a]["score_sequence"]
        sequence_b = state[ancillary_b]["score_sequence"]
        pair = tuple(sorted((ancillary_a,ancillary_b)))
        for sig_a, sig_b in zip(sequence_a,sequence_b):
            for (z_a, wbo_a), (z_b, wbo_b) in zip(sig_a, sig_b):
                if z_a > z_b:
                    return 1
                if z_a < z_b:
                    return -1
                if (wbo_a - wbo_b) > wbo_tol: #must be above the tolerance to be considered different
                    return 1
                if (wbo_b - wbo_a) > wbo_tol:
                    return -1
            if len(sig_a) > len(sig_a):
                return 1
            if len(sig_a) < len(sig_b):
                return -1
        symmetry[pair] = "symmetric"
        if ancillary_a < ancillary_b:
            return 1
        if ancillary_a > ancillary_b:
            return -1
        return 0
    ranked_ancillary = sorted(ancillaries, key=functools.cmp_to_key(compare_ancillary), reverse=True)
    symmetry_list = []
    for key, value in symmetry.items():
        symmetry_list.append(key)
    return ranked_ancillary, symmetry_list

#xyz position alignment
def alignment(r_indices,xyz,metal_idx,centroid,norm,spr, l_indices): #align the complex
    #align such that the centroid of the ring is the origin with the heaviest ring carbon as anchor
    centered_whole = xyz.positions - centroid
    z_axis = norm
    x_axis = spr
    metal_vector = centered_whole[metal_idx]
    if np.dot(z_axis,metal_vector) < 0:
        z_axis = -z_axis
    y_axis = np.cross(z_axis,x_axis)
    rotation = np.vstack([x_axis,y_axis,z_axis])
    aligned = np.dot(centered_whole,rotation.T)
    anchor = aligned[r_indices[0]]
    theta = np.arctan2(anchor[1],anchor[0])
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    r_z = np.array([[cos_t, -sin_t, 0],
                    [sin_t, cos_t, 0],
                    [0, 0, 1]])
    centroid_aligned = np.dot(aligned,r_z.T)
    #align such that the metal center is at the origin
    metal_place = centroid_aligned[metal_idx]
    metal_aligned = centroid_aligned - metal_place
    #align such that the anchor is the priority ligand
    priority = l_indices[0]
    priority_vec = aligned[priority]
    phi = np.arctan2(priority_vec[1],priority_vec[0])
    r_phi = np.array([[np.cos(-phi), -np.sin(-phi), 0],
                      [np.sin(-phi), np.cos(-phi), 0],
                      [0, 0, 1]])
    ligand_aligned = (r_phi @ aligned.T).T
    #make sure that the second priority is always at the positive y-axis, flip if necessary
    ligand_y = l_indices[1]
    ligand_aligned_y = ligand_aligned[ligand_y][1]
    if ligand_aligned_y < 0:
        ligand_aligned[:,1] *= -1
        is_reflected = 1
    else:
        is_reflected = 0
    #metal at origin again
    metal_place_2 = ligand_aligned[metal_idx]
    metal_aligned_2 = ligand_aligned - metal_place_2
    return x_axis, z_axis, centroid_aligned, metal_aligned, ligand_aligned, metal_aligned_2, is_reflected

#haptotropic shift metrics
def ring_slippage(aligned,m_index): #find the slippage metric between the metal and ring
    metal_aligned_pos = aligned[m_index]
    magnitude = np.linalg.norm(metal_aligned_pos)
    slip_angle = np.degrees(np.arccos(metal_aligned_pos[2]/magnitude))
    planar_slip = np.sqrt(metal_aligned_pos[0]**2 + metal_aligned_pos[1]**2)
    slip_x = metal_aligned_pos[0]
    slip_y = metal_aligned_pos[1]
    return magnitude, slip_angle, planar_slip, slip_x, slip_y
def ring_folding(cc_bonds,aligned,full_idx): #find the folding angle with two flap atoms
    flap_idx = full_idx[-2:]
    bound_idx = full_idx[:-2]
    set_bond = set(flap_idx)
    cc_set = [set(x) for x in cc_bonds]
    is_contiguous = set_bond in cc_set
    bound_coord = aligned[bound_idx]
    flap_coord = aligned[flap_idx]
    _, _, normal_bound, _ = best_fit_plane(bound_coord)
    normal_bound = normal_bound / np.linalg.norm(normal_bound)
    if is_contiguous:
       hinge_idx = []
       for f in flap_idx:
           for b in bound_idx:
               tuple_hinge = tuple((b,f))
               if set(tuple_hinge) in cc_set:
                  hinge_idx.append(b)
       hinge_coord = aligned[hinge_idx]
       flap_plane = np.vstack((flap_coord,np.array(hinge_coord)))
       _,_,normal_flap,_ = best_fit_plane(flap_plane)
       normal_flap = normal_flap/np.linalg.norm(normal_flap)
       angle_rad = np.arccos(np.clip(np.dot(normal_bound,normal_flap),-1.0, 1.0))
       angle_degree = np.degrees(angle_rad)
       if angle_degree > 90:
          angle_degree = 180 - angle_degree
       return angle_degree
    else:
       return 0.0

#determine the continuous shape measure from different coordinate modes
def shape_determination(coordinates_leg, coordinates_tetrahedron):
    shape_3_vertices = Shape(coordinates_leg)
    shape_4_vertices = Shape(coordinates_tetrahedron)
    vt_3 = shape_3_vertices.measure("vT-3", central_atom=1)
    f_voc_3 = shape_3_vertices.measure("fvOC-3", central_atom=1)
    t_4 = shape_4_vertices.measure("T-4", central_atom=1)
    vtbpy_4 = shape_4_vertices.measure("vTBPY-4", central_atom=1)
    shape_4_no_metal = Shape(coordinates_tetrahedron[1:])
    t_4_nm = shape_4_no_metal.measure("T-4")
    vtbpy_4_nm = shape_4_no_metal.measure("vTBPY-4")
    return vt_3, f_voc_3, t_4, vtbpy_4, t_4_nm, vtbpy_4_nm

#determination of chirality and symmetry of complex coordination sphere
def chirality_and_volume(position,position2,sym): #find the chirality product and the geometrical volumes of the tripod legs and the tetrahedron structure
    #calculate the tripod vectors
    v0 = position2[1]
    v1 = position2[2]
    v2 = position2[3]
    #calculate the tetrahedron vectors
    v3 = position[1] - position[-1]
    v4 = position[2] - position[-1]
    v5 = position[3] - position[-1]
    #calculate the chirality product
    ed = position[2] - position[1]
    ca = position[-1]
    cb = position[3]
    #normalize vector for chirality determination
    ed = ed / np.linalg.norm(ed)
    ca = ca / np.linalg.norm(ca)
    cb = cb / np.linalg.norm(cb)
    tripod_vol = np.abs(np.dot(v0,np.cross(v1,v2)))
    chirality_product = 0.0
    if not sym:
        chirality_product += np.dot(np.cross(ca,cb),ed)
    else:
        chirality_product += 0.0
    tetrahedron_vol = np.abs(np.dot(v3,np.cross(v4,v5)) / 6.0)
    return chirality_product, tripod_vol, tetrahedron_vol
def ring_symmetry_position(cc_pair,sym,size): #find symmetry patterns of ring
    ring_adj = defaultdict(list)
    for el1, el2 in cc_pair:
        ring_adj[el1].append(el2)
        ring_adj[el2].append(el1)
    nodes = list(ring_adj.keys())
    topological_matrix = {}
    for start_node in nodes:
        topological_matrix[start_node] = {start_node: 0}
        queue = deque([(start_node,0)])
        while queue:
            curr_node, current_dist = queue.popleft()
            for neighbor in ring_adj[curr_node]:
                if neighbor not in topological_matrix[start_node]:
                    topological_matrix[start_node][neighbor] = current_dist + 1
                    queue.append((neighbor,current_dist+1))
    max_dist = size // 2 #3 for benzene, 2 for cp
    pattern_count = {point:0 for point in range(1,max_dist+1)}
    for grp in sym:
        if len(grp) < 2:
            continue
        for at1,at2 in combinations(grp,2):
            true_dist = topological_matrix[at1][at2]
            pattern_count[true_dist] += 1
    return [pattern_count[d] for d in range(1,max_dist+1)]

#electronics of the ring and legs
def extract_directed_graph_feature(graph,comb_coord,metal,ring,leg,neigh):
    edges = []
    for atom_init, atom_target, keys, dat in graph.edges(keys=True, data=True):
        if atom_init in comb_coord and atom_target in comb_coord:
            edges.append({"source":atom_init,"target":atom_target,"donor_type":dat.get("feature_donor_nbo_type"), "acceptor_type":dat.get("feature_acceptor_nbo_type"), "e2":dat.get("feature_stabilisation_energy_max"), "acc_occ":dat.get("feature_acceptor_nbo_occupation"), "p_character":dat.get("feature_acceptor_nbo_p_occupation")
                          ,"don_occ":dat.get("feature_donor_nbo_occupation"),"don_p":dat.get("feature_donor_nbo_p_occupation")})
    edges = pd.DataFrame(edges) #utilize dataframe
    #ring backdonation to the empty pi orbital. Measures the capability of the ring to receive backdonation
    edges_m_to_r = edges[(edges["source"]== metal) & (edges["target"].isin(ring)) & (edges["p_character"] > 0.9)]
    edges_m_to_r = edges_m_to_r.groupby(["source","donor_type","acceptor_type","e2","acc_occ"])["target"].apply(set).reset_index()
    #ring to metal donation. Measures the donation ability of the ring's aromatic electrons
    edges_r_to_m = edges[(edges["source"].isin(ring)) & (edges["target"]==metal) & (edges["don_p"]>0.9) & (edges["acceptor_type"].isin(["BD*","LV"]))]
    edges_r_to_m = edges_r_to_m.groupby(["target","donor_type","acceptor_type","e2","don_occ"])["source"].apply(set).reset_index()
    #leg backdonation to the LV or antibonding orbital with its nearest neighbor. If not LV, must include clear interaction with the leg-neighbor antibonding
    edges_m_to_l = edges[(edges["source"]==metal) & (edges["target"].isin(leg))]
    edges_m_to_nb = edges[(edges["source"]==metal) & (edges["target"].isin(neigh))]
    edges_m_to_nb = pd.concat([edges_m_to_l,edges_m_to_nb],ignore_index=True)
    edges_backbonding = edges_m_to_nb.groupby(["source", "donor_type", "acceptor_type", "e2", "acc_occ"])["target"].apply(set).reset_index()
    edges_backbonding_normal = (edges_backbonding["acceptor_type"] != "LV") & (edges_backbonding["target"].apply(len) > 1)
    edges_backbonding_lv = (edges_backbonding["acceptor_type"] == "LV") & (edges_backbonding["target"].apply(len) == 1)
    edges_backbonding_filter = edges_backbonding_normal | edges_backbonding_lv
    edges_backbonding = edges_backbonding[edges_backbonding_filter]
    e2_sum = []
    for e2 in leg:
        is_in_set = edges_backbonding["target"].apply(lambda xu: e2 in xu)
        cond_sum = edges_backbonding.loc[is_in_set, "e2"].sum()
        e2_sum.append(cond_sum)
    #measures the donation capability of the leg to the metal such that it could strengthen the bond with metal
    edges_l_to_m = edges[(edges["source"].isin(leg)) & (edges["target"]==metal)]
    edges_l_to_l = edges[(edges["source"].isin(leg)) & (edges["target"].isin(leg))]
    edges_l_to_m = pd.concat([edges_l_to_m,edges_l_to_l])
    edges_l_to_m = edges_l_to_m.groupby(["source", "donor_type", "acceptor_type", "e2", "acc_occ"])["target"].apply(set).reset_index()
    removed = edges_l_to_m.apply(lambda rw: rw["source"] not in rw["target"], axis=1)
    edges_l_to_m = edges_l_to_m[removed]
    e2_sum_don = []
    for e2 in leg:
        is_in = edges_l_to_m["source"].apply(lambda xu: e2 == xu)
        cond_sum_2 = edges_l_to_m.loc[is_in, "e2"].sum()
        e2_sum_don.append(cond_sum_2)
    #consider anomalous cases from the ring side, including non-aromatic interaction with the metal center
    edges_m_to_r_anomaly = edges[(edges["source"] == metal) & (edges["target"].isin(ring)) & (edges["p_character"] > 0.9) & (edges["acceptor_type"].isin(["LV","3Cn"]))]
    edges_m_to_r_anomaly = edges_m_to_r_anomaly.groupby(["source", "donor_type", "acceptor_type", "e2", "acc_occ"])["target"].apply(set).reset_index()
    edges_r_to_m_anomaly = edges[(edges["source"].isin(ring)) & (edges["target"]==metal) & (edges["donor_type"].isin(["3C","3Cn"])) & (edges["don_p"] < 0.9) & (edges["acceptor_type"].isin(["BD*","LV"]))]
    edges_r_to_m_anomaly = edges_r_to_m_anomaly.groupby(["target", "donor_type", "acceptor_type", "e2", "acc_occ"])["source"].apply(set).reset_index()
    sum_backbonding_pi = sum(edges_m_to_r["e2"])
    sum_backbonding_pi_anomaly = sum(edges_m_to_r_anomaly["e2"])
    sum_bonding_pi = sum(edges_r_to_m["e2"])
    sum_bonding_pi_anomaly = sum(edges_r_to_m_anomaly["e2"])
    return e2_sum, e2_sum_don, sum_backbonding_pi, sum_backbonding_pi_anomaly, sum_bonding_pi, sum_bonding_pi_anomaly

"""data curation"""
#finding piano stool candidates by looping through xyz files
for _, row in tmQMg_Clean.iterrows():
    csd_code = row["id"]
    smiles = row["smiles"]
    charge = row["charge"]
    n_atoms = row["n_atoms"]
    xyz_path = f"{tmQMg_xyz_folder}/{csd_code}.xyz"
    gml_path = f"{tmQMg_gml_folder}/{csd_code}.gml"
    gml_path_2 = f"{tmQMg_gml_2_folder}/{csd_code}.gml"
    if not os.path.exists(xyz_path): #checks for proper mapping of csv and xyz
        continue
    if not os.path.exists(gml_path): #checks for proper mapping of csv and gml
        continue
    if not os.path.exists(gml_path_2): #checks for proper mapping of csv and directed graph gml
        continue
    complexes = read(xyz_path) #allows ase to read every xyz file in the folder
    all_pos = complexes.positions
    all_sym = set(complexes.get_chemical_symbols())
    all_idx = []
    all_idx_sym = []
    for i, atom in enumerate(complexes):
        all_idx.append(i)
        all_idx_sym.append(atom.symbol)
    for s in all_sym: #remove complexes that have atoms not existing in the ligands set
        if s not in ligands:
            continue
    bond_threshold, bond_threshold_raw = bond_distances(all_sym)
    bond_threshold_no_metal = {key:value for key,value in bond_threshold.items() if not set(key).intersection(target_metals)}
    metal_index = None
    metal_symbol = target_metals
    for i, atom in enumerate(complexes): #identify metal index of each xyz file
        if atom.symbol in metal_symbol:
            metal_index = i
            metal_symbol = atom.symbol
            break
        else:
            continue
    metal_atomic_number = complexes.get_atomic_numbers()[metal_index] #metal atomic number
    metal_principle_number = 0 #principle quantum number of d orbitals
    if metal_symbol == "Ir" or metal_symbol == "Os" or metal_symbol == "Re":
        metal_principle_number += 5
    else:
        metal_principle_number += 4
    metal_group_number = 0 #group number
    if metal_symbol == "Ru" or metal_symbol == "Os":
        metal_group_number += 8
    elif metal_symbol == "Re":
        metal_group_number += 7
    else:
        metal_group_number += 9
    electronegativity = metal_electronegativity[metal_symbol] #allen electronegativity of metal
    metal_pos = complexes[metal_index].position #determine coordinates of the metal
    ligand_pos = [] #determine coordinates of ligands
    for i, atom in enumerate(complexes):
        if i == metal_index:
            continue
        ligand_pos.append(atom.position)
    ligand_pos = np.array(ligand_pos)
    metal_coordination = []
    for i, atom in enumerate(complexes): #calculate distance of each atom relative to the metal
        if i == metal_index:
            continue
        dist = np.linalg.norm(atom.position - metal_pos)
        metal_coordination.append((i, atom.symbol, dist))
    pot_org_met_bond = [(i,dist) for i, sym, dist in metal_coordination if dist < bond_threshold[(metal_symbol,sym)] and sym == "C"] #potential organometallic bond
    pot_org_met_indices = [i for i, dist in pot_org_met_bond]
    num_carbons = len(pot_org_met_bond)
    pot_sigma_donors = [(sym, i, dist) for i, sym, dist in metal_coordination if dist < bond_threshold[(metal_symbol,sym)] and sym != "C"] #potential ancillary ligands
    pot_sigma_donors_symbol = [sym for sym, i, dist in pot_sigma_donors]
    pot_sigma_donors_indices = [i for sym, i, dist in pot_sigma_donors]
    num_hetero = len(pot_sigma_donors)

    #smiles pre-processing
    ox_state = ox_from_smiles(smiles, metal_symbol)
    borohydride = carboranes_from_smiles(smiles)
    if ox_state > 5: #exclude high OS cases
        continue
    if borohydride > 3: #removes carborane/borane clusters with assumption of at least three borohydride bonds in smiles
        continue

    #analysis with primary objective to identify the stool head and eliminate certain cases involving stool legs
    if (num_carbons >= 5) and (num_hetero <= 3): #not more than three potential legs
        #finding the ring
        ring_constituents = components(pot_org_met_indices,complexes,bond_threshold,{5,6})
        dump_carbons = components(pot_org_met_indices, complexes, bond_threshold, {2,3,4})
        other_carbon_ligands = components(pot_org_met_indices, complexes, bond_threshold, {1})
        ring_indices = [pot_org_met_indices[i] for i in ring_constituents] #mapping the carbon ring indices to xyz indices
        ring_indices_size = len(ring_indices)
        other_carbon_ligands_indices = [pot_org_met_indices[i] for i in other_carbon_ligands]
        if not ring_constituents:
            continue
        elif ring_constituents:
            if len(ring_constituents) >= 7: #removes metallocenes complexes
               continue

        #finalization of stool legs
        legs_indices = pot_sigma_donors_indices + other_carbon_ligands_indices #to account for possible C=X or other undetected CC pi bond or X-X bond such as oxygen
        legs_indices.sort()
        if len(legs_indices) != 3: #consider only those with 3 potential legs
            continue
        legs_constituents = components(legs_indices,complexes,bond_threshold,{2,3,4,5})
        legs_constituents_remains = components(legs_indices, complexes, bond_threshold, {1})
        indices_ase_legs = [legs_indices[i] for i in legs_constituents]
        indices_remains = [legs_indices[i] for i in legs_constituents_remains] #mapping the ancillary indices to xyz
        sym_remains = [sym for i, sym, dist in metal_coordination if i in indices_remains]
        dist_remains = [dist for i, sym, dist in metal_coordination if i in indices_remains]
        dist_remains_norm = [float(x) for x in dist_remains]
        legs_pos = complexes.positions[indices_remains]
        legs_xyz = legs_pos.tolist()
        #removes cases where there are multiple carbon or heteroatom pi ligand
        if dump_carbons:
            continue
        elif indices_ase_legs:
            continue
        #only give three-legged stool
        if len(indices_remains) != 3:
            continue

        #confirming the ring being a six membered or five membered and other ring data
        ring_pos = complexes.positions[ring_indices]
        ring_xyz = ring_pos.tolist()
        ring_dist_CC = [dist for dist in pdist(ring_pos).tolist() if dist < bond_threshold[("C", "C")]]  #represents bonds as edges
        ring_pairs = [(i, j) for i in range(ring_indices_size) for j in range(i + 1, ring_indices_size)]
        ring_xyz_pairs = [(ring_indices[i], ring_indices[j]) for i, j in ring_pairs]
        bonds = np.where(pdist(ring_pos) < bond_threshold[("C", "C")])[0]
        ring_pair = [ring_xyz_pairs[i] for i in bonds]
        ring_bonds = list(zip([ring_xyz_pairs[i] for i in bonds], ring_dist_CC))
        #checks for cyclic presence with the logic that edges will never be lesser than nodes in a cycle
        if len(ring_dist_CC) < ring_indices_size:
            continue
        #removes minority Ir or Rh arene complexes if any
        if metal_symbol == "Ir" or metal_symbol == "Rh":
            if ring_indices_size == 6:
                continue
        #statistics for the carbon-carbon bond
        ring_dist_CC_std, ring_dist_CC_max, ring_dist_CC_min, ring_dist_CC_mean = statistics(ring_dist_CC)

        #identify ring substituent and further confirmation of a sp2 ring
        pot_substituent = []
        hydrogen_pos = []
        for i, atom in enumerate(complexes):
            if i in ring_indices:
                continue
            if i == metal_index:
                continue
            if atom.symbol == "H":
                continue
            for x, y in enumerate(ring_indices):
                sub_dist = np.linalg.norm(atom.position - ring_pos[x])
                pot_substituent.append((y, i, atom.symbol, sub_dist))
        for i, atom in enumerate(complexes):
            for x, y in enumerate(ring_indices):
                if atom.symbol == "H":
                    sub_dist = np.linalg.norm(atom.position - ring_pos[x])
                    hydrogen_pos.append((y, i, atom.symbol, sub_dist))
        carbon_substituent = [(y, i, sym, dist) for y, i, sym, dist in pot_substituent if (dist < bond_threshold[("C",sym)] and sym == "C") or (dist < bond_threshold[("C",sym)] and sym != "C")]
        ring_substituted = [y for y, i, sym, dist in carbon_substituent]
        ring_substituent = [i for y, i, sym, dist in carbon_substituent]
        substitution_degree = len(ring_substituted)
        hydrogen_entity = [(y, i, sym, dist) for y, i, sym, dist in hydrogen_pos if (dist < bond_threshold[("C","H")])]
        hydrogen_indices = [i for y, i, sym, dist in hydrogen_entity]
        ring_neigh = ring_substituent + hydrogen_indices
        classification_ring, simplified, subgraph = ring_connection(complexes,metal_index,all_pos,bond_threshold,ring_substituted,ring_indices,indices_remains,hydrogen_indices)
        ring_total = []
        for key,value in simplified.items():
            ring_total.append(key)
            for itm in value:
                ring_total.append(itm)
        ring_full_indices = list(set(ring_total))
        ring_full_sym = []
        for i in ring_full_indices:
            ring_full_sym.append(complexes[i].symbol)
        ring_full_pos = complexes.positions[ring_full_indices]
        #removes crooked examples or non sp2 in ring
        all_sp2 = verify_planar(subgraph, ring_indices)
        if not all_sp2:
            continue

        #identify denticity of ancillary ligands
        dentate, ancillary_coordination = denticity(subgraph,indices_remains,ring_indices)
        ancillary_coordination = {k:(list(set(v))) for k, v in ancillary_coordination.items()} #avoid redundancy due to degree > 1
        ancillary_coordination = {k:(None if len(v)==0 else v) for k, v in ancillary_coordination.items()}
        for en in dentate:
            if not en["num_tether"]:
                en["num_tether"].append(0)
        is_dentate = max(dentate, key=lambda dent: dent["denticity"]) #highest value represents denticity of complex
        #exclude more complicated chelate systems and finalize the chelate ring size
        chelate_ring = None
        chelate_ring_size = None
        chelate_ring_tether_number = None
        if is_dentate["denticity"] == 2:
           if len(is_dentate["tether"]) != len(is_dentate["all_tether"]):
               length = [len(x) for x in is_dentate["all_tether"]]
               length.sort()
               length1 = [len(x) for x in is_dentate["tether"]]
               # remove those with similar sized multirings
               if length1 == 2:
                  if length[1]/length[0] <= 3:
                     continue
               else:
                  if length[1]/length[0] <= 2:
                     continue
           chelate_ring = [metal_index] + is_dentate["atoms"] + [item for sublist in is_dentate["tether"] for item in sublist]
           chelate_ring_size = len(chelate_ring)
           chelate_ring_tether_number = complexes[chelate_ring[3:]].get_atomic_numbers()
           chelate_ring_tether_number = chelate_ring_tether_number.tolist()
        elif is_dentate["denticity"] == 3:
            continue
        else:
           chelate_ring = indices_remains[:1] #placeholder for planarity calculations
           chelate_ring_size = 0
           chelate_ring_tether_number = 0
        #create dictionary for relevant chelating information
        denticity_book = {tuple(dent["atoms"]):tuple(dent["tether"]) for dent in dentate}
        denticity_book = {k: (None if v == () else v) for k, v in denticity_book.items()}
        denticity_book = {(k[0] if isinstance(k,tuple) and len(k) == 1 else k):(v[0] if isinstance(v,tuple) and len(v) == 1 else v) for k,v in denticity_book.items()}
        #identify chelating atoms
        chelate = []
        chelate_indices = []
        if is_dentate["denticity"] > 1:
           for i in is_dentate["atoms"]:
               dentate_atoms = complexes[i].symbol
               chelate.append(dentate_atoms)
               chelate_indices.append(i)
        else:
            chelate.append(None)
        average_chelate_atom_num = 0.0
        if chelate_ring_tether_number != 0:
            _, _, _, average_chelate_atom_num = statistics(chelate_ring_tether_number)
        donor_num = None
        donor_ave = 0.0
        if chelate_indices:
            donor_num = complexes[chelate_indices].get_atomic_numbers()
            donor_num = donor_num.tolist()
            _,_,_, donor_ave = statistics(donor_num)

        #finalize the classification of ring and ancillary
        ring_tag = ["bridged","fused_aromatic","heteroring","fulvene","bulky_substituent","simple_substituent"]
        ancillary_tag = ["is_all_monodentate","has_bidentate"]
        classified = extract_features(classification_ring,is_dentate, ring_tag, ancillary_tag)
        #remove heteroring or fulvene examples for simplicity
        if classified["is_heteroring"] == 1:
            continue
        del classified["is_heteroring"]
        if classified["is_fulvene"] == 1:
            continue
        del classified["is_fulvene"]
        #classify unsubstituted if ring has no substituent
        classified["num_ring_substituent"] = len(ring_substituted)
        if classified["num_ring_substituent"] == 0:
            classified["is_unsubstituted_ring"] = 1
        else:
            classified["is_unsubstituted_ring"] = 0

        #define the CIP ordering using bond order data from quantum graphs
        graph_qg_full = nx.read_gml(gml_path, label="id")
        graph_qg_ring = graph_qg_full.copy()
        #remove edges around ring to avoid considering the ring environment
        ring_edges = [(u,v,key) for u, v, key in list(graph_qg_ring.edges(keys=True)) if u in ring_indices and v in ring_indices]
        graph_qg_ring.remove_edges_from(ring_edges)
        #return ordered list and their degree of symmetry
        ordered_CIP_leg, symmetry_leg = cahn_ingold_prelog(indices_remains, graph_qg_full, metal_index)
        ordered_CIP_ring , symmetry_ring = cahn_ingold_prelog(ring_indices, graph_qg_ring, metal_index)
        #identify the atomic number as scores for indices in order
        ordered_CIP_num = complexes[ordered_CIP_leg].get_atomic_numbers()
        ordered_CIP_z = ordered_CIP_num.tolist()
        CIP_scores = dict(zip(ordered_CIP_leg,ordered_CIP_num))
        #identify the score of the centroid, which depend on the total number of carbons
        score_ring = ring_indices_size * 6
        CIP_scores.update({100: score_ring})
        sorted_CIP_dict = dict(sorted(CIP_scores.items(), key=lambda item: item[1], reverse=True))

        #determine ring planarity
        ring_plane, ring_centroid, normal_ring, spread_ring = best_fit_plane(ring_pos)  #main ring
        chelate_ring_pos = [complexes[i].position for i in chelate_ring]
        chelate_planarity, chelate_centroid, chelate_normal, chelate_spread = best_fit_plane(chelate_ring_pos)  #chelate ring, for monodentate the value will always be high
        #alignment of complex where ring centroid is at the origin and where metal is at the origin, with ring-centric anchor or leg-centric anchor
        most_spread, ring_normal, aligned_complex, metal_origin, aligned_ligand, metal_origin_2, reflection = alignment(ordered_CIP_ring,complexes,metal_index,ring_centroid,normal_ring,spread_ring,ordered_CIP_leg)

        #representing the core coordination positions by either three or four coordination number with metal as origin
        #CN=4
        aligned_centroid_pos = metal_origin[ordered_CIP_ring].mean(axis=0)
        sorted_CIP_pos = sorted_CIP_dict.copy()
        for key in sorted_CIP_pos:
            sorted_CIP_pos[key] = metal_origin[key] if key != 100 else aligned_centroid_pos
        tetrahedron_pos = []
        for key,value in sorted_CIP_pos.items():
            tetrahedron_pos.append(value)
        ML4_pos = [metal_origin[metal_index]] + tetrahedron_pos
        #CN=3
        legged_pos = [metal_origin[ind] for ind in ordered_CIP_leg]
        ML3_pos = [metal_origin[metal_index]] + legged_pos

        #calculate relevant ring-metal distortion factors
        ring_dist = [dist for i, dist in pot_org_met_bond if i in ring_indices] #find metal-carbon bond lengths
        ring_dist_norm = [float(x) for x in ring_dist]
        ring_dist_dict = dict(zip(ring_indices,ring_dist_norm))
        ring_dist_dict_sorted = {k:v for k,v in sorted(ring_dist_dict.items(),key=lambda item: item[1])}
        dist_sorted = []
        for k,v in ring_dist_dict_sorted.items():
            dist_sorted.append(k)
        ring_std, ring_max, ring_min, ring_mean = statistics(ring_dist)
        centroid_dist, slippage_angle, lateral_shift, slippage_x, slippage_y = ring_slippage(aligned_ligand,metal_index)
        folding_angle = ring_folding(ring_pair,aligned_complex,dist_sorted)

        #calculate coordination sphere angles
        #bite angle calculation, only relevant descriptor for bidentate
        bite_angle = 0
        if chelate_indices:
            bite_angle += complexes.get_angle(chelate_indices[0],metal_index,chelate_indices[1])
        else:
            bite_angle += 120.0 #placeholder value for bite angle in monodentate ligands, instead of using observed angles
        #calculate centroid-metal-leg angles
        centroid_above = [aligned_centroid_pos] + legged_pos #ensure centroid is always the first
        unit_vector = np.array(centroid_above) / (np.linalg.norm(centroid_above,axis=1, keepdims=True))
        angles = []
        for i in range(4):
            for j in range(i+1,4):
                dot_product = np.clip(np.dot(unit_vector[i],unit_vector[j]),-1.0,1.0)
                angle_deg = np.degrees(np.arccos(dot_product))
                angles.append(angle_deg)
        cent_leg1, cent_leg2, cent_leg3 = angles[0], angles[1], angles[2] #centroid to leg according to CIP order
        #calculate the angle deviation from ideal tetrahedral angle
        _,_,_,mean_angular_dev = statistics([abs(a - 109.47) for a in angles])
        #statistics for leg-only and cent-leg angles
        legs_angle_std, legs_angle_max, legs_angle_min, legs_angle_mean = statistics(angles[3:])
        cent_angle_std, cent_angle_max, cent_angle_min, cent_angle_mean = statistics(angles[:3])
        #centroid to resultant leg vector angle
        leg_sum = -np.sum(legged_pos,axis=0)
        leg_sum = leg_sum / np.linalg.norm(leg_sum)
        centroid_norm = aligned_centroid_pos / np.linalg.norm(aligned_centroid_pos)
        alignment_angle = np.clip(np.dot(leg_sum,centroid_norm),-1.0,1.0)
        alignment_degree = np.degrees(np.arccos(alignment_angle))
        print(csd_code,slippage_angle,alignment_degree)

        #calculate the strength of the metal-leg bonds and find the neighbors
        wbo_z = []
        neighbors_legs = []
        for am in ordered_CIP_leg:
            edge_leg = graph_qg_full.get_edge_data(metal_index,am)[0]
            neighbor_leg = [nb for nb in graph_qg_full.neighbors(am) if nb != metal_index]
            for ngb in neighbor_leg:
                neighbors_legs.append(ngb)
            wbo_leg = float(edge_leg.get("feature_wiberg_bond_order", 1.0))
            wbo_z.append(wbo_leg)
        wbo_std,wbo_max,wbo_min,wbo_mean = statistics(wbo_z)
        sym_ordered_CIP = [all_idx_sym[CIP] for CIP in ordered_CIP_leg]
        ideal_dist = [bond_threshold_raw[(metal_symbol, sym)] for sym in sym_ordered_CIP]
        observed_dist = [np.linalg.norm(CIP) for CIP in legged_pos]
        dist_ratio = [i / j for i,j in zip(observed_dist,ideal_dist)]

        #metal electronics
        metal_lp = float(graph_qg_full.nodes[metal_index].get("feature_lone_pair_max_occupation",0))
        metal_lv = float(graph_qg_full.nodes[metal_index].get("feature_lone_vacancy_min_occupation",0))
        metal_charge = float(graph_qg_full.nodes[metal_index].get("feature_natural_atomic_charge",0))
        metal_n_e = float(graph_qg_full.nodes[metal_index].get("feature_natural_electron_population_valence",0))
        metal_n_d = float(graph_qg_full.nodes[metal_index].get("feature_natural_electron_configuration_d_occupation",0))
        metal_n_lp = graph_qg_full.nodes[metal_index].get("feature_n_lone_pairs")
        metal_n_lv = graph_qg_full.nodes[metal_index].get("feature_n_lone_vacancies")
        ratio_d_e = metal_n_d / metal_n_e #population of d electrons

        #calculation of the continuous shape measure
        CShM_tet_3, CShM_oct_3, CShM_tet_4, CShM_bpy_4, CShM_tet_4_nm, CShM_bpy_4_nm = shape_determination(ML3_pos, ML4_pos)
        delta_tetrahedron = CShM_tet_4 - CShM_tet_4_nm
        delta_bipyramidal = CShM_bpy_4 - CShM_bpy_4_nm
        ratio_oct_tet = CShM_oct_3 / CShM_tet_3
        ratio_tet_bpy = CShM_tet_4 / CShM_bpy_4

        #calculate the chirality product, tetrahedral volume, and the tripod volume
        metal_chirality, tripod, tetrahedral = chirality_and_volume(ML4_pos,ML3_pos,symmetry_leg)
        tetrahedron_to_tripod = tetrahedral / tripod
        configuration = np.sign(metal_chirality)
        config = ""
        if configuration == -1.0:
            classified["is_chiral_at_metal"] = 1
            classified["configuration"] = 1
            config = "R"
        elif configuration == 1.0:
            classified["is_chiral_at_metal"] = 1
            classified["configuration"] = -1
            config = "S"
        else:
            classified["is_chiral_at_metal"] = 0
            classified["configuration"] = 0
            config = ""
        degree_of_symmetry = len(symmetry_leg)
        #determine the total number of symmetrical carbons in ring
        symmetry_graph = nx.Graph()
        symmetry_graph.add_edges_from(symmetry_ring)
        symmetry_ring_final = [list(sorted(con)) for con in nx.connected_components(symmetry_graph)]
        symmetry_set = set([sub for sublist in symmetry_ring_final for sub in sublist])
        for itm in ordered_CIP_ring:
            if itm not in symmetry_set:
                symmetry_ring_final.append([itm])
        for itm in symmetry_ring_final:
            if len(itm) == ring_indices_size: #perfectly symmetric (unsubstituted or fully substituted)
                classified["has_highly_symmetric_ring"] = 1
            else:
                classified["has_highly_symmetric_ring"] = 0
        #determine the degree of chaos from asymmetrical substitution
        entropy = 0.0
        for group in symmetry_ring_final:
            p_i = len(group) / ring_indices_size
            entropy -= p_i * np.log(p_i)
        #determine the pattern of substitution based on number of patterns
        ring_sub_pattern = ring_symmetry_position(ring_pair,symmetry_ring_final,ring_indices_size)
        num_ortho = ring_sub_pattern[0]
        num_meta = ring_sub_pattern[1]
        num_para = 0
        if ring_indices_size == 6:
            num_para += ring_sub_pattern[2]
        else:
            num_para += 0
        classified["num_1,2_symmetry"] = num_ortho
        classified["num_1,3_symmetry"] = num_meta
        classified["num_1,4_symmetry"] = num_para

        #determine the electronics of the coordination sphere using organometallic rules
        dir_graph = nx.read_gml(gml_path_2,label="id")
        coordination_indices = [metal_index] + ordered_CIP_ring + ordered_CIP_leg + neighbors_legs
        pi_acidity, sigma_basicity, ring_acidity, ring_acidity_anomaly, ring_basicity, ring_basicity_anomaly = extract_directed_graph_feature(dir_graph,coordination_indices,metal_index,ordered_CIP_ring,ordered_CIP_leg,neighbors_legs)
        try:
            ring_softness = ring_acidity / ring_basicity
        except ZeroDivisionError:
            ring_softness = ring_acidity
        ring_softness_corrected = ring_softness
        if ring_basicity_anomaly != 0:
            ring_softness_corrected = ring_acidity / (ring_basicity_anomaly+ring_basicity)
        try:
            hapticity_to_localized = ring_acidity_anomaly / ring_acidity
        except ZeroDivisionError:
            hapticity_to_localized = 0
        acceptor_tendency = [ab / (bd + 0.1) for ab, bd in zip(pi_acidity, sigma_basicity)]
        print(ring_basicity_anomaly,ring_basicity, ring_softness)

        #finalization of data acquisition
        structure = {"n_atoms": n_atoms, "metal":metal_symbol, "all_atom_symbol":all_idx_sym, "charge": charge, "oxidation_state":ox_state, "ring_size": ring_indices_size, "ancillary":sym_ordered_CIP}
        alignment_position = {"csd":csd_code,"ligand_anchor":aligned_ligand, "metal_origin_ligand":metal_origin_2, "reflection":reflection,"carbon_anchor":aligned_complex, "metal_origin":metal_origin}
        topology = {"metal_index": metal_index, "ring_indices":ordered_CIP_ring, "ring_substituent":simplified, "ring_neighbor":ring_neigh, "ancillary_indices":ordered_CIP_leg, "chelate_relevant_information":denticity_book, "ancillary_coordination":ancillary_coordination, "full_ring_indices":ring_full_indices}
        geometry = {"id":csd_code, "metal":metal_symbol, "charge":charge, "oxidation_state":ox_state, "group_number":metal_group_number, "principal_quantum_number":metal_principle_number, "metal_atomic_number":metal_atomic_number, "metal_electronegativity":electronegativity, "metal_charge":metal_charge, "metal_num_LP":metal_n_lp, "metal_basicity":metal_lp, "metal_num_LV":metal_n_lv, "metal_acidity":metal_lv, "metal_d_ratio":ratio_d_e,
                      "ring_size":ring_indices_size, "unsubstituted":bool(classified["is_unsubstituted_ring"]), "fused_ring":bool(classified["is_fused_aromatic"]),
                      "num_substituent":len(ring_substituted), "ansa_bridge":bool(classified["is_bridged"]), "num_bridge":classified["num_bridge"], "ring_rmsd":ring_plane, "centroid_distance":centroid_dist, "metal_ring_mean":ring_mean, "metal_ring_stddev":ring_std,
                      "metal_ring_max":ring_max, "metal_ring_min":ring_min, "slip_angle":slippage_angle, "folding_angle":folding_angle,"lateral_slip_magnitude":lateral_shift, "aromatic_backbonding_stabilization":ring_acidity, "aromatic_donation_stabilization":ring_basicity, "backbonding_asymmetry":hapticity_to_localized, "non_aromatic_donation":ring_basicity_anomaly, "aromatic_softness":ring_softness, "corrected_softness":ring_softness_corrected, "slip_direction_x":slippage_x, "slip_direction_y":slippage_y,
                      "leg_1_atomic_num":ordered_CIP_z[0],"leg_1_wbo":wbo_z[0],"leg_1_backbonding_stabilization":pi_acidity[0], "leg_1_donation_stabilization":sigma_basicity[0], "leg_1_softness":acceptor_tendency[0], "centroid_leg1_angle":cent_leg1, "leg_2_atomic_num":ordered_CIP_z[1], "leg_2_wbo":wbo_z[1], "leg_2_backbonding_stabilization":pi_acidity[1],
                      "leg_2_donation_stability":sigma_basicity[1], "leg_2_softness":acceptor_tendency[1], "centroid_leg2_angle":cent_leg2,"leg_3_atomic_num":ordered_CIP_z[2], "leg_3_wbo":wbo_z[2], "leg_3_backbonding_stabilization":pi_acidity[2],"leg_3_donation_stabilization":sigma_basicity[2],"leg_3_softness":acceptor_tendency[2],"centroid_leg3_angle":cent_leg3,
                      "centroid_leg_mean":cent_angle_mean,"centroid_leg_stddev":cent_angle_std, "bond_order_mean":wbo_mean, "bond_order_stddev":wbo_std,
                      "has_bidentate":bool(classified["has_bidentate"]),"chelating_donor_atoms_average_size":donor_ave,"chelating_backbone_atoms_average_size":average_chelate_atom_num, "chelate_ring_size":chelate_ring_size, "bite_angle":bite_angle, "chelate_planarity":chelate_planarity,
                      "angle_leg_mean":legs_angle_mean, "angle_leg_stddev":legs_angle_std, "mean_angular_deviation_from_tetrahedral":mean_angular_dev, "alignment_angle":alignment_degree,
                      "tripod_volume":tripod, "tetrahedron_volume":tetrahedral, "ratio_volume":tetrahedron_to_tripod, "CShM_oct_3":CShM_oct_3, "CShM_tet_3":CShM_tet_3,
                      "CShM_tet_4":CShM_tet_4, "CShM_pyr_4":CShM_bpy_4, "CShM_empty_tet":CShM_tet_4_nm, "CShM_empty_pyr":CShM_bpy_4_nm, "ratio_oct_tet":ratio_oct_tet,
                      "ratio_tet_pyr":ratio_tet_bpy, "delta_filled_empty_tet":delta_tetrahedron, "delta_filled_empty_pyr":delta_bipyramidal, "metal_chirality":bool(classified["is_chiral_at_metal"]), "metal_coordination_symmetry":degree_of_symmetry,
                      "metal_config":classified["configuration"], "chirality_product":metal_chirality, "ring_perfectly_symmetric":bool(classified["has_highly_symmetric_ring"]),
                      "ring_symmetry_entropy":entropy, "num_symmetry_1,2":num_ortho, "num_symmetry_1,3":num_meta, "num_symmetry_1,4":num_para}

        #exit loop
        if metal_symbol in {"Ru","Rh","Ir"}:
            results.append(geometry)
            results_aligned_array.append(alignment_position)
            results_csd.append(csd_code)
            results_further_info.append({"csd": csd_code,"smiles": smiles, "metal_configuration":config, "structure": structure, "topology": topology, "classification": classified})
        elif metal_symbol == "Os":
            results_Os.append(geometry)
            results_aligned_array_Os.append(alignment_position)
            results_csd_Os.append(csd_code)
            results_further_info_Os.append({"csd": csd_code, "smiles": smiles, "metal_configuration":config, "structure": structure, "topology": topology,"classification": classified})
        else:
            results_Re.append(geometry)
            results_aligned_array_Re.append(alignment_position)
            results_csd_Re.append(csd_code)
            results_further_info_Re.append(
                {"csd": csd_code, "smiles": smiles, "metal_configuration": config, "structure": structure,
                 "topology": topology, "classification": classified})

"""summarize the dataset"""
results_df = pd.DataFrame(results)
results_Os_df = pd.DataFrame(results_Os)
results_Re_df = pd.DataFrame(results_Re)
#for m in target_metals:
    #for c in charges:
       # for r in ring_size:
               # count = results_df[(results_df["metal"] == m) & (results_df["charge"] == c) & (results_df["ring_size"] == r)].shape[0]
               # if count == 0:
                   # continue
                #if m != "Os":
                  # data_summary.append({"Metal":m, "Charge":c, "Ring_Size":r, "Count":count})
              #  else:
                  # data_summary_Os.append({"Metal": m, "Charge": c, "Ring_Size": r, "Count": count})

"""export to csv or json"""
#summary_df = pd.DataFrame(data_summary)
#summary_Os_df = pd.DataFrame(data_summary_Os)
tmQMg_filtered = tmQMg_Met[tmQMg_Met["id"].isin(results_csd)]
tmQMg_filtered_Os = tmQMg_Met[tmQMg_Met["id"].isin(results_csd_Os)]
tmQMg_filtered_Re = tmQMg_Met[tmQMg_Met["id"].isin(results_csd_Re)]
#tmQMg_filtered.to_csv(r"C:\Users\Taufik\Downloads\tmQMgFiltered____RuRhIr.csv", index=False)
#tmQMg_filtered_Os.to_csv(r"C:\Users\Taufik\Downloads\tmQMgFiltered____Os.csv", index=False)
tmQMg_filtered_Re.to_csv(r"C:\Users\Taufik\Downloads\tmQMgFiltered____Re.csv", index=False)
#results_df.to_csv(r"C:\Users\Taufik\Downloads\ResultsPt1__RuRhIr.csv", index=False)
#results_Os_df.to_csv(r"C:\Users\Taufik\Downloads\ResultsPt1__Os.csv", index=False)
results_Re_df.to_csv(r"C:\Users\Taufik\Downloads\ResultsPt1__Re.csv", index=False)
#summary_df.to_csv(r"C:\Users\Taufik\Downloads\ResultsFinalSummary____RuRhIr.csv", index=False)
#summary_Os_df.to_csv(r"C:\Users\Taufik\Downloads\ResultsFinalSummary____Os.csv", index=False)
results_further_df = pd.DataFrame(results_further_info)
#results_further_df.to_json(r"C:\Users\Taufik\Downloads\Results___More_Info_RuRhIr.json", orient="records", indent=2)
results_further_Os_df = pd.DataFrame(results_further_info_Os)
#results_further_Os_df.to_json(r"C:\Users\Taufik\Downloads\Results___More_Info_Os.json", orient="records", indent=2)
results_further_Re_df = pd.DataFrame(results_further_info_Re)
results_further_Re_df.to_json(r"C:\Users\Taufik\Downloads\Results___More_Info_Re.json", orient="records", indent=2)
results_aligned_array_df = pd.DataFrame(results_aligned_array)
results_aligned_array_Os_df = pd.DataFrame(results_aligned_array_Os)
results_aligned_array_Re_df = pd.DataFrame(results_aligned_array_Re)
#results_aligned_array_df.to_json(r"C:\Users\Taufik\Downloads\Results___Aligned_RuRhIr.json", orient="records", indent=2)
#results_aligned_array_Os_df.to_json(r"C:\Users\Taufik\Downloads\Results___Aligned_Os.json", orient="records", indent=2)
results_aligned_array_Re_df.to_json(r"C:\Users\Taufik\Downloads\Results___Aligned_Re.json", orient="records", indent=2)
