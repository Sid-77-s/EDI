"""
scripts/asset_pack_builder/categories/chemistry.py — 30 Educational Chemistry, Element & Molecular Models.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple
import numpy as np

from integration.generation.model3d import Material3D, Model3D, ModelNode3D
from scripts.asset_pack_builder.primitives import (
    make_box,
    make_cone,
    make_cylinder,
    make_ellipsoid,
    make_helix,
    make_ring_disc,
    make_sphere,
)


def _meta(name: str, aliases: List[str], desc: str, features: List[str], subcat: str = "molecules") -> Dict:
    return {
        "name": name,
        "aliases": aliases,
        "category": "Chemistry and the Periodic Table",
        "subcategory": subcat,
        "description": desc,
        "distinguishing_features": features,
        "educational_purpose": f"Visualizes the 3D structure, bonding, and stoichiometry of {name}.",
        "license": "CC0-1.0",
        "author": "Magic Mirror Spatial AI Team",
        "source": "Magic Mirror 3D Asset Library",
        "attribution": "Curated for Magic Mirror under CC0 1.0 Universal",
        "download_date": "2026-11-22",
    }


def _bohr_atom(element_name: str, symbol: str, z: int, n: int, shells: List[int],
               nucleus_color: Tuple[int, int, int] = (40, 40, 220)) -> Model3D:
    root = ModelNode3D("root")
    # Nucleus
    nuc_mesh = make_sphere(radius=0.35, color=nucleus_color, roughness=0.3, name="nucleus")
    root.add_child(ModelNode3D("nucleus_node", mesh=nuc_mesh))

    # Electron shells
    for s_idx, e_count in enumerate(shells):
        s_rad = 0.75 + s_idx * 0.55
        ring_mesh = make_ring_disc(r_inner=s_rad - 0.015, r_outer=s_rad + 0.015, segs=36,
                                   color=(180, 180, 180), opacity=0.5, name=f"shell_orbit_{s_idx}")
        root.add_child(ModelNode3D(f"shell_ring_{s_idx}", mesh=ring_mesh))
        for e_idx in range(e_count):
            ang = 2.0 * math.pi * e_idx / e_count
            ex = s_rad * math.cos(ang)
            ez = s_rad * math.sin(ang)
            e_mesh = make_sphere(radius=0.08, color=(220, 140, 30), name=f"electron_{s_idx}_{e_idx}")  # Blue/cyan electron
            root.add_child(ModelNode3D(f"electron_node_{s_idx}_{e_idx}", mesh=e_mesh,
                                      position=np.array([ex, 0.0, ez], dtype=np.float32)))
    return Model3D(concept=f"{element_name} Atom", root=root)


def get_models() -> List[Tuple[Model3D, Dict]]:
    models = []

    # 1. Hydrogen Atom (Z=1, 1 electron)
    m1 = _bohr_atom("Hydrogen", "H", 1, 0, [1])
    models.append((m1, _meta("Hydrogen Atom", ["Hydrogen", "H atom", "Bohr hydrogen model"],
                             "Simplest chemical element consisting of 1 proton nucleus orbited by 1 electron.",
                             ["Central proton nucleus", "Single K-shell orbit", "One valence electron"], "elements_and_atoms")))

    # 2. Helium Atom (Z=2, 2 electrons)
    m2 = _bohr_atom("Helium", "He", 2, 2, [2])
    models.append((m2, _meta("Helium Atom", ["Helium", "He atom", "Bohr helium model", "noble gas helium"],
                             "Noble gas element with full valence K-shell containing 2 electrons.",
                             ["Two-proton nucleus", "Stable filled K-shell", "Two paired orbital electrons"], "elements_and_atoms")))

    # 3. Lithium Atom (Z=3: 2, 1)
    m3 = _bohr_atom("Lithium", "Li", 3, 4, [2, 1])
    models.append((m3, _meta("Lithium Atom", ["Lithium", "Li atom", "alkali metal lithium"],
                             "Lightest alkali metal with 2 core electrons and 1 reactive valence electron in L shell.",
                             ["Nucleus with 3 protons", "Inner filled K-shell (2 e-)", "Single outer L-shell valence electron (1 e-)"], "elements_and_atoms")))

    # 4. Carbon Atom (Z=6: 2, 4)
    m4 = _bohr_atom("Carbon", "C", 6, 6, [2, 4], nucleus_color=(40, 40, 40))
    models.append((m4, _meta("Carbon Atom", ["Carbon", "C atom", "Bohr carbon model"],
                             "Fundamental tetravalent element of organic chemistry with 4 valence electrons in L shell.",
                             ["Carbon nucleus", "Inner shell with 2 electrons", "Outer valence shell with 4 electrons"], "elements_and_atoms")))

    # 5. Nitrogen Atom (Z=7: 2, 5)
    m5 = _bohr_atom("Nitrogen", "N", 7, 7, [2, 5], nucleus_color=(210, 80, 40))
    models.append((m5, _meta("Nitrogen Atom", ["Nitrogen", "N atom", "Bohr nitrogen model"],
                             "Pnictogen element forming 78% of Earth's atmosphere with 5 valence electrons.",
                             ["Seven-proton nucleus", "Inner shell with 2 electrons", "Outer valence shell with 5 electrons"], "elements_and_atoms")))

    # 6. Oxygen Atom (Z=8: 2, 6) — STRICTLY single atom, distinct from O2
    m6 = _bohr_atom("Oxygen", "O", 8, 8, [2, 6], nucleus_color=(35, 35, 220))
    models.append((m6, _meta("Oxygen Atom", ["Oxygen", "O atom", "atomic oxygen", "Bohr oxygen model"],
                             "Chalcogen element with 8 protons and 6 valence electrons; distinct from diatomic oxygen molecule O2.",
                             ["Eight-proton nucleus", "Inner shell with 2 electrons", "Outer shell with 6 valence electrons"], "elements_and_atoms")))

    # 7. Sodium Atom (Z=11: 2, 8, 1)
    m7 = _bohr_atom("Sodium", "Na", 11, 12, [2, 8, 1], nucleus_color=(180, 160, 40))
    models.append((m7, _meta("Sodium Atom", ["Sodium", "Na atom", "alkali metal sodium"],
                             "Highly reactive alkali metal with a single valence electron in the M shell.",
                             ["Three concentric electron shells", "Filled K (2) and L (8) shells", "Single reactive M-shell electron"], "elements_and_atoms")))

    # 8. Chlorine Atom (Z=17: 2, 8, 7)
    m8 = _bohr_atom("Chlorine", "Cl", 17, 18, [2, 8, 7], nucleus_color=(50, 180, 60))
    models.append((m8, _meta("Chlorine Atom", ["Chlorine", "Cl atom", "halogen chlorine"],
                             "Halogen element requiring 1 electron to complete its outer octet shell.",
                             ["Three electron shells", "Seven valence electrons in M shell", "Greenish halogen identity"], "elements_and_atoms")))

    # 9. Iron Atom (Z=26: 2, 8, 14, 2)
    m9 = _bohr_atom("Iron", "Fe", 26, 30, [2, 8, 14, 2], nucleus_color=(120, 120, 125))
    models.append((m9, _meta("Iron Atom", ["Iron", "Fe atom", "transition metal iron", "ferrum"],
                             "Transition metal element central to planetary cores, metallurgy, and hemoglobin.",
                             ["Dense transition metal nucleus", "Four distinct electron shells", "Partially filled d-subshell"], "elements_and_atoms")))

    # 10. Gold Atom (Z=79: 2, 8, 18, 32, 18, 1)
    m10 = _bohr_atom("Gold", "Au", 79, 118, [2, 8, 18, 18, 1], nucleus_color=(40, 185, 235))  # Gold BGR
    models.append((m10, _meta("Gold Atom", ["Gold", "Au atom", "aurum", "noble metal gold"],
                             "Dense noble metal resistant to oxidation with relativistic electron stabilization.",
                             ["Heavy mass nucleus", "Lustrous golden core", "Five electron shells represented"], "elements_and_atoms")))

    # 11. Copper Atom (Z=29: 2, 8, 18, 1)
    m11 = _bohr_atom("Copper", "Cu", 29, 35, [2, 8, 18, 1], nucleus_color=(50, 100, 205))
    models.append((m11, _meta("Copper Atom", ["Copper", "Cu atom", "cuprum"],
                             "High electrical conductivity transition metal with filled 3d subshell and 1 4s electron.",
                             ["Reddish-bronze nucleus", "Four electron shells", "Single valence conduction electron"], "elements_and_atoms")))

    # 12. Silicon Atom (Z=14: 2, 8, 4)
    m12 = _bohr_atom("Silicon", "Si", 14, 14, [2, 8, 4], nucleus_color=(150, 160, 170))
    models.append((m12, _meta("Silicon Atom", ["Silicon", "Si atom", "semiconductor silicon"],
                             "Semiconductor metalloid forming the foundation of modern computer microelectronics.",
                             ["Four valence electrons", "Three concentric shells", "Tetravalent bonding potential"], "elements_and_atoms")))

    # 13. Sulfur Atom (Z=16: 2, 8, 6)
    m13 = _bohr_atom("Sulfur", "S", 16, 16, [2, 8, 6], nucleus_color=(40, 200, 220))
    models.append((m13, _meta("Sulfur Atom", ["Sulfur", "S atom", "brimstone"],
                             "Non-metal element essential to proteins and disulfide cross-links.",
                             ["Bright yellow elemental nucleus", "Six valence electrons", "Three electron shells"], "elements_and_atoms")))

    # 14. Uranium Atom (Z=92: 2, 8, 18, 32, 21, 9, 2)
    m14 = _bohr_atom("Uranium", "U", 92, 146, [2, 8, 18, 18, 9, 2], nucleus_color=(60, 170, 70))
    models.append((m14, _meta("Uranium Atom", ["Uranium", "U atom", "actinide uranium"],
                             "Dense actinide element with 92 protons, naturally occurring fissile isotope U-235.",
                             ["Massive actinide nucleus", "Six electron shell levels", "Heavy radioisotope representation"], "elements_and_atoms")))

    # 15. Water Molecule (H2O) — Bent geometry ~104.5 degrees
    r15 = ModelNode3D("root")
    m15_o = make_sphere(radius=0.45, color=(35, 35, 220), name="oxygen_atom")
    m15_h1 = make_sphere(radius=0.25, color=(240, 240, 245), name="hydrogen_1")
    m15_h2 = make_sphere(radius=0.25, color=(240, 240, 245), name="hydrogen_2")
    m15_b1 = make_cylinder(radius_top=0.08, radius_bottom=0.08, height=0.6, color=(180, 180, 180), name="bond_1")
    m15_b2 = make_cylinder(radius_top=0.08, radius_bottom=0.08, height=0.6, color=(180, 180, 180), name="bond_2")
    r15.add_child(ModelNode3D("o_node", mesh=m15_o))
    # H1 at 52.25 deg, H2 at -52.25 deg
    r15.add_child(ModelNode3D("h1_node", mesh=m15_h1, position=np.array([0.65, 0.45, 0.0], dtype=np.float32)))
    r15.add_child(ModelNode3D("h2_node", mesh=m15_h2, position=np.array([-0.65, 0.45, 0.0], dtype=np.float32)))
    r15.add_child(ModelNode3D("b1_node", mesh=m15_b1, position=np.array([0.32, 0.22, 0.0], dtype=np.float32), rotation=np.array([0, 0, -0.65], dtype=np.float32)))
    r15.add_child(ModelNode3D("b2_node", mesh=m15_b2, position=np.array([-0.32, 0.22, 0.0], dtype=np.float32), rotation=np.array([0, 0, 0.65], dtype=np.float32)))
    models.append((Model3D(concept="Water Molecule", root=r15),
                   _meta("Water Molecule", ["Water", "H2O", "H2O molecule", "water chemical structure"],
                         "Polar molecule consisting of 1 oxygen atom bonded to 2 hydrogen atoms at a 104.5 degree bent angle.",
                         ["Central red oxygen atom", "Two white hydrogen atoms", "Characteristic 104.5 degree bent bond angle"], "molecules")))

    # 16. Carbon Dioxide Molecule (CO2) — Linear geometry 180 degrees
    r16 = ModelNode3D("root")
    m16_c = make_sphere(radius=0.4, color=(40, 40, 40), name="carbon_atom")
    m16_o1 = make_sphere(radius=0.38, color=(35, 35, 220), name="oxygen_1")
    m16_o2 = make_sphere(radius=0.38, color=(35, 35, 220), name="oxygen_2")
    m16_b1 = make_cylinder(radius_top=0.09, radius_bottom=0.09, height=0.7, color=(160, 160, 160), name="double_bond_1")
    m16_b2 = make_cylinder(radius_top=0.09, radius_bottom=0.09, height=0.7, color=(160, 160, 160), name="double_bond_2")
    r16.add_child(ModelNode3D("c_node", mesh=m16_c))
    r16.add_child(ModelNode3D("o1_node", mesh=m16_o1, position=np.array([0.9, 0, 0], dtype=np.float32)))
    r16.add_child(ModelNode3D("o2_node", mesh=m16_o2, position=np.array([-0.9, 0, 0], dtype=np.float32)))
    r16.add_child(ModelNode3D("b1_node", mesh=m16_b1, position=np.array([0.45, 0, 0], dtype=np.float32), rotation=np.array([0, 0, 1.57], dtype=np.float32)))
    r16.add_child(ModelNode3D("b2_node", mesh=m16_b2, position=np.array([-0.45, 0, 0], dtype=np.float32), rotation=np.array([0, 0, 1.57], dtype=np.float32)))
    models.append((Model3D(concept="Carbon Dioxide", root=r16),
                   _meta("Carbon Dioxide", ["CO2", "CO2 molecule", "carbon dioxide gas"],
                         "Linear nonpolar molecule composed of a central carbon atom double-bonded to two flanking oxygen atoms.",
                         ["Central black carbon atom", "Two opposing red oxygen atoms", "Linear 180-degree bond geometry"], "molecules")))

    # 17. Oxygen Molecule (O2) — Diatomic molecule (DISTINCT from oxygen atom!)
    r17 = ModelNode3D("root")
    m17_o1 = make_sphere(radius=0.42, color=(35, 35, 220), name="oxygen_atom_1")
    m17_o2 = make_sphere(radius=0.42, color=(35, 35, 220), name="oxygen_atom_2")
    m17_bond = make_cylinder(radius_top=0.1, radius_bottom=0.1, height=0.6, color=(170, 170, 170), name="covalent_double_bond")
    r17.add_child(ModelNode3D("o1_node", mesh=m17_o1, position=np.array([0.45, 0, 0], dtype=np.float32)))
    r17.add_child(ModelNode3D("o2_node", mesh=m17_o2, position=np.array([-0.45, 0, 0], dtype=np.float32)))
    r17.add_child(ModelNode3D("bond_node", mesh=m17_bond, rotation=np.array([0, 0, 1.57], dtype=np.float32)))
    models.append((Model3D(concept="Oxygen Molecule", root=r17),
                   _meta("Oxygen Molecule", ["O2", "O2 molecule", "dioxygen", "molecular oxygen"],
                         "Diatomic oxygen gas (O2) comprising two covalently bonded oxygen atoms; distinct from atomic oxygen.",
                         ["Two identical red oxygen atoms", "Covalent double-bond bridge", "Diatomic molecular symmetry"], "molecules")))

    # 18. Nitrogen Molecule (N2) — Diatomic triple bond
    r18 = ModelNode3D("root")
    m18_n1 = make_sphere(radius=0.4, color=(210, 80, 40), name="nitrogen_atom_1")
    m18_n2 = make_sphere(radius=0.4, color=(210, 80, 40), name="nitrogen_atom_2")
    m18_bond = make_cylinder(radius_top=0.12, radius_bottom=0.12, height=0.55, color=(170, 170, 170), name="triple_bond")
    r18.add_child(ModelNode3D("n1_node", mesh=m18_n1, position=np.array([0.42, 0, 0], dtype=np.float32)))
    r18.add_child(ModelNode3D("n2_node", mesh=m18_n2, position=np.array([-0.42, 0, 0], dtype=np.float32)))
    r18.add_child(ModelNode3D("bond_node", mesh=m18_bond, rotation=np.array([0, 0, 1.57], dtype=np.float32)))
    models.append((Model3D(concept="Nitrogen Molecule", root=r18),
                   _meta("Nitrogen Molecule", ["N2", "N2 molecule", "dinitrogen", "molecular nitrogen"],
                         "Diatomic nitrogen molecule held together by an exceptionally strong covalent triple bond.",
                         ["Two blue nitrogen atoms", "Covalent triple bond cylinder", "Diatomic molecule"], "molecules")))

    # 19. Hydrogen Molecule (H2)
    r19 = ModelNode3D("root")
    m19_h1 = make_sphere(radius=0.28, color=(240, 240, 245), name="hydrogen_1")
    m19_h2 = make_sphere(radius=0.28, color=(240, 240, 245), name="hydrogen_2")
    m19_bond = make_cylinder(radius_top=0.06, radius_bottom=0.06, height=0.4, color=(180, 180, 180), name="single_bond")
    r19.add_child(ModelNode3D("h1_node", mesh=m19_h1, position=np.array([0.3, 0, 0], dtype=np.float32)))
    r19.add_child(ModelNode3D("h2_node", mesh=m19_h2, position=np.array([-0.3, 0, 0], dtype=np.float32)))
    r19.add_child(ModelNode3D("bond_node", mesh=m19_bond, rotation=np.array([0, 0, 1.57], dtype=np.float32)))
    models.append((Model3D(concept="Hydrogen Molecule", root=r19),
                   _meta("Hydrogen Molecule", ["H2", "H2 molecule", "dihydrogen", "molecular hydrogen"],
                         "Lightest molecule in the universe, consisting of two hydrogen atoms sharing a single covalent bond.",
                         ["Two white hydrogen atoms", "Single covalent bond", "Smallest diatomic molecule"], "molecules")))

    # 20. Methane (CH4) — Tetrahedral geometry ~109.5 degrees
    r20 = ModelNode3D("root")
    m20_c = make_sphere(radius=0.4, color=(40, 40, 40), name="carbon_center")
    r20.add_child(ModelNode3D("c_node", mesh=m20_c))
    h_coords = [
        (0.0, 0.65, 0.0),
        (0.61, -0.22, 0.0),
        (-0.31, -0.22, 0.53),
        (-0.31, -0.22, -0.53),
    ]
    for idx, (hx, hy, hz) in enumerate(h_coords):
        h_mesh = make_sphere(radius=0.22, color=(240, 240, 245), name=f"h_{idx}")
        b_mesh = make_cylinder(radius_top=0.07, radius_bottom=0.07, height=0.5, color=(170, 170, 170), name=f"bond_{idx}")
        r20.add_child(ModelNode3D(f"h_node_{idx}", mesh=h_mesh, position=np.array([hx, hy, hz], dtype=np.float32)))
        r20.add_child(ModelNode3D(f"b_node_{idx}", mesh=b_mesh, position=np.array([hx * 0.5, hy * 0.5, hz * 0.5], dtype=np.float32)))
    models.append((Model3D(concept="Methane", root=r20),
                   _meta("Methane", ["CH4", "CH4 molecule", "natural gas methane"],
                         "Simplest alkane hydrocarbon with tetrahedral symmetry (109.5 degree bond angles).",
                         ["Central black carbon atom", "Four white hydrogen ligands", "Regular tetrahedral geometry"], "molecules")))

    # 21. Ammonia (NH3) — Trigonal pyramid
    r21 = ModelNode3D("root")
    m21_n = make_sphere(radius=0.4, color=(210, 80, 40), name="nitrogen_apex")
    r21.add_child(ModelNode3D("n_node", mesh=m21_n, position=np.array([0, 0.25, 0], dtype=np.float32)))
    for idx in range(3):
        ang = 2.0 * math.pi * idx / 3.0
        hx, hy, hz = 0.55 * math.cos(ang), -0.3, 0.55 * math.sin(ang)
        h_mesh = make_sphere(radius=0.22, color=(240, 240, 245), name=f"ammonia_h_{idx}")
        b_mesh = make_cylinder(radius_top=0.07, radius_bottom=0.07, height=0.5, color=(170, 170, 170), name=f"ammonia_b_{idx}")
        r21.add_child(ModelNode3D(f"h_node_{idx}", mesh=h_mesh, position=np.array([hx, hy, hz], dtype=np.float32)))
        r21.add_child(ModelNode3D(f"b_node_{idx}", mesh=b_mesh, position=np.array([hx * 0.5, (hy + 0.25) * 0.5, hz * 0.5], dtype=np.float32)))
    models.append((Model3D(concept="Ammonia", root=r21),
                   _meta("Ammonia", ["NH3", "NH3 molecule", "ammonia chemical structure"],
                         "Inorganic nitrogen hydride with trigonal pyramidal geometry and a lone electron pair at the apex.",
                         ["Central blue nitrogen atom", "Three white hydrogen base atoms", "Trigonal pyramidal geometry"], "molecules")))

    # 22. Sodium Chloride (NaCl) — Ionic crystal lattice
    r22 = ModelNode3D("root")
    for x in (-0.4, 0.4):
        for y in (-0.4, 0.4):
            for z in (-0.4, 0.4):
                is_na = ((x > 0) ^ (y > 0) ^ (z > 0))
                col = (180, 60, 140) if is_na else (60, 200, 70)  # Purple Na+, Green Cl-
                rad = 0.16 if is_na else 0.24
                ion = make_sphere(radius=rad, color=col, name="na_ion" if is_na else "cl_ion")
                r22.add_child(ModelNode3D("ion_node", mesh=ion, position=np.array([x, y, z], dtype=np.float32)))
    models.append((Model3D(concept="Sodium Chloride", root=r22),
                   _meta("Sodium Chloride", ["NaCl", "table salt", "salt crystal", "halite crystal"],
                         "Face-centered cubic (FCC) ionic crystal lattice of alternating sodium cations and chloride anions.",
                         ["Cubic crystal arrangement", "Purple sodium cations (Na+)", "Larger green chloride anions (Cl-)"], "molecules")))

    # 23. Glucose (C6H12O6)
    r23 = ModelNode3D("root")
    for i in range(6):
        ang = 2.0 * math.pi * i / 6.0
        cx, cz = 0.65 * math.cos(ang), 0.65 * math.sin(ang)
        col = (35, 35, 220) if i == 0 else (40, 40, 40)  # 1 oxygen, 5 carbons in pyranose ring
        ring_atom = make_sphere(radius=0.22, color=col, name=f"glucose_ring_{i}")
        r23.add_child(ModelNode3D("ring_node", mesh=ring_atom, position=np.array([cx, 0.08 * (i % 2), cz], dtype=np.float32)))
    models.append((Model3D(concept="Glucose", root=r23),
                   _meta("Glucose", ["C6H12O6", "blood sugar", "dextrose", "glucose molecule"],
                         "Primary biological monosaccharide energy source with six-membered pyranose ring backbone.",
                         ["Six-membered pyranose chair ring", "Red ring oxygen heteroatom", "Hydrocarbon ring scaffold"], "molecules")))

    # 24. Ethanol (C2H5OH)
    r24 = ModelNode3D("root")
    c1 = make_sphere(radius=0.32, color=(40, 40, 40), name="carbon_1")
    c2 = make_sphere(radius=0.32, color=(40, 40, 40), name="carbon_2")
    oh_o = make_sphere(radius=0.3, color=(35, 35, 220), name="hydroxyl_o")
    oh_h = make_sphere(radius=0.2, color=(240, 240, 245), name="hydroxyl_h")
    r24.add_child(ModelNode3D("c1_node", mesh=c1, position=np.array([-0.5, 0, 0], dtype=np.float32)))
    r24.add_child(ModelNode3D("c2_node", mesh=c2, position=np.array([0.1, 0, 0], dtype=np.float32)))
    r24.add_child(ModelNode3D("o_node", mesh=oh_o, position=np.array([0.65, 0.35, 0], dtype=np.float32)))
    r24.add_child(ModelNode3D("h_node", mesh=oh_h, position=np.array([1.05, 0.35, 0], dtype=np.float32)))
    models.append((Model3D(concept="Ethanol", root=r24),
                   _meta("Ethanol", ["C2H5OH", "ethyl alcohol", "grain alcohol", "alcohol molecule"],
                         "Two-carbon primary alcohol with an ethyl group connected to a terminal hydroxyl group.",
                         ["Two-carbon alkane chain", "Red hydroxyl oxygen atom", "White terminal hydrogen"], "molecules")))

    # 25. DNA Double Helix
    r25 = ModelNode3D("root")
    helix_pts = 32
    turns = 1.5
    for k in range(helix_pts):
        ang = 2.0 * math.pi * turns * k / helix_pts
        y = (k / helix_pts - 0.5) * 2.2
        r = 0.55
        x1, z1 = r * math.cos(ang), r * math.sin(ang)
        x2, z2 = -x1, -z1
        s1 = make_sphere(radius=0.08, color=(220, 130, 40), name="backbone_strand_1")  # Blue
        s2 = make_sphere(radius=0.08, color=(40, 180, 220), name="backbone_strand_2")  # Orange
        rung = make_cylinder(radius_top=0.03, radius_bottom=0.03, height=1.1, color=(60, 200, 60), name="base_pair_rung")
        r25.add_child(ModelNode3D("s1_node", mesh=s1, position=np.array([x1, y, z1], dtype=np.float32)))
        r25.add_child(ModelNode3D("s2_node", mesh=s2, position=np.array([x2, y, z2], dtype=np.float32)))
        if k % 2 == 0:
            r25.add_child(ModelNode3D("rung_node", mesh=rung, position=np.array([0, y, 0], dtype=np.float32),
                                      rotation=np.array([0, ang, 1.57], dtype=np.float32)))
    models.append((Model3D(concept="DNA Double Helix", root=r25),
                   _meta("DNA Double Helix", ["DNA", "deoxyribonucleic acid", "DNA molecule", "double helix"],
                         "Biological macromolecule storing genetic instructions in two intertwined helical strands connected by base-pair rungs.",
                         ["Dual intertwined helical strands", "Antiparallel right-handed spiral", "Horizontal base-pair rungs (A-T, G-C)"], "molecules")))

    # 26. Chemistry Beaker
    r26 = ModelNode3D("root")
    m26_glass = make_cylinder(radius_top=0.5, radius_bottom=0.48, height=1.1, color=(235, 240, 245), opacity=0.7, name="beaker_glass")
    m26_liquid = make_cylinder(radius_top=0.46, radius_bottom=0.45, height=0.6, color=(210, 120, 40), opacity=0.85, name="beaker_liquid")  # Blue solution
    r26.add_child(ModelNode3D("glass_node", mesh=m26_glass))
    r26.add_child(ModelNode3D("liquid_node", mesh=m26_liquid, position=np.array([0, -0.22, 0], dtype=np.float32)))
    models.append((Model3D(concept="Chemistry Beaker", root=r26),
                   _meta("Chemistry Beaker", ["Beaker", "glass beaker", "laboratory beaker", "lab beaker"],
                         "Standard laboratory glassware container for stirring, mixing and heating chemical liquids.",
                         ["Cylindrical transparent glass vessel", "Flat stable base", "Colored solution liquid interior"], "chemistry_equipment")))

    # 27. Erlenmeyer Flask
    r27 = ModelNode3D("root")
    m27_cone = make_cylinder(radius_top=0.2, radius_bottom=0.65, height=0.9, color=(235, 240, 245), opacity=0.7, name="flask_body")
    m27_neck = make_cylinder(radius_top=0.2, radius_bottom=0.2, height=0.45, color=(235, 240, 245), opacity=0.7, name="flask_neck")
    r27.add_child(ModelNode3D("cone_node", mesh=m27_cone, position=np.array([0, -0.2, 0], dtype=np.float32)))
    r27.add_child(ModelNode3D("neck_node", mesh=m27_neck, position=np.array([0, 0.45, 0], dtype=np.float32)))
    models.append((Model3D(concept="Erlenmeyer Flask", root=r27),
                   _meta("Erlenmeyer Flask", ["Conical flask", "titration flask", "science flask", "lab flask"],
                         "Conical laboratory flask with wide flat base and narrow neck for swirling liquids without splashing.",
                         ["Conical glassware body", "Narrow cylindrical neck", "Transparent glass material"], "chemistry_equipment")))

    # 28. Test Tube
    r28 = ModelNode3D("root")
    m28_tube = make_cylinder(radius_top=0.15, radius_bottom=0.15, height=1.3, color=(235, 240, 245), opacity=0.7, name="tube_glass")
    m28_bottom = make_sphere(radius=0.15, color=(235, 240, 245), opacity=0.7, name="tube_hemisphere_bottom")
    m28_fluid = make_cylinder(radius_top=0.13, radius_bottom=0.13, height=0.7, color=(50, 180, 230), opacity=0.85, name="reagent_fluid")  # Amber reagent
    r28.add_child(ModelNode3D("tube_node", mesh=m28_tube))
    r28.add_child(ModelNode3D("bot_node", mesh=m28_bottom, position=np.array([0, -0.65, 0], dtype=np.float32)))
    r28.add_child(ModelNode3D("fluid_node", mesh=m28_fluid, position=np.array([0, -0.28, 0], dtype=np.float32)))
    models.append((Model3D(concept="Test Tube", root=r28),
                   _meta("Test Tube", ["Laboratory test tube", "glass test tube", "culture tube", "chemical test tube"],
                         "Slender cylindrical glass container with rounded U-shaped hemispherical bottom for chemical tests.",
                         ["Cylindrical glass body", "Rounded hemispherical bottom", "Colored chemical liquid reagent"], "chemistry_equipment")))

    # 29. Bunsen Burner
    r29 = ModelNode3D("root")
    m29_base = make_cylinder(radius_top=0.55, radius_bottom=0.6, height=0.12, color=(60, 60, 65), metallic=0.7, name="cast_iron_base")
    m29_barrel = make_cylinder(radius_top=0.12, radius_bottom=0.12, height=1.0, color=(160, 160, 165), metallic=0.8, name="burner_barrel")
    m29_collar = make_cylinder(radius_top=0.15, radius_bottom=0.15, height=0.16, color=(50, 160, 200), metallic=0.6, name="air_hole_collar")  # Brass collar
    m29_flame = make_cone(radius=0.18, height=0.55, color=(240, 120, 20), opacity=0.85, name="luminous_blue_flame")
    r29.add_child(ModelNode3D("base_node", mesh=m29_base, position=np.array([0, -0.5, 0], dtype=np.float32)))
    r29.add_child(ModelNode3D("barrel_node", mesh=m29_barrel, position=np.array([0, 0.05, 0], dtype=np.float32)))
    r29.add_child(ModelNode3D("collar_node", mesh=m29_collar, position=np.array([0, -0.28, 0], dtype=np.float32)))
    r29.add_child(ModelNode3D("flame_node", mesh=m29_flame, position=np.array([0, 0.82, 0], dtype=np.float32)))
    models.append((Model3D(concept="Bunsen Burner", root=r29),
                   _meta("Bunsen Burner", ["Lab burner", "gas burner", "laboratory flame"],
                         "Laboratory gas burner producing a single open gas flame for heating and sterilization.",
                         ["Heavy cast iron base", "Vertical metal barrel tube", "Adjustable air collar and cone flame"], "chemistry_equipment")))

    # 30. Pipette
    r30 = ModelNode3D("root")
    m30_stem = make_cylinder(radius_top=0.06, radius_bottom=0.06, height=1.3, color=(235, 240, 245), opacity=0.75, name="pipette_stem")
    m30_tip = make_cone(radius=0.06, height=0.35, color=(235, 240, 245), opacity=0.75, name="dispensing_tip")
    m30_bulb = make_ellipsoid(rx=0.15, ry=0.25, rz=0.15, color=(40, 40, 210), name="rubber_suction_bulb")  # Red bulb
    r30.add_child(ModelNode3D("stem_node", mesh=m30_stem))
    r30.add_child(ModelNode3D("tip_node", mesh=m30_tip, position=np.array([0, -0.82, 0], dtype=np.float32), rotation=np.array([3.14, 0, 0], dtype=np.float32)))
    r30.add_child(ModelNode3D("bulb_node", mesh=m30_bulb, position=np.array([0, 0.85, 0], dtype=np.float32)))
    models.append((Model3D(concept="Pipette", root=r30),
                   _meta("Pipette", ["Eye dropper", "dropper pipette", "laboratory pipette", "transfer pipette"],
                         "Precision laboratory liquid measurement tool featuring a narrow graduated glass stem and rubber suction bulb.",
                         ["Narrow dispensing glass stem", "Tapered delivery tip", "Flexible rubber suction bulb"], "chemistry_equipment")))

    return models
