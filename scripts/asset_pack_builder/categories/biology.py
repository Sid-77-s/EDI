"""
scripts/asset_pack_builder/categories/biology.py — 25 Biology and Human Anatomy Models.
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
    make_platonic_solid,
    make_sphere,
    make_torus,
)


def _meta(name: str, aliases: List[str], desc: str, features: List[str], subcat: str = "anatomy") -> Dict:
    return {
        "name": name,
        "aliases": aliases,
        "category": "Biology and Human Anatomy",
        "subcategory": subcat,
        "description": desc,
        "distinguishing_features": features,
        "educational_purpose": f"Visualizes the 3D biological structure and anatomical organization of {name}.",
        "license": "CC0-1.0",
        "author": "Magic Mirror Spatial AI Team",
        "source": "Magic Mirror 3D Asset Library",
        "attribution": "Curated for Magic Mirror under CC0 1.0 Universal",
        "download_date": "2026-11-22",
    }


def get_models() -> List[Tuple[Model3D, Dict]]:
    models = []

    # 1. Human Heart
    r1 = ModelNode3D("root")
    m1_vent = make_ellipsoid(rx=0.6, ry=0.75, rz=0.55, color=(40, 40, 180), name="ventricles")  # Crimson BGR
    m1_aorta = make_torus(r_major=0.45, r_minor=0.14, color=(30, 30, 210), name="aorta_arch")
    m1_pa = make_cylinder(radius_top=0.15, radius_bottom=0.15, height=0.6, color=(190, 50, 40), name="pulm_art")
    r1.add_child(ModelNode3D("ventricles", mesh=m1_vent))
    r1.add_child(ModelNode3D("aorta", mesh=m1_aorta, position=np.array([0.1, 0.65, 0], dtype=np.float32), rotation=np.array([0, 0, 0.4], dtype=np.float32)))
    r1.add_child(ModelNode3D("pa", mesh=m1_pa, position=np.array([-0.25, 0.6, 0.1], dtype=np.float32), rotation=np.array([0.3, 0, -0.4], dtype=np.float32)))
    models.append((Model3D(concept="Human Heart", root=r1),
                   _meta("Human Heart", ["heart", "cardiac muscle", "cor", "human heart model"],
                         "Four-chambered muscular organ pumping blood through the circulatory system, with ventricles, aortic arch, and pulmonary trunk.",
                         ["Muscular conical ventricular mass", "Prominent aortic arch curve", "Pulmonary artery bifurcations"], "human_anatomy")))

    # 2. Human Brain
    r2 = ModelNode3D("root")
    m2_left = make_ellipsoid(rx=0.45, ry=0.55, rz=0.65, color=(180, 190, 220), roughness=0.7, name="left_hemisphere")
    m2_right = make_ellipsoid(rx=0.45, ry=0.55, rz=0.65, color=(175, 185, 215), roughness=0.7, name="right_hemisphere")
    m2_stem = make_cylinder(radius_top=0.18, radius_bottom=0.12, height=0.6, color=(190, 200, 220), name="brainstem")
    m2_cerebellum = make_ellipsoid(rx=0.4, ry=0.25, rz=0.3, color=(160, 170, 200), name="cerebellum")
    r2.add_child(ModelNode3D("left_hemi", mesh=m2_left, position=np.array([-0.35, 0.2, 0], dtype=np.float32)))
    r2.add_child(ModelNode3D("right_hemi", mesh=m2_right, position=np.array([0.35, 0.2, 0], dtype=np.float32)))
    r2.add_child(ModelNode3D("stem", mesh=m2_stem, position=np.array([0, -0.45, -0.2], dtype=np.float32)))
    r2.add_child(ModelNode3D("cerebellum", mesh=m2_cerebellum, position=np.array([0, -0.25, -0.45], dtype=np.float32)))
    models.append((Model3D(concept="Human Brain", root=r2),
                   _meta("Human Brain", ["brain", "cerebrum", "human brain model", "encephalon"],
                         "Central organ of the human nervous system with bilateral cerebral hemispheres, cerebellum, and brainstem.",
                         ["Bilateral cerebral hemispheres with longitudinal fissure", "Posterior inferior cerebellum", "Inferior projecting brainstem"], "human_anatomy")))

    # 3. Human Skull
    r3 = ModelNode3D("root")
    m3_cranium = make_ellipsoid(rx=0.6, ry=0.65, rz=0.7, color=(210, 225, 235), roughness=0.4, name="cranium")
    m3_jaw = make_box(dx=0.55, dy=0.35, dz=0.45, color=(195, 210, 220), name="mandible")
    m3_eye_l = make_sphere(radius=0.15, color=(30, 30, 35), name="orbit_l")
    m3_eye_r = make_sphere(radius=0.15, color=(30, 30, 35), name="orbit_r")
    r3.add_child(ModelNode3D("cranium", mesh=m3_cranium, position=np.array([0, 0.2, 0], dtype=np.float32)))
    r3.add_child(ModelNode3D("jaw", mesh=m3_jaw, position=np.array([0, -0.4, 0.25], dtype=np.float32)))
    r3.add_child(ModelNode3D("orbit_l", mesh=m3_eye_l, position=np.array([-0.25, 0.15, 0.6], dtype=np.float32)))
    r3.add_child(ModelNode3D("orbit_r", mesh=m3_eye_r, position=np.array([0.25, 0.15, 0.6], dtype=np.float32)))
    models.append((Model3D(concept="Human Skull", root=r3),
                   _meta("Human Skull", ["skull", "cranium", "cranium bone", "human skull model"],
                         "Bony structure forming the head in human skeleton, encasing the brain and featuring orbits and lower jaw.",
                         ["Rounded neurocranium vault", "Bilateral orbital eye sockets", "Articulated mandible jaw region"], "human_anatomy")))

    # 4. Human Lungs
    r4 = ModelNode3D("root")
    m4_trachea = make_cylinder(radius_top=0.12, radius_bottom=0.12, height=0.7, color=(200, 210, 225), name="trachea")
    m4_lung_l = make_ellipsoid(rx=0.4, ry=0.75, rz=0.45, color=(140, 150, 210), name="left_lung")  # Pinkish red
    m4_lung_r = make_ellipsoid(rx=0.48, ry=0.78, rz=0.48, color=(140, 150, 210), name="right_lung")
    r4.add_child(ModelNode3D("trachea", mesh=m4_trachea, position=np.array([0, 0.6, 0], dtype=np.float32)))
    r4.add_child(ModelNode3D("lung_l", mesh=m4_lung_l, position=np.array([-0.5, 0, 0], dtype=np.float32)))
    r4.add_child(ModelNode3D("lung_r", mesh=m4_lung_r, position=np.array([0.52, -0.02, 0], dtype=np.float32)))
    models.append((Model3D(concept="Human Lungs", root=r4),
                   _meta("Human Lungs", ["lungs", "respiratory system", "pulmonary organs", "pulmo"],
                         "Essential respiratory organs with superior trachea tube and asymmetric right (three lobes) and left lobes.",
                         ["Central vertical cartilaginous trachea", "Bilateral spongeous pulmonary lobes", "Asymmetrical left cardiac notch"], "human_anatomy")))

    # 5. Human Eye
    r5 = ModelNode3D("root")
    m5_sclera = make_sphere(radius=0.7, color=(240, 245, 250), roughness=0.2, name="sclera")
    m5_cornea = make_ellipsoid(rx=0.35, ry=0.35, rz=0.2, color=(160, 200, 240), opacity=0.8, name="cornea")
    m5_iris = make_cylinder(radius_top=0.3, radius_bottom=0.3, height=0.04, color=(180, 110, 40), name="iris")  # Brown/amber iris
    m5_pupil = make_cylinder(radius_top=0.12, radius_bottom=0.12, height=0.06, color=(15, 15, 15), name="pupil")
    m5_nerve = make_cylinder(radius_top=0.1, radius_bottom=0.1, height=0.5, color=(190, 200, 210), name="optic_nerve")
    r5.add_child(ModelNode3D("sclera", mesh=m5_sclera))
    r5.add_child(ModelNode3D("cornea", mesh=m5_cornea, position=np.array([0, 0, 0.65], dtype=np.float32)))
    r5.add_child(ModelNode3D("iris", mesh=m5_iris, position=np.array([0, 0, 0.68], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r5.add_child(ModelNode3D("pupil", mesh=m5_pupil, position=np.array([0, 0, 0.7], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r5.add_child(ModelNode3D("nerve", mesh=m5_nerve, position=np.array([0, 0, -0.8], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Human Eye", root=r5),
                   _meta("Human Eye", ["eyeball", "oculus", "eye model", "vision organ"],
                         "Sensory visual organ showing outer white sclera, bulging cornea, colored iris disc, pupil, and posterior optic nerve.",
                         ["Spherical white sclera eyeball", "Anterior curved cornea and colored iris", "Posterior projecting optic nerve"], "human_anatomy")))

    # 6. Human Tooth
    r6 = ModelNode3D("root")
    m6_crown = make_box(dx=0.6, dy=0.55, dz=0.5, color=(240, 245, 250), roughness=0.2, name="crown")
    m6_root1 = make_cone(radius=0.15, height=0.7, color=(215, 220, 230), name="root_1")
    m6_root2 = make_cone(radius=0.15, height=0.7, color=(215, 220, 230), name="root_2")
    r6.add_child(ModelNode3D("crown", mesh=m6_crown, position=np.array([0, 0.3, 0], dtype=np.float32)))
    r6.add_child(ModelNode3D("root1", mesh=m6_root1, position=np.array([-0.18, -0.35, 0], dtype=np.float32)))
    r6.add_child(ModelNode3D("root2", mesh=m6_root2, position=np.array([0.18, -0.35, 0], dtype=np.float32)))
    models.append((Model3D(concept="Human Tooth", root=r6),
                   _meta("Human Tooth", ["molar", "tooth", "incisor", "dental anatomy"],
                         "Molar tooth structure displaying enamelled crown with cusps, cervical neck, and bifurcated root anchors.",
                         ["White enamelled occlusal crown", "Cervical line anatomical neck", "Bifurcated root projections"], "human_anatomy")))

    # 7. Red Blood Cell
    r7 = ModelNode3D("root")
    m7_cell = make_cylinder(radius_top=0.75, radius_bottom=0.75, height=0.25, segs=24, color=(30, 30, 200), name="rbc_disc")
    m7_dimple_t = make_ellipsoid(rx=0.35, ry=0.1, rz=0.35, color=(20, 20, 160), name="dimple_top")
    m7_dimple_b = make_ellipsoid(rx=0.35, ry=0.1, rz=0.35, color=(20, 20, 160), name="dimple_bot")
    r7.add_child(ModelNode3D("disc", mesh=m7_cell))
    r7.add_child(ModelNode3D("dimple_t", mesh=m7_dimple_t, position=np.array([0, 0.1, 0], dtype=np.float32)))
    r7.add_child(ModelNode3D("dimple_b", mesh=m7_dimple_b, position=np.array([0, -0.1, 0], dtype=np.float32)))
    models.append((Model3D(concept="Red Blood Cell", root=r7),
                   _meta("Red Blood Cell", ["erythrocyte", "RBC", "blood cell"],
                         "Biconcave circular disc shaped erythrocyte maximizing surface area for oxygen transport via hemoglobin.",
                         ["Biconcave circular disc profile", "Indented central dimple on both faces", "Vivid crimson erythrocyte color"], "cellular_biology")))

    # 8. Neuron
    r8 = ModelNode3D("root")
    m8_soma = make_sphere(radius=0.45, color=(80, 160, 220), name="soma")
    m8_axon = make_cylinder(radius_top=0.06, radius_bottom=0.06, height=1.4, color=(100, 180, 230), name="axon")
    r8.add_child(ModelNode3D("soma", mesh=m8_soma, position=np.array([-0.6, 0, 0], dtype=np.float32)))
    r8.add_child(ModelNode3D("axon", mesh=m8_axon, position=np.array([0.2, 0, 0], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    # Dendrites
    for ang in [0.8, -0.8, 2.3, -2.3]:
        d = make_cylinder(radius_top=0.03, radius_bottom=0.05, height=0.5, color=(120, 200, 240), name=f"dend_{ang}")
        dx, dy = -0.6 + 0.4 * math.cos(ang), 0.4 * math.sin(ang)
        r8.add_child(ModelNode3D(f"dend_{ang}", mesh=d, position=np.array([dx, dy, 0], dtype=np.float32), rotation=np.array([0, 0, ang], dtype=np.float32)))
    models.append((Model3D(concept="Neuron", root=r8),
                   _meta("Neuron", ["nerve cell", "brain neuron", "multipolar neuron"],
                         "Electrically excitable cell communicating via synapses, with cell body soma, radiating dendrites, and elongated axon.",
                         ["Cell soma with central nucleus", "Branching radiating dendrite tree", "Elongated axon transmission cable"], "cellular_biology")))

    # 9. Animal Cell
    r9 = ModelNode3D("root")
    m9_mem = make_sphere(radius=0.85, color=(160, 210, 230), opacity=0.75, name="cell_membrane")
    m9_nuc = make_sphere(radius=0.35, color=(180, 60, 160), name="nucleus")
    m9_mito1 = make_ellipsoid(rx=0.15, ry=0.08, rz=0.08, color=(40, 180, 180), name="mito_1")
    m9_mito2 = make_ellipsoid(rx=0.15, ry=0.08, rz=0.08, color=(40, 180, 180), name="mito_2")
    r9.add_child(ModelNode3D("membrane", mesh=m9_mem))
    r9.add_child(ModelNode3D("nucleus", mesh=m9_nuc, position=np.array([0.1, 0.1, 0.1], dtype=np.float32)))
    r9.add_child(ModelNode3D("mito1", mesh=m9_mito1, position=np.array([-0.35, -0.2, 0.2], dtype=np.float32), rotation=np.array([0.4, 0.3, 0], dtype=np.float32)))
    r9.add_child(ModelNode3D("mito2", mesh=m9_mito2, position=np.array([0.3, -0.3, -0.2], dtype=np.float32), rotation=np.array([0, 0.6, 0.4], dtype=np.float32)))
    models.append((Model3D(concept="Animal Cell", root=r9),
                   _meta("Animal Cell", ["eukaryotic animal cell", "typical animal cell", "cell model"],
                         "Typical eukaryotic animal cell with flexible outer plasma membrane, prominent nucleus, and organelles.",
                         ["Flexible spherical plasma membrane", "Distinct spherical nucleus with genetic material", "Suspended cytoplasmic organelles"], "cellular_biology")))

    # 10. Plant Cell
    r10 = ModelNode3D("root")
    m10_wall = make_box(dx=1.4, dy=1.1, dz=0.9, color=(50, 160, 60), opacity=0.65, name="cell_wall")  # Green cell wall
    m10_vac = make_ellipsoid(rx=0.45, ry=0.35, rz=0.3, color=(220, 220, 140), opacity=0.8, name="vacuole")
    m10_nuc = make_sphere(radius=0.25, color=(160, 50, 140), name="plant_nuc")
    m10_chloro1 = make_ellipsoid(rx=0.12, ry=0.07, rz=0.07, color=(20, 140, 20), name="chloro_1")
    m10_chloro2 = make_ellipsoid(rx=0.12, ry=0.07, rz=0.07, color=(20, 140, 20), name="chloro_2")
    r10.add_child(ModelNode3D("wall", mesh=m10_wall))
    r10.add_child(ModelNode3D("vacuole", mesh=m10_vac, position=np.array([0, 0, 0], dtype=np.float32)))
    r10.add_child(ModelNode3D("nuc", mesh=m10_nuc, position=np.array([-0.4, 0.25, 0.15], dtype=np.float32)))
    r10.add_child(ModelNode3D("c1", mesh=m10_chloro1, position=np.array([0.45, 0.3, -0.2], dtype=np.float32)))
    r10.add_child(ModelNode3D("c2", mesh=m10_chloro2, position=np.array([0.4, -0.3, 0.2], dtype=np.float32)))
    models.append((Model3D(concept="Plant Cell", root=r10),
                   _meta("Plant Cell", ["eukaryotic plant cell", "plant cell model"],
                         "Eukaryotic plant cell featuring rigid cellulosic cell wall, large central sap vacuole, and photosynthetic chloroplasts.",
                         ["Rigid rectangular cellulose cell wall", "Large central fluid-filled vacuole", "Green chloroplast organelles for photosynthesis"], "cellular_biology")))

    # 11. Bacterium
    r11 = ModelNode3D("root")
    m11_body = make_cylinder(radius_top=0.35, radius_bottom=0.35, height=1.0, color=(100, 190, 100), name="bacillus_body")
    m11_cap1 = make_sphere(radius=0.35, color=(100, 190, 100), name="cap_top")
    m11_cap2 = make_sphere(radius=0.35, color=(100, 190, 100), name="cap_bot")
    m11_flag = make_helix(radius=0.1, pitch=0.2, turns=3.0, tube_radius=0.02, color=(140, 220, 140), name="flagellum")
    r11.add_child(ModelNode3D("body", mesh=m11_body))
    r11.add_child(ModelNode3D("cap_t", mesh=m11_cap1, position=np.array([0, 0.5, 0], dtype=np.float32)))
    r11.add_child(ModelNode3D("cap_b", mesh=m11_cap2, position=np.array([0, -0.5, 0], dtype=np.float32)))
    r11.add_child(ModelNode3D("flag", mesh=m11_flag, position=np.array([0, -1.0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Bacterium", root=r11),
                   _meta("Bacterium", ["bacteria", "bacillus", "prokaryote", "microbe"],
                         "Single-celled prokaryotic microorganism with rod-shaped capsule and locomotory helical flagellum.",
                         ["Cylindrical rod-shaped bacillus capsule", "Hemispherical polar ends", "Whip-like rotary flagellum tail"], "microbiology")))

    # 12. Virus (Bacteriophage)
    r12 = ModelNode3D("root")
    m12_head = make_platonic_solid(solid_type="icosahedron", radius=0.45, color=(60, 180, 220), name="capsid_head")
    m12_sheath = make_cylinder(radius_top=0.08, radius_bottom=0.08, height=0.7, color=(160, 180, 200), name="tail_sheath")
    m12_base = make_cylinder(radius_top=0.18, radius_bottom=0.18, height=0.06, color=(100, 110, 130), name="baseplate")
    r12.add_child(ModelNode3D("head", mesh=m12_head, position=np.array([0, 0.75, 0], dtype=np.float32)))
    r12.add_child(ModelNode3D("sheath", mesh=m12_sheath, position=np.array([0, 0.25, 0], dtype=np.float32)))
    r12.add_child(ModelNode3D("base", mesh=m12_base, position=np.array([0, -0.1, 0], dtype=np.float32)))
    for ang in [0, 1.05, 2.1, 3.14, 4.2, 5.24]:
        leg = make_cylinder(radius_top=0.02, radius_bottom=0.02, height=0.5, color=(140, 140, 150), name=f"leg_{ang}")
        lx, lz = 0.25 * math.cos(ang), 0.25 * math.sin(ang)
        r12.add_child(ModelNode3D(f"fiber_{ang}", mesh=leg, position=np.array([lx, -0.3, lz], dtype=np.float32), rotation=np.array([0.4 * math.sin(ang), 0, 0.4 * math.cos(ang)], dtype=np.float32)))
    models.append((Model3D(concept="Virus", root=r12),
                   _meta("Virus", ["bacteriophage", "phage", "viral particle", "capsid"],
                         "Bacteriophage T4 viral particle with icosahedral capsid head containing genome, contractile sheath, and tail fibers.",
                         ["Icosahedral protein capsid head", "Contractile cylindrical tail sheath", "Six radial tail fiber landing pins"], "microbiology")))

    # 13. Mitochondrion
    r13 = ModelNode3D("root")
    m13_outer = make_ellipsoid(rx=0.45, ry=0.8, rz=0.45, color=(60, 140, 210), opacity=0.75, name="outer_membrane")
    m13_cristae = make_helix(radius=0.25, pitch=0.18, turns=3.5, tube_radius=0.05, color=(20, 80, 180), name="cristae_folds")
    r13.add_child(ModelNode3D("outer", mesh=m13_outer))
    r13.add_child(ModelNode3D("cristae", mesh=m13_cristae))
    models.append((Model3D(concept="Mitochondrion", root=r13),
                   _meta("Mitochondrion", ["mitochondria", "cellular powerhouse", "chondriosome"],
                         "Double-membrane-bound cellular organelle generating most chemical energy ATP via deeply folded internal cristae.",
                         ["Oval rod-shaped outer membrane", "Extensively folded inner cristae convolutions", "Cellular energy generation site"], "cellular_biology")))

    # 14. Chloroplast
    r14 = ModelNode3D("root")
    m14_envelope = make_ellipsoid(rx=0.7, ry=0.35, rz=0.7, color=(30, 150, 40), opacity=0.75, name="chloroplast_envelope")
    r14.add_child(ModelNode3D("envelope", mesh=m14_envelope))
    for dx, dz in [(-0.3, -0.2), (0.2, 0.2), (-0.1, 0.3), (0.3, -0.25)]:
        grana = make_cylinder(radius_top=0.15, radius_bottom=0.15, height=0.3, color=(15, 100, 25), name=f"grana_{dx}")
        r14.add_child(ModelNode3D(f"grana_{dx}", mesh=grana, position=np.array([dx, 0, dz], dtype=np.float32)))
    models.append((Model3D(concept="Chloroplast", root=r14),
                   _meta("Chloroplast", ["chloroplast organelle", "plastid", "photosynthetic organelle"],
                         "Photosynthetic organelle in plant cells with double envelope containing stacked thylakoid grana discs.",
                         ["Biconvex lens outer envelope", "Stacked disc-like thylakoid grana columns", "Chlorophyll pigmentation site"], "cellular_biology")))

    # 15. Chromosome
    r15 = ModelNode3D("root")
    # X-shaped duplicated chromatids
    m15_arm1 = make_cylinder(radius_top=0.1, radius_bottom=0.1, height=1.3, color=(190, 60, 170), name="chromatid_1")
    m15_arm2 = make_cylinder(radius_top=0.1, radius_bottom=0.1, height=1.3, color=(190, 60, 170), name="chromatid_2")
    m15_centro = make_sphere(radius=0.14, color=(240, 200, 40), name="centromere")
    r15.add_child(ModelNode3D("arm1", mesh=m15_arm1, rotation=np.array([0, 0, 0.45], dtype=np.float32)))
    r15.add_child(ModelNode3D("arm2", mesh=m15_arm2, rotation=np.array([0, 0, -0.45], dtype=np.float32)))
    r15.add_child(ModelNode3D("centro", mesh=m15_centro))
    models.append((Model3D(concept="Chromosome", root=r15),
                   _meta("Chromosome", ["sister chromatids", "human chromosome", "genetic chromosome"],
                         "Duplicated thread-like structure of nucleic acids and protein forming characteristic X-shape joined at centromere.",
                         ["Characteristic X-shaped sister chromatid arms", "Central constricted centromere junction", "Evenly dense chromatin arms"], "genetics")))

    # 16. Plant Leaf
    r16 = ModelNode3D("root")
    m16_blade = make_ellipsoid(rx=0.55, ry=0.03, rz=0.95, color=(40, 160, 50), roughness=0.6, name="leaf_blade")
    m16_petiole = make_cylinder(radius_top=0.04, radius_bottom=0.04, height=0.6, color=(30, 120, 40), name="leaf_petiole")
    r16.add_child(ModelNode3D("blade", mesh=m16_blade, position=np.array([0, 0, 0.3], dtype=np.float32)))
    r16.add_child(ModelNode3D("petiole", mesh=m16_petiole, position=np.array([0, 0, -0.6], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Leaf", root=r16),
                   _meta("Leaf", ["plant leaf", "foliage", "tree leaf", "green leaf"],
                         "Dorsiventral photosynthetic plant leaf with broad flat lamina blade, central midrib vein, and supporting petiole stalk.",
                         ["Flat elliptical lamina blade", "Central longitudinal midrib axis", "Basal petiole stem attachment"], "botany")))

    # 17. Flower
    r17 = ModelNode3D("root")
    m17_center = make_sphere(radius=0.25, color=(30, 180, 240), name="receptacle")  # Golden yellow BGR
    r17.add_child(ModelNode3D("center", mesh=m17_center))
    for p in range(6):
        ang = 2.0 * math.pi * p / 6
        petal = make_ellipsoid(rx=0.18, ry=0.05, rz=0.45, color=(180, 40, 200), name=f"petal_{p}")  # Magenta/violet
        px, pz = 0.45 * math.cos(ang), 0.45 * math.sin(ang)
        r17.add_child(ModelNode3D(f"petal_{p}", mesh=petal, position=np.array([px, 0, pz], dtype=np.float32), rotation=np.array([0, ang, 0], dtype=np.float32)))
    m17_stem = make_cylinder(radius_top=0.04, radius_bottom=0.04, height=0.8, color=(30, 130, 40), name="stem")
    r17.add_child(ModelNode3D("stem", mesh=m17_stem, position=np.array([0, -0.5, 0], dtype=np.float32)))
    models.append((Model3D(concept="Flower", root=r17),
                   _meta("Flower", ["blossom", "angiosperm flower", "floral bloom"],
                         "Reproductive structure of angiosperms consisting of central nectar-rich receptacle disc surrounded by radial colored petals.",
                         ["Central circular floral disc", "Six symmetric radiating floral petals", "Inferior supporting pedicel stem"], "botany")))

    # 18. Tree
    r18 = ModelNode3D("root")
    m18_trunk = make_cylinder(radius_top=0.15, radius_bottom=0.25, height=1.0, color=(60, 90, 130), roughness=0.9, name="trunk")  # Brown bark
    m18_crown = make_sphere(radius=0.75, color=(30, 130, 40), roughness=0.8, name="foliage_crown")
    r18.add_child(ModelNode3D("trunk", mesh=m18_trunk, position=np.array([0, -0.4, 0], dtype=np.float32)))
    r18.add_child(ModelNode3D("crown", mesh=m18_crown, position=np.array([0, 0.6, 0], dtype=np.float32)))
    models.append((Model3D(concept="Tree", root=r18),
                   _meta("Tree", ["deciduous tree", "forest tree", "timber tree"],
                         "Perennial woody plant featuring an upright supporting trunk cylinder and broad spherical photosynthetic foliage canopy.",
                         ["Tapered woody trunk cylinder with bark texture", "Elevated globular photosynthetic canopy crown", "Arboreal proportion and scale"], "botany")))

    # 19. Mushroom
    r19 = ModelNode3D("root")
    m19_stipe = make_cylinder(radius_top=0.12, radius_bottom=0.18, height=0.75, color=(225, 230, 235), name="stipe")
    m19_pileus = make_cone(radius=0.6, height=0.45, segs=20, color=(40, 50, 200), name="cap")  # Red cap with white spots
    r19.add_child(ModelNode3D("stipe", mesh=m19_stipe, position=np.array([0, -0.3, 0], dtype=np.float32)))
    r19.add_child(ModelNode3D("cap", mesh=m19_pileus, position=np.array([0, 0.25, 0], dtype=np.float32)))
    models.append((Model3D(concept="Mushroom", root=r19),
                   _meta("Mushroom", ["toadstool", "fungus", "mushroom fruiting body", "agaric"],
                         "Fleshy, spore-bearing fruiting body of a fungus, typically produced above ground, with stipe stem and domed cap.",
                         ["Conical umbrella-shaped pileus cap", "Vertical fleshy stipe stalk", "Fungal fruiting anatomy"], "mycology")))

    # 20. Femur Bone
    r20 = ModelNode3D("root")
    m20_shaft = make_cylinder(radius_top=0.1, radius_bottom=0.1, height=1.4, color=(230, 235, 240), roughness=0.3, name="diaphysis")
    m20_head = make_sphere(radius=0.22, color=(225, 230, 235), name="femur_head")
    m20_condyle1 = make_sphere(radius=0.18, color=(225, 230, 235), name="condyle_l")
    m20_condyle2 = make_sphere(radius=0.18, color=(225, 230, 235), name="condyle_r")
    r20.add_child(ModelNode3D("shaft", mesh=m20_shaft))
    r20.add_child(ModelNode3D("head", mesh=m20_head, position=np.array([-0.2, 0.75, 0], dtype=np.float32)))
    r20.add_child(ModelNode3D("cond_l", mesh=m20_condyle1, position=np.array([-0.12, -0.75, 0], dtype=np.float32)))
    r20.add_child(ModelNode3D("cond_r", mesh=m20_condyle2, position=np.array([0.12, -0.75, 0], dtype=np.float32)))
    models.append((Model3D(concept="Bone", root=r20),
                   _meta("Bone", ["femur", "thigh bone", "human bone", "skeletal bone"],
                         "Human femur thigh bone, the longest and strongest bone in the human skeleton, with femoral head and dual condyles.",
                         ["Long cylindrical cortical diaphysis shaft", "Angled proximal femoral head sphere", "Bilateral distal articular condyles"], "human_anatomy")))

    # 21. DNA Double Helix
    r21 = ModelNode3D("root")
    m21_strand1 = make_helix(radius=0.35, pitch=0.4, turns=3.0, tube_radius=0.035, color=(220, 60, 60), name="strand_1")
    m21_strand2 = make_helix(radius=0.35, pitch=0.4, turns=3.0, tube_radius=0.035, color=(60, 100, 220), name="strand_2")
    r21.add_child(ModelNode3D("strand1", mesh=m21_strand1))
    r21.add_child(ModelNode3D("strand2", mesh=m21_strand2, rotation=np.array([0, math.pi, 0], dtype=np.float32)))
    # Add horizontal base pair steps
    for s in range(12):
        y = -0.5 + s * 0.09
        ang = y * 4.7
        rung = make_cylinder(radius_top=0.02, radius_bottom=0.02, height=0.65, color=(240, 210, 80), name=f"base_{s}")
        r21.add_child(ModelNode3D(f"base_{s}", mesh=rung, position=np.array([0, y, 0], dtype=np.float32), rotation=np.array([0, ang, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="DNA Double Helix", root=r21),
                   _meta("DNA Double Helix", ["DNA", "deoxyribonucleic acid", "double helix", "genetic helix"],
                         "Molecular double helix of DNA consisting of complementary antiparallel polynucleotide chains connected by base pair rungs.",
                         ["Antiparallel intertwined double helices", "Connecting transverse base pair rungs", "Accurate helical pitch and major/minor grooves"], "molecular_biology")))

    # 22. Ribosome
    r22 = ModelNode3D("root")
    m22_large = make_ellipsoid(rx=0.55, ry=0.45, rz=0.5, color=(80, 160, 220), roughness=0.6, name="large_subunit_60s")
    m22_small = make_ellipsoid(rx=0.38, ry=0.25, rz=0.35, color=(120, 200, 240), roughness=0.6, name="small_subunit_40s")
    r22.add_child(ModelNode3D("large_sub", mesh=m22_large, position=np.array([0, 0.25, 0], dtype=np.float32)))
    r22.add_child(ModelNode3D("small_sub", mesh=m22_small, position=np.array([0, -0.3, 0], dtype=np.float32)))
    models.append((Model3D(concept="Ribosome", root=r22),
                   _meta("Ribosome", ["ribosomal complex", "protein factory", "70S/80S ribosome"],
                         "Complex cellular machine consisting of large and small ribonucleoprotein subunits responsible for biological protein synthesis.",
                         ["Asymmetric large ribosomal subunit", "Interlocking small ribosomal subunit", "Inter-subunit mRNA/tRNA binding cleft"], "molecular_biology")))

    # 23. Stomach
    r23 = ModelNode3D("root")
    m23_body = make_torus(r_major=0.5, r_minor=0.28, color=(120, 140, 210), name="stomach_curv")
    m23_esoph = make_cylinder(radius_top=0.12, radius_bottom=0.12, height=0.4, color=(160, 170, 220), name="esophagus_inlet")
    m23_pylorus = make_cylinder(radius_top=0.1, radius_bottom=0.1, height=0.4, color=(160, 170, 220), name="pylorus_outlet")
    r23.add_child(ModelNode3D("body", mesh=m23_body))
    r23.add_child(ModelNode3D("esoph", mesh=m23_esoph, position=np.array([-0.45, 0.45, 0], dtype=np.float32), rotation=np.array([0, 0, -0.3], dtype=np.float32)))
    r23.add_child(ModelNode3D("pylorus", mesh=m23_pylorus, position=np.array([0.45, -0.45, 0], dtype=np.float32), rotation=np.array([0, 0, 0.4], dtype=np.float32)))
    models.append((Model3D(concept="Stomach", root=r23),
                   _meta("Stomach", ["human stomach", "gastric organ", "ventriculus"],
                         "Muscular, hollow organ in the gastrointestinal tract with greater and lesser curvatures, fundus, and pylorus.",
                         ["C-shaped greater and lesser curvatures", "Superior cardiac esophageal inlet", "Inferior pyloric duodenal outlet"], "human_anatomy")))

    # 24. Kidney
    r24 = ModelNode3D("root")
    m24_body = make_ellipsoid(rx=0.45, ry=0.75, rz=0.4, color=(40, 50, 160), name="renal_cortex")  # Deep red/brown
    m24_ureter = make_cylinder(radius_top=0.06, radius_bottom=0.06, height=0.7, color=(220, 220, 230), name="ureter_tube")
    m24_vein = make_cylinder(radius_top=0.08, radius_bottom=0.08, height=0.4, color=(190, 40, 30), name="renal_vein")
    r24.add_child(ModelNode3D("cortex", mesh=m24_body))
    r24.add_child(ModelNode3D("ureter", mesh=m24_ureter, position=np.array([-0.2, -0.6, 0], dtype=np.float32), rotation=np.array([0, 0, 0.2], dtype=np.float32)))
    r24.add_child(ModelNode3D("vein", mesh=m24_vein, position=np.array([-0.3, 0.05, 0], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="Kidney", root=r24),
                   _meta("Kidney", ["human kidney", "renal organ", "ren"],
                         "Bean-shaped organ filtering blood to extract waste, with medial indented hilum providing entry for renal vessels and ureter.",
                         ["Characteristic reniform bean silhouette", "Medial concave hilum indentation", "Emerging renal vascular pedicle and ureter"], "human_anatomy")))

    # 25. Skeleton Torso
    r25 = ModelNode3D("root")
    m25_spine = make_cylinder(radius_top=0.08, radius_bottom=0.08, height=1.3, color=(220, 225, 235), name="vertebrae")
    m25_sternum = make_box(dx=0.1, dy=0.6, dz=0.05, color=(210, 220, 230), name="sternum")
    r25.add_child(ModelNode3D("spine", mesh=m25_spine, position=np.array([0, 0, -0.2], dtype=np.float32)))
    r25.add_child(ModelNode3D("sternum", mesh=m25_sternum, position=np.array([0, 0.1, 0.25], dtype=np.float32)))
    for r_idx in range(5):
        y = -0.2 + r_idx * 0.15
        rib = make_torus(r_major=0.35 + 0.03 * r_idx, r_minor=0.025, color=(220, 225, 235), name=f"rib_{r_idx}")
        r25.add_child(ModelNode3D(f"rib_{r_idx}", mesh=rib, position=np.array([0, y, 0], dtype=np.float32)))
    models.append((Model3D(concept="Skeleton Torso", root=r25),
                   _meta("Skeleton Torso", ["ribcage", "thoracic cage", "skeleton chest", "ribs"],
                         "Thoracic cage formed by the vertebral column, twelve pairs of curved ribs, and anterior breastbone sternum.",
                         ["Posterior segmented vertebral column", "Curved bilateral thoracic ribs", "Anterior central breastbone sternum"], "human_anatomy")))

    return models
