"""
scripts/asset_pack_builder/categories/geography.py — 15 Geography, Earth Science, and Environment Models.
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
    make_ring_disc,
    make_sphere,
    make_torus,
)


def _meta(name: str, aliases: List[str], desc: str, features: List[str], subcat: str = "landforms") -> Dict:
    return {
        "name": name,
        "aliases": aliases,
        "category": "Geography, Earth Science and Environment",
        "subcategory": subcat,
        "description": desc,
        "distinguishing_features": features,
        "educational_purpose": f"Visualizes the 3D geological and environmental characteristics of {name}.",
        "license": "CC0-1.0",
        "author": "Magic Mirror Spatial AI Team",
        "source": "Magic Mirror 3D Asset Library",
        "attribution": "Curated for Magic Mirror under CC0 1.0 Universal",
        "download_date": "2026-11-22",
    }


def get_models() -> List[Tuple[Model3D, Dict]]:
    models = []

    # 1. Desktop Globe
    r1 = ModelNode3D("root")
    m1_sphere = make_sphere(radius=0.75, color=(180, 110, 40), roughness=0.5, name="terrestrial_globe")  # Blue oceans
    m1_arc = make_torus(r_major=0.9, r_minor=0.04, color=(40, 180, 220), metallic=0.8, name="meridian_arc")  # Brass arc
    m1_stand = make_cylinder(radius_top=0.35, radius_bottom=0.4, height=0.1, color=(110, 75, 45), name="wooden_base")
    r1.add_child(ModelNode3D("sphere", mesh=m1_sphere, rotation=np.array([0.41, 0, 0], dtype=np.float32)))  # 23.5 deg axial tilt
    r1.add_child(ModelNode3D("arc", mesh=m1_arc, rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    r1.add_child(ModelNode3D("stand", mesh=m1_stand, position=np.array([0, -1.0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Globe", root=r1),
                   _meta("Globe", ["terrestrial globe", "world globe", "earth globe", "desk globe"],
                         "Three-dimensional spherical scale model of Earth mounted on an inclined 23.5-degree meridian arc and tabletop stand.",
                         ["Spherical planetary representation", "Standard 23.5-degree axial tilt angle", "Semi-meridian calibrated support arc and pedestal base"], "cartography")))

    # 2. Mountain Peak
    r2 = ModelNode3D("root")
    m2_rock = make_cone(radius=1.1, height=1.4, segs=16, color=(100, 105, 115), roughness=0.9, name="rock_base")
    m2_snow = make_cone(radius=0.4, height=0.5, segs=16, color=(245, 248, 255), roughness=0.3, name="snow_cap")
    r2.add_child(ModelNode3D("rock", mesh=m2_rock))
    r2.add_child(ModelNode3D("snow", mesh=m2_snow, position=np.array([0, 0.45, 0], dtype=np.float32)))
    models.append((Model3D(concept="Mountain Peak", root=r2),
                   _meta("Mountain Peak", ["mountain", "alpine peak", "summit", "snowy mountain"],
                         "Prominent elevated landform rising conspicuously above surroundings, with rugged crags and perennial glacial snowcap.",
                         ["Steep conical elevated rocky summit", "High-albedo perennial snow/ice cap", "Rugged alpine terrain silhouette"], "landforms")))

    # 3. Active Volcano
    r3 = ModelNode3D("root")
    m3_cone = make_cylinder(radius_top=0.35, radius_bottom=1.2, height=1.2, color=(50, 50, 55), roughness=0.9, name="volcano_cone")
    m3_crater = make_cylinder(radius_top=0.32, radius_bottom=0.2, height=0.1, color=(30, 60, 240), name="magma_lake")  # Glowing lava BGR
    r3.add_child(ModelNode3D("cone", mesh=m3_cone))
    r3.add_child(ModelNode3D("crater", mesh=m3_crater, position=np.array([0, 0.58, 0], dtype=np.float32)))
    models.append((Model3D(concept="Volcano", root=r3),
                   _meta("Volcano", ["active volcano", "stratovolcano", "volcano cone", "caldera"],
                         "Stratovolcano rupture in the crust where molten lava, volcanic ash, and gases escape from a summit caldera crater.",
                         ["Conical basaltic volcanic edifice", "Summit caldera summit crater depression", "Incandescent molten magma/lava vent pool"], "volcanology")))

    # 4. Earth Layers Cross-Section
    r4 = ModelNode3D("root")
    m4_mantle = make_sphere(radius=0.9, color=(30, 40, 200), name="mantle")  # Red mantle BGR
    m4_outer = make_sphere(radius=0.6, color=(30, 130, 230), name="outer_core")  # Orange outer core
    m4_inner = make_sphere(radius=0.32, color=(40, 220, 245), name="inner_core")  # Yellow/white inner core
    r4.add_child(ModelNode3D("mantle", mesh=m4_mantle))
    r4.add_child(ModelNode3D("outer", mesh=m4_outer))
    r4.add_child(ModelNode3D("inner", mesh=m4_inner))
    models.append((Model3D(concept="Earth Layers", root=r4),
                   _meta("Earth Layers", ["internal earth structure", "earth core", "geosphere layers"],
                         "Internal concentric mechanical and compositional layers of Earth: solid inner core, liquid outer core, and mantle.",
                         ["Concentric spherical geological shells", "Dense solid iron-nickel inner core", "Silicate upper mantle and molten outer core zones"], "geology")))

    # 5. Coral Reef
    r5 = ModelNode3D("root")
    m5_seabed = make_cylinder(radius_top=1.0, radius_bottom=1.0, height=0.1, color=(160, 140, 110), name="seabed_sand")
    r5.add_child(ModelNode3D("seabed", mesh=m5_seabed))
    for ang, col in [(0.4, (180, 50, 200)), (1.8, (40, 180, 220)), (3.5, (50, 210, 80)), (5.0, (200, 80, 60))]:
        coral = make_cone(radius=0.2, height=0.6, segs=8, color=col, name=f"coral_{ang}")
        cx, cz = 0.55 * math.cos(ang), 0.55 * math.sin(ang)
        r5.add_child(ModelNode3D(f"coral_{ang}", mesh=coral, position=np.array([cx, 0.3, cz], dtype=np.float32)))
    models.append((Model3D(concept="Coral Reef", root=r5),
                   _meta("Coral Reef", ["marine reef", "coral ecosystem", "barrier reef"],
                         "Underwater ecosystem characterized by reef-building corals formed of colonies of coral polyps held together by calcium carbonate.",
                         ["Calcium carbonate substrate bedrock", "Diverse pigmented branching coral polyp structures", "Marine biodiversity reef habitat architecture"], "marine_ecology")))

    # 6. Alpine Glacier
    r6 = ModelNode3D("root")
    m6_valley = make_box(dx=1.6, dy=0.4, dz=1.8, color=(110, 115, 125), name="glacial_valley")
    m6_ice = make_box(dx=0.8, dy=0.3, dz=1.6, color=(230, 210, 160), name="ice_tongue")  # Glacial blue ice BGR
    r6.add_child(ModelNode3D("valley", mesh=m6_valley))
    r6.add_child(ModelNode3D("ice", mesh=m6_ice, position=np.array([0, 0.15, 0], dtype=np.float32), rotation=np.array([0.15, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Glacier", root=r6),
                   _meta("Glacier", ["valley glacier", "alpine glacier", "ice stream", "serac"],
                         "Persistent body of dense ice that is constantly moving under its own weight, carving a distinctive U-shaped valley.",
                         ["High-density compacted blue ice tongue", "Glaciated mountain valley trough", "Gravitational downhill ice flow gradient"], "glaciology")))

    # 7. Tectonic Plates
    r7 = ModelNode3D("root")
    m7_plate1 = make_box(dx=0.9, dy=0.25, dz=1.2, color=(100, 90, 80), name="continental_plate")
    m7_plate2 = make_box(dx=0.9, dy=0.18, dz=1.2, color=(60, 80, 110), name="oceanic_plate")
    r7.add_child(ModelNode3D("p1", mesh=m7_plate1, position=np.array([-0.5, 0.05, 0], dtype=np.float32)))
    r7.add_child(ModelNode3D("p2", mesh=m7_plate2, position=np.array([0.45, -0.08, 0], dtype=np.float32), rotation=np.array([0, 0, -0.2], dtype=np.float32)))
    models.append((Model3D(concept="Tectonic Plates", root=r7),
                   _meta("Tectonic Plates", ["plate tectonics", "subduction zone", "lithospheric plates", "convergent boundary"],
                         "Convergent lithospheric plate boundary where a denser oceanic plate undergoes subduction beneath lighter continental crust.",
                         ["Thick continental lithospheric slab", "Subducting dense oceanic crustal slab", "Deep oceanic trench convergent margin"], "tectonics")))

    # 8. Quartz Crystal Cluster
    r8 = ModelNode3D("root")
    m8_matrix = make_box(dx=1.0, dy=0.25, dz=1.0, color=(70, 70, 75), name="rock_matrix")
    r8.add_child(ModelNode3D("matrix", mesh=m8_matrix))
    for ang, h, rad in [(0, 0.9, 0.16), (1.3, 0.75, 0.14), (2.5, 0.85, 0.15), (4.2, 0.7, 0.13)]:
        c_body = make_cylinder(radius_top=rad, radius_bottom=rad, height=h, segs=6, color=(220, 230, 240), opacity=0.8, name=f"crystal_{ang}")
        c_tip = make_cone(radius=rad, height=rad * 1.5, segs=6, color=(220, 230, 240), opacity=0.8, name=f"tip_{ang}")
        cx, cz = 0.3 * math.cos(ang), 0.3 * math.sin(ang)
        r8.add_child(ModelNode3D(f"c_b_{ang}", mesh=c_body, position=np.array([cx, 0.15 + h*0.5, cz], dtype=np.float32)))
        r8.add_child(ModelNode3D(f"c_t_{ang}", mesh=c_tip, position=np.array([cx, 0.15 + h + rad*0.75, cz], dtype=np.float32)))
    models.append((Model3D(concept="Crystal", root=r8),
                   _meta("Crystal", ["quartz crystal", "mineral cluster", "crystalline prism", "rock crystal"],
                         "Cluster of prismatic quartz crystals displaying hexagonal columnar habits terminated by sharp six-sided pyramidal facets.",
                         ["Hexagonal columnar prismatic crystal habit", "Pointed six-sided pyramidal apex terminations", "Translucent mineral cluster matrix"], "mineralogy")))

    # 9. Waterfall
    r9 = ModelNode3D("root")
    m9_cliff = make_box(dx=1.2, dy=1.2, dz=0.6, color=(80, 80, 85), name="cliff_ledge")
    m9_water = make_box(dx=0.5, dy=1.1, dz=0.04, color=(230, 190, 80), opacity=0.8, name="falling_water")  # Blue cascade BGR
    m9_pool = make_cylinder(radius_top=0.6, radius_bottom=0.6, height=0.1, color=(210, 160, 60), name="plunge_pool")
    r9.add_child(ModelNode3D("cliff", mesh=m9_cliff, position=np.array([0, 0.3, -0.3], dtype=np.float32)))
    r9.add_child(ModelNode3D("water", mesh=m9_water, position=np.array([0, 0.3, 0.05], dtype=np.float32)))
    r9.add_child(ModelNode3D("pool", mesh=m9_pool, position=np.array([0, -0.3, 0.35], dtype=np.float32)))
    models.append((Model3D(concept="Waterfall", root=r9),
                   _meta("Waterfall", ["cascade", "falls", "plunge waterfall", "cataract"],
                         "River or other body of water's steep fall over a rocky ledge into a plunge pool below.",
                         ["Vertical rock cliff escarpment ledge", "Cascading vertical fluid water sheet", "Basal plunge pool splash basin"], "hydrology")))

    # 10. Cumulus Cloud
    r10 = ModelNode3D("root")
    m10_c1 = make_sphere(radius=0.55, color=(245, 248, 255), roughness=0.9, name="cloud_sphere_1")
    m10_c2 = make_sphere(radius=0.45, color=(245, 248, 255), roughness=0.9, name="cloud_sphere_2")
    m10_c3 = make_sphere(radius=0.48, color=(245, 248, 255), roughness=0.9, name="cloud_sphere_3")
    r10.add_child(ModelNode3D("c1", mesh=m10_c1))
    r10.add_child(ModelNode3D("c2", mesh=m10_c2, position=np.array([0.5, -0.05, 0], dtype=np.float32)))
    r10.add_child(ModelNode3D("c3", mesh=m10_c3, position=np.array([-0.45, -0.05, 0], dtype=np.float32)))
    models.append((Model3D(concept="Cloud", root=r10),
                   _meta("Cloud", ["cumulus", "cumulus cloud", "fluffy cloud", "rain cloud"],
                         "Clouds having flat bases and distinct lumpy, cauliflower-like globular billowing outlines.",
                         ["Billowing cauliflower-like globular tops", "Flat horizontal condensation cloud base", "High-albedo white condensed water droplet volume"], "meteorology")))

    # 11. Tornado
    r11 = ModelNode3D("root")
    m11_funnel = make_cone(radius=0.8, height=1.6, segs=20, color=(140, 145, 155), opacity=0.85, name="tornado_funnel")
    r11.add_child(ModelNode3D("funnel", mesh=m11_funnel, rotation=np.array([math.pi, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Tornado", root=r11),
                   _meta("Tornado", ["twister", "funnel cloud", "cyclone", "whirlwind"],
                         "Violently rotating column of air extending between the surface of the Earth and a cumulonimbus cloud.",
                         ["Inverted narrow vortex funnel cone", "Extreme rotational updraft column", "Atmospheric condensation funnel sheath"], "meteorology")))

    # 12. Water Cycle
    r12 = ModelNode3D("root")
    m12_land = make_box(dx=1.6, dy=0.2, dz=1.2, color=(60, 110, 60), name="landscape_ground")
    m12_ocean = make_box(dx=0.7, dy=0.22, dz=1.2, color=(190, 120, 40), name="ocean_basin")
    m12_cloud = make_sphere(radius=0.35, color=(245, 245, 250), name="rain_cloud")
    r12.add_child(ModelNode3D("land", mesh=m12_land))
    r12.add_child(ModelNode3D("ocean", mesh=m12_ocean, position=np.array([0.45, 0.02, 0], dtype=np.float32)))
    r12.add_child(ModelNode3D("cloud", mesh=m12_cloud, position=np.array([-0.3, 0.7, 0], dtype=np.float32)))
    models.append((Model3D(concept="Water Cycle", root=r12),
                   _meta("Water Cycle", ["hydrologic cycle", "hydrological cycle", "evaporation cycle"],
                         "Biogeochemical cycle describing continuous movement of water on, above, and below surface of the Earth via evaporation and precipitation.",
                         ["Integrated ocean reservoir and terrestrial landscape", "Atmospheric condensation cloud phase", "Continuous hydrological cycle representation"], "hydrology")))

    # 13. Fossil Ammonite
    r13 = ModelNode3D("root")
    m13_spiral = make_helix(radius=0.45, pitch=0.08, turns=2.5, tube_radius=0.1, color=(140, 130, 115), roughness=0.8, name="ammonite_shell")
    r13.add_child(ModelNode3D("spiral", mesh=m13_spiral, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Fossil", root=r13),
                   _meta("Fossil", ["ammonite", "fossil shell", "paleontology specimen", "ammonite fossil"],
                         "Extinct marine mollusc cephalopod shell preserved in sedimentary rock showing characteristic logarithmic planispiral coiling.",
                         ["Planispiral logarithmic shell coiling", "Ribbed chambered shell ornamentation", "Mineralized fossil sedimentary petrifaction"], "paleontology")))

    # 14. Sand Dune
    r14 = ModelNode3D("root")
    m14_dune = make_cylinder(radius_top=0.02, radius_bottom=1.1, height=0.45, segs=16, color=(70, 150, 210), roughness=0.9, name="barchan_dune")  # Desert gold BGR
    r14.add_child(ModelNode3D("dune", mesh=m14_dune, rotation=np.array([0.2, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Sand Dune", root=r14),
                   _meta("Sand Dune", ["dune", "desert dune", "barchan dune", "sand hill"],
                         "Hill of loose sand built by aeolian processes (wind action), characterized by gentle windward slope and steep slip face.",
                         ["Asymmetric windward slope and steep slip face", "Golden desert aeolian sand coloration", "Crescentic barchan landform morphology"], "geomorphology")))

    # 15. River Canyon
    r15 = ModelNode3D("root")
    m15_canyon_l = make_box(dx=0.55, dy=0.8, dz=1.8, color=(90, 80, 110), name="canyon_wall_left")
    m15_canyon_r = make_box(dx=0.55, dy=0.8, dz=1.8, color=(90, 80, 110), name="canyon_wall_right")
    m15_river = make_box(dx=0.4, dy=0.04, dz=1.8, color=(190, 130, 50), name="canyon_river")
    r15.add_child(ModelNode3D("wall_l", mesh=m15_canyon_l, position=np.array([-0.55, 0, 0], dtype=np.float32)))
    r15.add_child(ModelNode3D("wall_r", mesh=m15_canyon_r, position=np.array([0.55, 0, 0], dtype=np.float32)))
    r15.add_child(ModelNode3D("river", mesh=m15_river, position=np.array([0, -0.38, 0], dtype=np.float32)))
    models.append((Model3D(concept="Canyon", root=r15),
                   _meta("Canyon", ["gorge", "river canyon", "ravine", "grand canyon"],
                         "Deep cleft between escarpments or cliffs resulting from weathering and the erosive activity of a river over geologic time.",
                         ["Steep stratified sedimentary canyon walls", "Deep erosional incision gorge profile", "Incised meandering river channel floor"], "geomorphology")))

    return models
