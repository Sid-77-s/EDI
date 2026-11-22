"""
scripts/asset_pack_builder/categories/astronomy.py — 25 Educational Astronomy & Space Models.
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
    make_ring_disc,
    make_sphere,
)


def _meta(name: str, aliases: List[str], desc: str, features: List[str], subcat: str = "celestial_bodies") -> Dict:
    return {
        "name": name,
        "aliases": aliases,
        "category": "Astronomy and Space",
        "subcategory": subcat,
        "description": desc,
        "distinguishing_features": features,
        "educational_purpose": f"Visualizes the 3D structure and astronomical properties of {name}.",
        "license": "CC0-1.0",
        "author": "Magic Mirror Spatial AI Team",
        "source": "Magic Mirror 3D Asset Library",
        "attribution": "Curated for Magic Mirror under CC0 1.0 Universal",
        "download_date": "2026-11-22",
    }


def get_models() -> List[Tuple[Model3D, Dict]]:
    models = []

    # 1. Mercury
    m1 = make_sphere(radius=0.7, color=(140, 140, 145), roughness=0.85, name="mercury_body")
    models.append((Model3D(concept="Mercury", root=ModelNode3D("root", mesh=m1)),
                   _meta("Mercury", ["Planet Mercury", "Mercury planet", "first planet"],
                         "Smallest terrestrial planet, cratered grey surface with high iron core.",
                         ["Cratered surface", "Terrestrial rocky sphere", "Dark grey appearance"], "planets")))

    # 2. Venus
    m2 = make_sphere(radius=0.95, color=(160, 215, 235), roughness=0.3, name="venus_cloud_deck")
    models.append((Model3D(concept="Venus", root=ModelNode3D("root", mesh=m2)),
                   _meta("Venus", ["Planet Venus", "Venus planet", "morning star"],
                         "Second planet from Sun, shrouded in thick yellowish-white sulfuric acid clouds.",
                         ["Dense cloud atmosphere", "Warm yellowish-cream hue", "Spherical planetary form"], "planets")))

    # 3. Earth
    r3 = ModelNode3D("root")
    m3_ocean = make_sphere(radius=1.0, color=(215, 105, 30), roughness=0.3, name="earth_ocean")  # BGR blue
    m3_land = make_ellipsoid(rx=1.01, ry=0.9, rz=1.01, color=(45, 145, 55), roughness=0.7, name="earth_continents")
    m3_ice = make_sphere(radius=1.015, color=(245, 245, 250), roughness=0.5, name="earth_polar_ice")
    r3.add_child(ModelNode3D("ocean_node", mesh=m3_ocean))
    r3.add_child(ModelNode3D("land_node", mesh=m3_land, scale=np.array([0.98, 0.7, 0.98], dtype=np.float32)))
    r3.add_child(ModelNode3D("ice_node", mesh=m3_ice, scale=np.array([0.4, 0.25, 0.4], dtype=np.float32), position=np.array([0, 0.88, 0], dtype=np.float32)))
    models.append((Model3D(concept="Earth", root=r3),
                   _meta("Earth", ["Planet Earth", "The Earth", "Blue planet", "world globe"],
                         "Third planet from Sun, ocean world with continents and polar ice caps.",
                         ["Blue ocean base", "Green continental landmasses", "White polar ice cap"], "planets")))

    # 4. Mars
    r4 = ModelNode3D("root")
    m4_body = make_sphere(radius=0.75, color=(45, 85, 205), roughness=0.8, name="mars_crust")  # BGR rusty red
    m4_ice = make_sphere(radius=0.76, color=(245, 245, 250), roughness=0.4, name="mars_north_cap")
    r4.add_child(ModelNode3D("crust_node", mesh=m4_body))
    r4.add_child(ModelNode3D("polar_node", mesh=m4_ice, scale=np.array([0.3, 0.15, 0.3], dtype=np.float32), position=np.array([0, 0.68, 0], dtype=np.float32)))
    models.append((Model3D(concept="Mars", root=r4),
                   _meta("Mars", ["Planet Mars", "Red planet", "fourth planet"],
                         "Fourth planet, iron-oxide rich red surface and bright white carbon-dioxide/water ice cap.",
                         ["Rusty red-orange surface", "White northern polar cap", "Rocky desert terrain"], "planets")))

    # 5. Jupiter
    r5 = ModelNode3D("root")
    m5_gas = make_sphere(radius=1.35, color=(120, 175, 215), roughness=0.5, name="jupiter_atmosphere")
    m5_belt = make_cylinder(radius_top=1.36, radius_bottom=1.36, height=0.4, color=(60, 100, 170), name="jupiter_equatorial_belt")
    m5_spot = make_ellipsoid(rx=0.3, ry=0.18, rz=0.1, color=(35, 45, 190), name="jupiter_great_red_spot")  # BGR dark red oval
    r5.add_child(ModelNode3D("gas_body", mesh=m5_gas))
    r5.add_child(ModelNode3D("belt_node", mesh=m5_belt))
    r5.add_child(ModelNode3D("spot_node", mesh=m5_spot, position=np.array([0.95, -0.35, 0.8], dtype=np.float32)))
    models.append((Model3D(concept="Jupiter", root=r5),
                   _meta("Jupiter", ["Planet Jupiter", "Gas giant Jupiter", "fifth planet", "king of planets"],
                         "Largest planet in solar system, prominent atmospheric banding and Great Red Spot anticyclonic storm.",
                         ["Gas giant scale", "Atmospheric cloud bands", "Distinguishable Great Red Spot oval"], "planets")))

    # 6. Saturn
    r6 = ModelNode3D("root")
    m6_body = make_sphere(radius=1.1, color=(150, 205, 230), roughness=0.5, name="saturn_gas_body")
    m6_ring = make_ring_disc(r_inner=1.45, r_outer=2.45, segs=48, color=(170, 190, 215), name="saturn_ring_disc")
    r6.add_child(ModelNode3D("body_node", mesh=m6_body))
    r6.add_child(ModelNode3D("ring_node", mesh=m6_ring, rotation=np.array([0.45, 0, 0.1], dtype=np.float32)))
    models.append((Model3D(concept="Saturn", root=r6),
                   _meta("Saturn", ["Planet Saturn", "Ringed planet", "sixth planet"],
                         "Second largest planet, distinguished by its magnificent, highly reflective icy ring system.",
                         ["Gas giant body", "Prominent concentric ring disc", "Tilted orbital ring plane"], "planets")))

    # 7. Uranus
    m7 = make_sphere(radius=0.9, color=(225, 220, 175), roughness=0.4, name="uranus_methane_atmosphere")
    models.append((Model3D(concept="Uranus", root=ModelNode3D("root", mesh=m7)),
                   _meta("Uranus", ["Planet Uranus", "Ice giant Uranus", "seventh planet"],
                         "Ice giant composed of water, ammonia, and methane ices with featureless cyan-blue appearance.",
                         ["Pale cyan-aquamarine colour", "Ice giant sphere", "Methane absorption tint"], "planets")))

    # 8. Neptune
    m8 = make_sphere(radius=0.88, color=(240, 140, 35), roughness=0.4, name="neptune_atmosphere")  # BGR deep azure blue
    models.append((Model3D(concept="Neptune", root=ModelNode3D("root", mesh=m8)),
                   _meta("Neptune", ["Planet Neptune", "Eighth planet", "deep blue planet"],
                         "Outermost major planet, intense deep azure blue atmosphere shaped by supersonic winds.",
                         ["Vivid deep azure blue colour", "Ice giant spherical form", "Atmospheric haze depth"], "planets")))

    # 9. Sun
    m9 = make_sphere(radius=1.6, color=(30, 215, 255), roughness=0.2, metallic=0.0, name="solar_photosphere")
    models.append((Model3D(concept="Sun", root=ModelNode3D("root", mesh=m9)),
                   _meta("Sun", ["The Sun", "Solar star", "Sol", "our star"],
                         "G-type main-sequence star at the center of the solar system, radiating heat and light.",
                         ["Large central sphere", "Radiant golden-yellow surface", "Stellar scale"], "celestial_bodies")))

    # 10. Earth's Moon
    m10 = make_sphere(radius=0.6, color=(165, 165, 170), roughness=0.9, name="lunar_regolith")
    models.append((Model3D(concept="Moon", root=ModelNode3D("root", mesh=m10)),
                   _meta("Moon", ["Earth Moon", "The Moon", "Luna", "Earth's moon"],
                         "Earth's sole natural satellite, airless world with cratered highlands and basaltic maria.",
                         ["Light grey regolith", "Spherical natural satellite", "Cratered body representation"], "moons")))

    # 11. Europa
    m11 = make_sphere(radius=0.55, color=(210, 230, 240), roughness=0.25, name="europa_ice_crust")
    models.append((Model3D(concept="Europa", root=ModelNode3D("root", mesh=m11)),
                   _meta("Europa", ["Jupiter Moon Europa", "Moon Europa", "Jovian moon Europa"],
                         "Icy Jovian moon harboring a subsurface liquid water ocean beneath a fractured ice crust.",
                         ["Bright reflective water-ice crust", "Slight brownish fracture markings", "Spherical moon form"], "moons")))

    # 12. Titan
    m12 = make_sphere(radius=0.72, color=(50, 140, 225), roughness=0.5, name="titan_smog_atmosphere")  # BGR orange
    models.append((Model3D(concept="Titan", root=ModelNode3D("root", mesh=m12)),
                   _meta("Titan", ["Saturn Moon Titan", "Moon Titan", "Titan moon"],
                         "Saturn's largest moon, only moon with a substantial nitrogen-rich atmosphere and methane lakes.",
                         ["Dense golden-orange smog layer", "Spherical moon form", "Atmospheric obscuration"], "moons")))

    # 13. Ganymede
    m13 = make_sphere(radius=0.78, color=(125, 130, 135), roughness=0.8, name="ganymede_surface")
    models.append((Model3D(concept="Ganymede", root=ModelNode3D("root", mesh=m13)),
                   _meta("Ganymede", ["Jupiter Moon Ganymede", "Moon Ganymede", "largest moon"],
                         "Largest moon in the Solar System, larger than Mercury, grooved icy-rock terrain.",
                         ["Dark and light grooved terrain", "Large Jovian moon scale", "Rocky-ice surface"], "moons")))

    # 14. Pluto
    r14 = ModelNode3D("root")
    m14_body = make_sphere(radius=0.5, color=(110, 135, 175), roughness=0.8, name="pluto_crust")
    m14_heart = make_ellipsoid(rx=0.25, ry=0.2, rz=0.1, color=(220, 230, 240), name="tombaugh_regio")  # Bright heart plain
    r14.add_child(ModelNode3D("body_node", mesh=m14_body))
    r14.add_child(ModelNode3D("heart_node", mesh=m14_heart, position=np.array([0.35, -0.05, 0.32], dtype=np.float32)))
    models.append((Model3D(concept="Pluto", root=r14),
                   _meta("Pluto", ["Dwarf Planet Pluto", "Pluto dwarf planet", "Kuiper Belt object Pluto"],
                         "Famous dwarf planet in the Kuiper Belt with heart-shaped nitrogen glacier (Tombaugh Regio).",
                         ["Reddish-tan surface", "Bright heart-shaped nitrogen plain (Tombaugh Regio)", "Dwarf planet scale"], "dwarf_planets")))

    # 15. Asteroid
    m15 = make_ellipsoid(rx=0.8, ry=0.55, rz=0.5, lat_segs=10, lon_segs=14, color=(110, 110, 115), roughness=0.9, name="asteroid_body")
    models.append((Model3D(concept="Asteroid", root=ModelNode3D("root", mesh=m15)),
                   _meta("Asteroid", ["Space asteroid", "space rock", "Ceres asteroid", "minor planet"],
                         "Irregular rocky remnant from the early solar system, lacking sufficient gravity to form a sphere.",
                         ["Non-spherical tumbling shape", "Cratered rocky facets", "Dark stony-carbonaceous surface"], "small_bodies")))

    # 16. Comet
    r16 = ModelNode3D("root")
    m16_nucleus = make_ellipsoid(rx=0.4, ry=0.28, rz=0.25, color=(80, 85, 90), roughness=0.9, name="comet_nucleus")
    m16_coma = make_sphere(radius=0.65, color=(220, 240, 250), opacity=0.6, name="comet_coma")
    m16_tail = make_cone(radius=0.45, height=2.2, color=(210, 235, 255), name="comet_tail")
    r16.add_child(ModelNode3D("nucleus_node", mesh=m16_nucleus))
    r16.add_child(ModelNode3D("coma_node", mesh=m16_coma))
    r16.add_child(ModelNode3D("tail_node", mesh=m16_tail, position=np.array([0, -1.1, 0], dtype=np.float32)))
    models.append((Model3D(concept="Comet", root=r16),
                   _meta("Comet", ["Halley's Comet", "ice comet", "dirty snowball", "comet with tail"],
                         "Small Solar System body composed of ice, dust and rock with a glowing coma and extended dust tail.",
                         ["Dark icy nucleus", "Luminous gas coma", "Elongated dust and ion tail streamer"], "small_bodies")))

    # 17. Meteorite
    m17 = make_box(dx=0.7, dy=0.6, dz=0.5, color=(50, 50, 55), roughness=0.85, metallic=0.3, name="meteorite_specimen")
    models.append((Model3D(concept="Meteorite", root=ModelNode3D("root", mesh=m17)),
                   _meta("Meteorite", ["Fallen meteorite", "meteorite rock", "stony-iron meteorite", "space fragment"],
                         "Extraterrestrial rock specimen that survived transit through Earth's atmosphere to reach the ground.",
                         ["Dark fusion-crust exterior", "Irregular fractured facets", "Metallic sheen"], "small_bodies")))

    # 18. Black Hole
    r18 = ModelNode3D("root")
    m18_horizon = make_sphere(radius=0.6, color=(10, 10, 10), roughness=0.9, metallic=0.0, name="event_horizon")  # Pitch black
    m18_accretion = make_ring_disc(r_inner=0.85, r_outer=2.1, segs=36, color=(30, 160, 255), opacity=0.95, name="accretion_disc")  # Glowing amber/orange
    r18.add_child(ModelNode3D("horizon_node", mesh=m18_horizon))
    r18.add_child(ModelNode3D("accretion_node", mesh=m18_accretion, rotation=np.array([0.35, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Black Hole", root=r18),
                   _meta("Black Hole", ["Singularity", "Supermassive black hole", "stellar black hole", "event horizon"],
                         "Educational visualization of a black hole showing the central event horizon and glowing swirling accretion disk.",
                         ["Central black event horizon", "Glowing incandescent accretion disk", "Relativistic visual silhouette"], "astronomical_phenomena")))

    # 19. Solar System
    r19 = ModelNode3D("root")
    m19_sun = make_sphere(radius=0.45, color=(30, 210, 255), name="central_sun")
    r19.add_child(ModelNode3D("sun_node", mesh=m19_sun))
    # Add concentric orbit rings and miniature planets
    orbit_radii = [0.8, 1.2, 1.6, 2.0]
    planet_colors = [(140, 140, 140), (160, 210, 230), (210, 100, 30), (45, 85, 205)]
    for idx, (rad, col) in enumerate(zip(orbit_radii, planet_colors)):
        r_disc = make_ring_disc(r_inner=rad - 0.02, r_outer=rad + 0.02, segs=32, color=(90, 90, 90), opacity=0.4, name=f"orbit_ring_{idx}")
        p_body = make_sphere(radius=0.1, color=col, name=f"planet_body_{idx}")
        r19.add_child(ModelNode3D(f"orbit_node_{idx}", mesh=r_disc))
        r19.add_child(ModelNode3D(f"planet_node_{idx}", mesh=p_body, position=np.array([rad * math.cos(idx * 1.5), 0, rad * math.sin(idx * 1.5)], dtype=np.float32)))
    models.append((Model3D(concept="Solar System", root=r19),
                   _meta("Solar System", ["Our solar system", "heliocentric system", "planetary system"],
                         "Overview model showing the central Sun, concentric orbital tracks, and orbiting terrestrial planets.",
                         ["Central luminous Sun", "Concentric orbital boundary rings", "Multiple distinct orbiting planetary bodies"], "astronomical_systems")))

    # 20. Space Rocket
    r20 = ModelNode3D("root")
    m20_body = make_cylinder(radius_top=0.3, radius_bottom=0.3, height=2.2, color=(240, 240, 245), name="rocket_fuselage")
    m20_nose = make_cone(radius=0.3, height=0.7, color=(40, 40, 220), name="rocket_fairing")  # Red nose
    m20_fin1 = make_box(dx=0.8, dy=0.4, dz=0.04, color=(40, 40, 220), name="rocket_fin_x")
    m20_fin2 = make_box(dx=0.04, dy=0.4, dz=0.8, color=(40, 40, 220), name="rocket_fin_z")
    r20.add_child(ModelNode3D("fuselage_node", mesh=m20_body))
    r20.add_child(ModelNode3D("nose_node", mesh=m20_nose, position=np.array([0, 1.45, 0], dtype=np.float32)))
    r20.add_child(ModelNode3D("fin1_node", mesh=m20_fin1, position=np.array([0, -0.9, 0], dtype=np.float32)))
    r20.add_child(ModelNode3D("fin2_node", mesh=m20_fin2, position=np.array([0, -0.9, 0], dtype=np.float32)))
    models.append((Model3D(concept="Space Rocket", root=r20),
                   _meta("Space Rocket", ["Rocket", "Launch vehicle", "Moon rocket", "Saturn V rocket", "Falcon rocket"],
                         "Multistage space launch vehicle with cylindrical body, aerodynamic nose cone fairing, and stabilization fins.",
                         ["Cylindrical booster fuselage", "Aerodynamic nose fairing", "Cruciform stabilizing launch fins"], "space_exploration")))

    # 21. Space Shuttle
    r21 = ModelNode3D("root")
    m21_fuselage = make_cylinder(radius_top=0.25, radius_bottom=0.25, height=1.6, color=(240, 240, 240), name="shuttle_fuselage")
    m21_wings = make_box(dx=1.8, dy=0.06, dz=0.8, color=(230, 230, 230), name="shuttle_delta_wings")
    m21_tail = make_box(dx=0.06, dy=0.6, dz=0.4, color=(240, 240, 240), name="shuttle_tailfin")
    r21.add_child(ModelNode3D("fuse_node", mesh=m21_fuselage, rotation=np.array([1.57, 0, 0], dtype=np.float32)))
    r21.add_child(ModelNode3D("wing_node", mesh=m21_wings, position=np.array([0, -0.1, -0.2], dtype=np.float32)))
    r21.add_child(ModelNode3D("tail_node", mesh=m21_tail, position=np.array([0, 0.35, -0.6], dtype=np.float32)))
    models.append((Model3D(concept="Space Shuttle", root=r21),
                   _meta("Space Shuttle", ["NASA Space Shuttle", "Orbiter", "Spaceplane", "shuttle orbiter"],
                         "Reusable orbital spacecraft with delta wings, cargo payload bay, and vertical rudder tailfin.",
                         ["Aerodynamic orbiter fuselage", "Delta wing geometry", "Vertical tailfin stabilizer"], "space_exploration")))

    # 22. Satellite
    r22 = ModelNode3D("root")
    m22_bus = make_box(dx=0.5, dy=0.6, dz=0.5, color=(190, 190, 195), metallic=0.5, name="satellite_bus")
    m22_panel_l = make_box(dx=1.2, dy=0.4, dz=0.03, color=(200, 120, 40), metallic=0.7, name="solar_panel_left")  # Blue solar cells
    m22_panel_r = make_box(dx=1.2, dy=0.4, dz=0.03, color=(200, 120, 40), metallic=0.7, name="solar_panel_right")
    m22_dish = make_cone(radius=0.25, height=0.15, color=(220, 220, 220), name="antenna_dish")
    r22.add_child(ModelNode3D("bus_node", mesh=m22_bus))
    r22.add_child(ModelNode3D("panel_l_node", mesh=m22_panel_l, position=np.array([-0.95, 0, 0], dtype=np.float32)))
    r22.add_child(ModelNode3D("panel_r_node", mesh=m22_panel_r, position=np.array([0.95, 0, 0], dtype=np.float32)))
    r22.add_child(ModelNode3D("dish_node", mesh=m22_dish, position=np.array([0, 0.38, 0], dtype=np.float32), rotation=np.array([3.14, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Satellite", root=r22),
                   _meta("Satellite", ["Communications satellite", "orbital satellite", "weather satellite", "space probe"],
                         "Orbital satellite featuring instrument payload bus, dual deployable solar cell arrays, and high-gain dish antenna.",
                         ["Central instrument bus body", "Dual lateral solar panel wings", "Communication parabolic dish"], "space_exploration")))

    # 23. Space Station
    r23 = ModelNode3D("root")
    m23_core = make_cylinder(radius_top=0.25, radius_bottom=0.25, height=1.8, color=(230, 230, 235), metallic=0.3, name="station_core_module")
    m23_truss = make_box(dx=2.6, dy=0.12, dz=0.12, color=(160, 160, 165), name="integrated_truss")
    m23_array1 = make_box(dx=0.5, dy=1.2, dz=0.02, color=(190, 110, 30), name="solar_array_1")
    m23_array2 = make_box(dx=0.5, dy=1.2, dz=0.02, color=(190, 110, 30), name="solar_array_2")
    r23.add_child(ModelNode3D("core_node", mesh=m23_core, rotation=np.array([0, 0, 1.57], dtype=np.float32)))
    r23.add_child(ModelNode3D("truss_node", mesh=m23_truss, position=np.array([0, 0.4, 0], dtype=np.float32)))
    r23.add_child(ModelNode3D("array1_node", mesh=m23_array1, position=np.array([-1.1, 0.4, 0], dtype=np.float32)))
    r23.add_child(ModelNode3D("array2_node", mesh=m23_array2, position=np.array([1.1, 0.4, 0], dtype=np.float32)))
    models.append((Model3D(concept="Space Station", root=r23),
                   _meta("Space Station", ["International Space Station", "ISS", "orbital station", "space outpost"],
                         "Modular orbital research laboratory with pressurized living habitats, structural truss, and massive solar power arrays.",
                         ["Habitation module core", "Long structural support truss", "Twin high-capacity solar wing arrays"], "space_exploration")))

    # 24. Mars Rover
    r24 = ModelNode3D("root")
    m24_chassis = make_box(dx=0.8, dy=0.35, dz=0.6, color=(220, 220, 225), name="rover_chassis")
    m24_mast = make_cylinder(radius_top=0.04, radius_bottom=0.04, height=0.6, color=(160, 160, 160), name="camera_mast")
    m24_head = make_box(dx=0.2, dy=0.12, dz=0.15, color=(50, 50, 50), name="mastcam_head")
    r24.add_child(ModelNode3D("chassis_node", mesh=m24_chassis))
    r24.add_child(ModelNode3D("mast_node", mesh=m24_mast, position=np.array([0.25, 0.45, 0.2], dtype=np.float32)))
    r24.add_child(ModelNode3D("head_node", mesh=m24_head, position=np.array([0.25, 0.78, 0.2], dtype=np.float32)))
    # 4 corner wheels
    for wx, wz in ((-0.45, -0.35), (0.45, -0.35), (-0.45, 0.35), (0.45, 0.35)):
        w_mesh = make_cylinder(radius_top=0.15, radius_bottom=0.15, height=0.1, color=(40, 40, 40), name="rover_wheel")
        r24.add_child(ModelNode3D("wheel_node", mesh=w_mesh, position=np.array([wx, -0.25, wz], dtype=np.float32), rotation=np.array([0, 0, 1.57], dtype=np.float32)))
    models.append((Model3D(concept="Mars Rover", root=r24),
                   _meta("Mars Rover", ["Curiosity rover", "Perseverance rover", "robotic rover", "Mars exploration rover"],
                         "Robotic exploratory vehicle equipped with instrument chassis, high mast camera, and rugged all-terrain wheels.",
                         ["Rectangular electronics chassis", "Articulated mast camera (Mastcam)", "Rugged planetary rover wheels"], "space_exploration")))

    # 25. Space Telescope
    r25 = ModelNode3D("root")
    m25_tube = make_cylinder(radius_top=0.4, radius_bottom=0.4, height=1.6, color=(210, 210, 215), metallic=0.6, name="telescope_barrel")
    m25_shield = make_box(dx=1.8, dy=0.04, dz=1.2, color=(180, 180, 190), metallic=0.7, name="sunshield_baffle")
    m25_aperture = make_cone(radius=0.4, height=0.2, color=(30, 30, 30), name="primary_mirror_recess")
    r25.add_child(ModelNode3D("tube_node", mesh=m25_tube, rotation=np.array([1.57, 0, 0], dtype=np.float32)))
    r25.add_child(ModelNode3D("shield_node", mesh=m25_shield, position=np.array([0, -0.45, 0], dtype=np.float32)))
    r25.add_child(ModelNode3D("mirror_node", mesh=m25_aperture, position=np.array([0, 0, 0.8], dtype=np.float32), rotation=np.array([-1.57, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Space Telescope", root=r25),
                   _meta("Space Telescope", ["Hubble space telescope", "James Webb space telescope", "orbital telescope", "astronomical observatory"],
                         "Space-borne astronomical observatory with light baffle barrel, optical mirror, and protective sunshield.",
                         ["Cylindrical optical baffle barrel", "Reflective optical aperture mirror", "Broad planar thermal sunshield"], "space_exploration")))

    return models
