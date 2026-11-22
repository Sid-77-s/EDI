# Magic Mirror — Expanded 3D Asset Library

## Overview

This library provides an expanded, fully validated, local-first 3D asset collection for **Magic Mirror**,
a real-time multimodal spatial visualization system. Every asset satisfies the strict core principles of
identity correctness, visual recognizability, educational distinguishability, standard GLB v2 compatibility,
zero-paid-API local execution, and permissive open licensing (**CC0 1.0 Universal**).

- **Total 3D Assets:** 220 models
- **Categories:** 11 distinct domains
- **Total Polygons:** 442,404 triangles (optimized for ultra-low latency & >= 30 FPS rendering)
- **Total Library Size:** 11.21 MB
- **License:** CC0-1.0 Universal (Public Domain Dedication)

## Category Summary

| Category | Slug | Model Count | Key Educational Distinctions |
|---|---|---|---|
| **Astronomy and Space** | `astronomy/` | 25 | Strict physical/biological/mechanical differentiation |
| **Chemistry and the Periodic Table** | `chemistry/` | 30 | Strict physical/biological/mechanical differentiation |
| **Physics and Mathematics** | `physics/` | 20 | Strict physical/biological/mechanical differentiation |
| **Biology and Human Anatomy** | `biology/` | 25 | Strict physical/biological/mechanical differentiation |
| **Animals and Wildlife** | `animals/` | 20 | Strict physical/biological/mechanical differentiation |
| **Sports Equipment and Balls** | `sports/` | 15 | Strict physical/biological/mechanical differentiation |
| **Vehicles and Transportation** | `vehicles/` | 20 | Strict physical/biological/mechanical differentiation |
| **Engineering, Machines and Technology** | `engineering/` | 15 | Strict physical/biological/mechanical differentiation |
| **Everyday Objects and Furniture** | `everyday/` | 25 | Strict physical/biological/mechanical differentiation |
| **Geography, Earth Science and Environment** | `geography/` | 15 | Strict physical/biological/mechanical differentiation |
| **History, Architecture and Culture** | `history/` | 10 | Strict physical/biological/mechanical differentiation |

## Technical Architecture & Importer Compatibility

1. **Format:** Binary glTF 2.0 (`.glb`) compliant with Khronos Group specifications and Magic Mirror's native `load_glb()` parser (`integration/generation/model3d.py`).
2. **Chunk Layout:** 12-byte header (`0x46546C67`, version 2), structured JSON chunk (`0x4E4F534A`), binary buffer chunk (`0x004E4942`), and 4-byte alignment padding.
3. **Zero External Dependencies:** Every asset renders entirely locally in offline mode without requiring network connectivity, external texture URLs, or paid APIs.
4. **Sidecar Metadata:** Every `<model>.glb` is accompanied by an authoritative `<model>.json` sidecar detailing canonical name, search aliases, category, subcategory, distinguishing features, educational purpose, and CC0 attribution.

## Model Catalogue Index

| Name | Category | File | Triangles | Aliases | Distinguishing Features |
|---|---|---|---|---|---|
| Mercury | Astronomy and Space | `astronomy/mercury.glb` | 720 | Planet Mercury, Mercury planet, first planet | Cratered surface; Terrestrial rocky sphere |
| Venus | Astronomy and Space | `astronomy/venus.glb` | 720 | Planet Venus, Venus planet, morning star | Dense cloud atmosphere; Warm yellowish-cream hue |
| Earth | Astronomy and Space | `astronomy/earth.glb` | 2160 | Planet Earth, The Earth, Blue planet | Blue ocean base; Green continental landmasses |
| Mars | Astronomy and Space | `astronomy/mars.glb` | 1440 | Planet Mars, Red planet, fourth planet | Rusty red-orange surface; White northern polar cap |
| Jupiter | Astronomy and Space | `astronomy/jupiter.glb` | 1512 | Planet Jupiter, Gas giant Jupiter, fifth planet | Gas giant scale; Atmospheric cloud bands |
| Saturn | Astronomy and Space | `astronomy/saturn.glb` | 912 | Planet Saturn, Ringed planet, sixth planet | Gas giant body; Prominent concentric ring disc |
| Uranus | Astronomy and Space | `astronomy/uranus.glb` | 720 | Planet Uranus, Ice giant Uranus, seventh planet | Pale cyan-aquamarine colour; Ice giant sphere |
| Neptune | Astronomy and Space | `astronomy/neptune.glb` | 720 | Planet Neptune, Eighth planet, deep blue planet | Vivid deep azure blue colour; Ice giant spherical form |
| Sun | Astronomy and Space | `astronomy/sun.glb` | 720 | The Sun, Solar star, Sol | Large central sphere; Radiant golden-yellow surface |
| Moon | Astronomy and Space | `astronomy/moon.glb` | 720 | Earth Moon, The Moon, Luna | Light grey regolith; Spherical natural satellite |
| Europa | Astronomy and Space | `astronomy/europa.glb` | 720 | Jupiter Moon Europa, Moon Europa, Jovian moon Europa | Bright reflective water-ice crust; Slight brownish fracture markings |
| Titan | Astronomy and Space | `astronomy/titan.glb` | 720 | Saturn Moon Titan, Moon Titan, Titan moon | Dense golden-orange smog layer; Spherical moon form |
| Ganymede | Astronomy and Space | `astronomy/ganymede.glb` | 720 | Jupiter Moon Ganymede, Moon Ganymede, largest moon | Dark and light grooved terrain; Large Jovian moon scale |
| Pluto | Astronomy and Space | `astronomy/pluto.glb` | 1440 | Dwarf Planet Pluto, Pluto dwarf planet, Kuiper Belt object Pluto | Reddish-tan surface; Bright heart-shaped nitrogen plain (Tombaugh Regio) |
| Asteroid | Astronomy and Space | `astronomy/asteroid.glb` | 252 | Space asteroid, space rock, Ceres asteroid | Non-spherical tumbling shape; Cratered rocky facets |
| Comet | Astronomy and Space | `astronomy/comet.glb` | 1512 | Halley's Comet, ice comet, dirty snowball | Dark icy nucleus; Luminous gas coma |
| Meteorite | Astronomy and Space | `astronomy/meteorite.glb` | 12 | Fallen meteorite, meteorite rock, stony-iron meteorite | Dark fusion-crust exterior; Irregular fractured facets |
| Black Hole | Astronomy and Space | `astronomy/black_hole.glb` | 864 | Singularity, Supermassive black hole, stellar black hole | Central black event horizon; Glowing incandescent accretion disk |
| Solar System | Astronomy and Space | `astronomy/solar_system.glb` | 4112 | Our solar system, heliocentric system, planetary system | Central luminous Sun; Concentric orbital boundary rings |
| Space Rocket | Astronomy and Space | `astronomy/space_rocket.glb` | 168 | Rocket, Launch vehicle, Moon rocket | Cylindrical booster fuselage; Aerodynamic nose fairing |
| Space Shuttle | Astronomy and Space | `astronomy/space_shuttle.glb` | 96 | NASA Space Shuttle, Orbiter, Spaceplane | Aerodynamic orbiter fuselage; Delta wing geometry |
| Satellite | Astronomy and Space | `astronomy/satellite.glb` | 108 | Communications satellite, orbital satellite, weather satellite | Central instrument bus body; Dual lateral solar panel wings |
| Space Station | Astronomy and Space | `astronomy/space_station.glb` | 108 | International Space Station, ISS, orbital station | Habitation module core; Long structural support truss |
| Mars Rover | Astronomy and Space | `astronomy/mars_rover.glb` | 384 | Curiosity rover, Perseverance rover, robotic rover | Rectangular electronics chassis; Articulated mast camera (Mastcam) |
| Space Telescope | Astronomy and Space | `astronomy/space_telescope.glb` | 156 | Hubble space telescope, James Webb space telescope, orbital telescope | Cylindrical optical baffle barrel; Reflective optical aperture mirror |
| Hydrogen Atom | Chemistry and the Periodic Table | `chemistry/hydrogen_atom.glb` | 1584 | Hydrogen, H atom, Bohr hydrogen model | Central proton nucleus; Single K-shell orbit |
| Helium Atom | Chemistry and the Periodic Table | `chemistry/helium_atom.glb` | 2304 | Helium, He atom, Bohr helium model | Two-proton nucleus; Stable filled K-shell |
| Lithium Atom | Chemistry and the Periodic Table | `chemistry/lithium_atom.glb` | 3168 | Lithium, Li atom, alkali metal lithium | Nucleus with 3 protons; Inner filled K-shell (2 e-) |
| Carbon Atom | Chemistry and the Periodic Table | `chemistry/carbon_atom.glb` | 5328 | Carbon, C atom, Bohr carbon model | Carbon nucleus; Inner shell with 2 electrons |
| Nitrogen Atom | Chemistry and the Periodic Table | `chemistry/nitrogen_atom.glb` | 6048 | Nitrogen, N atom, Bohr nitrogen model | Seven-proton nucleus; Inner shell with 2 electrons |
| Oxygen Atom | Chemistry and the Periodic Table | `chemistry/oxygen_atom.glb` | 6768 | Oxygen, O atom, atomic oxygen | Eight-proton nucleus; Inner shell with 2 electrons |
| Sodium Atom | Chemistry and the Periodic Table | `chemistry/sodium_atom.glb` | 9072 | Sodium, Na atom, alkali metal sodium | Three concentric electron shells; Filled K (2) and L (8) shells |
| Chlorine Atom | Chemistry and the Periodic Table | `chemistry/chlorine_atom.glb` | 13392 | Chlorine, Cl atom, halogen chlorine | Three electron shells; Seven valence electrons in M shell |
| Iron Atom | Chemistry and the Periodic Table | `chemistry/iron_atom.glb` | 20016 | Iron, Fe atom, transition metal iron | Dense transition metal nucleus; Four distinct electron shells |
| Gold Atom | Chemistry and the Periodic Table | `chemistry/gold_atom.glb` | 35280 | Gold, Au atom, aurum | Heavy mass nucleus; Lustrous golden core |
| Copper Atom | Chemistry and the Periodic Table | `chemistry/copper_atom.glb` | 22176 | Copper, Cu atom, cuprum | Reddish-bronze nucleus; Four electron shells |
| Silicon Atom | Chemistry and the Periodic Table | `chemistry/silicon_atom.glb` | 11232 | Silicon, Si atom, semiconductor silicon | Four valence electrons; Three concentric shells |
| Sulfur Atom | Chemistry and the Periodic Table | `chemistry/sulfur_atom.glb` | 12672 | Sulfur, S atom, brimstone | Bright yellow elemental nucleus; Six valence electrons |
| Uranium Atom | Chemistry and the Periodic Table | `chemistry/uranium_atom.glb` | 42624 | Uranium, U atom, actinide uranium | Massive actinide nucleus; Six electron shell levels |
| Water Molecule | Chemistry and the Periodic Table | `chemistry/water_molecule.glb` | 2304 | Water, H2O, H2O molecule | Central red oxygen atom; Two white hydrogen atoms |
| Carbon Dioxide | Chemistry and the Periodic Table | `chemistry/carbon_dioxide.glb` | 2304 | CO2, CO2 molecule, carbon dioxide gas | Central black carbon atom; Two opposing red oxygen atoms |
| Oxygen Molecule | Chemistry and the Periodic Table | `chemistry/oxygen_molecule.glb` | 1512 | O2, O2 molecule, dioxygen | Two identical red oxygen atoms; Covalent double-bond bridge |
| Nitrogen Molecule | Chemistry and the Periodic Table | `chemistry/nitrogen_molecule.glb` | 1512 | N2, N2 molecule, dinitrogen | Two blue nitrogen atoms; Covalent triple bond cylinder |
| Hydrogen Molecule | Chemistry and the Periodic Table | `chemistry/hydrogen_molecule.glb` | 1512 | H2, H2 molecule, dihydrogen | Two white hydrogen atoms; Single covalent bond |
| Methane | Chemistry and the Periodic Table | `chemistry/methane.glb` | 3888 | CH4, CH4 molecule, natural gas methane | Central black carbon atom; Four white hydrogen ligands |
| Ammonia | Chemistry and the Periodic Table | `chemistry/ammonia.glb` | 3096 | NH3, NH3 molecule, ammonia chemical structure | Central blue nitrogen atom; Three white hydrogen base atoms |
| Sodium Chloride | Chemistry and the Periodic Table | `chemistry/sodium_chloride.glb` | 5760 | NaCl, table salt, salt crystal | Cubic crystal arrangement; Purple sodium cations (Na+) |
| Glucose | Chemistry and the Periodic Table | `chemistry/glucose.glb` | 4320 | C6H12O6, blood sugar, dextrose | Six-membered pyranose chair ring; Red ring oxygen heteroatom |
| Ethanol | Chemistry and the Periodic Table | `chemistry/ethanol.glb` | 2880 | C2H5OH, ethyl alcohol, grain alcohol | Two-carbon alkane chain; Red hydroxyl oxygen atom |
| DNA Double Helix | Chemistry and the Periodic Table | `chemistry/dna_double_helix.glb` | 47232 | DNA, deoxyribonucleic acid, DNA molecule | Dual intertwined helical strands; Antiparallel right-handed spiral |
| Chemistry Beaker | Chemistry and the Periodic Table | `chemistry/chemistry_beaker.glb` | 144 | Beaker, glass beaker, laboratory beaker | Cylindrical transparent glass vessel; Flat stable base |
| Erlenmeyer Flask | Chemistry and the Periodic Table | `chemistry/erlenmeyer_flask.glb` | 144 | Conical flask, titration flask, science flask | Conical glassware body; Narrow cylindrical neck |
| Test Tube | Chemistry and the Periodic Table | `chemistry/test_tube.glb` | 864 | Laboratory test tube, glass test tube, culture tube | Cylindrical glass body; Rounded hemispherical bottom |
| Bunsen Burner | Chemistry and the Periodic Table | `chemistry/bunsen_burner.glb` | 288 | Lab burner, gas burner, laboratory flame | Heavy cast iron base; Vertical metal barrel tube |
| Pipette | Chemistry and the Periodic Table | `chemistry/pipette.glb` | 864 | Eye dropper, dropper pipette, laboratory pipette | Narrow dispensing glass stem; Tapered delivery tip |
| Proton | Physics and Mathematics | `physics/proton.glb` | 744 | p+, proton particle, positive nucleon | Spherical subatomic nucleon; Red standard color-coding |
| Neutron | Physics and Mathematics | `physics/neutron.glb` | 1296 | n0, neutron particle, neutral nucleon | Spherical neutral nucleon; Grey neutral color-coding |
| Electron | Physics and Mathematics | `physics/electron.glb` | 732 | e-, beta particle, elementary lepton | Compact elementary particle radius; Electric blue color-coding |
| Atomic Nucleus | Physics and Mathematics | `physics/atomic_nucleus.glb` | 5760 | atomic core, nuclear cluster, nucleon cluster | Tightly bound nucleon cluster; Red protons and grey neutrons |
| Bar Magnet | Physics and Mathematics | `physics/bar_magnet.glb` | 24 | permanent magnet, dipole magnet, magnetic bar | Bicolor magnetic dipole; Red North pole segment |
| Horseshoe Magnet | Physics and Mathematics | `physics/horseshoe_magnet.glb` | 120 | U magnet, horseshoe dipole | Characteristic U-shape geometry; Red North and blue South tips |
| Magnetic Field | Physics and Mathematics | `physics/magnetic_field.glb` | 1740 | magnetic flux lines, dipole field lines, magnetic demonstration | Central dipole magnet; Elliptical concentric magnetic flux loops |
| Pendulum | Physics and Mathematics | `physics/pendulum.glb` | 864 | simple pendulum, harmonic oscillator, clock pendulum | Vertical support pivot; Inelastic suspension wire |
| Spring | Physics and Mathematics | `physics/spring.glb` | 944 | helical spring, coil spring, Hooke spring | Continuous helical coil winding; Uniform pitch and diameter |
| Gears | Physics and Mathematics | `physics/gears.glb` | 112 | gear pair, interlocking gears, spur gears | Driver and driven spur gear cylinders; Segmented perimeter gear teeth |
| Pulley | Physics and Mathematics | `physics/pulley.glb` | 156 | simple machine pulley, sheave, pulley system | Grooved rim wheel sheave; Central axle pin and suspension bracket |
| Lever | Physics and Mathematics | `physics/lever.glb` | 88 | simple lever, class 1 lever, fulcrum beam | Rigid horizontal lever beam; Triangular pivot fulcrum |
| Inclined Plane | Physics and Mathematics | `physics/inclined_plane.glb` | 36 | ramp, slope, simple machine plane | Tilted inclined plane surface; Sliding friction block |
| Prism | Physics and Mathematics | `physics/prism.glb` | 84 | optical prism, triangular prism, refraction prism | Triangular prism cross-section; Translucent optical glass medium |
| Wave | Physics and Mathematics | `physics/wave.glb` | 18000 | sine wave, transverse wave, harmonic wave | Sinusoidal spatial profile; Periodic crests and troughs |
| Cube | Physics and Mathematics | `physics/cube.glb` | 12 | regular hexahedron, geometric cube, 3d square | Six congruent square faces; Right angles (90 deg) throughout |
| Tetrahedron | Physics and Mathematics | `physics/tetrahedron.glb` | 4 | regular tetrahedron, triangular pyramid, 4-faced polyhedron | Four equilateral triangular faces; Four vertices and six edges |
| Octahedron | Physics and Mathematics | `physics/octahedron.glb` | 8 | regular octahedron, 8-faced polyhedron | Eight equilateral triangular faces; Six vertices and twelve edges |
| Icosahedron | Physics and Mathematics | `physics/icosahedron.glb` | 20 | regular icosahedron, 20-faced polyhedron | Twenty equilateral triangular faces; Twelve vertices with five faces meeting at each |
| Coordinate Axes | Physics and Mathematics | `physics/coordinate_axes.glb` | 432 | 3D axes, Cartesian coordinate system, XYZ axes | Three orthogonal 90-degree axes; Standard RGB (XYZ) vector color-coding |
| Human Heart | Biology and Human Anatomy | `biology/human_heart.glb` | 1368 | heart, cardiac muscle, cor | Muscular conical ventricular mass; Prominent aortic arch curve |
| Human Brain | Biology and Human Anatomy | `biology/human_brain.glb` | 2232 | brain, cerebrum, human brain model | Bilateral cerebral hemispheres with longitudinal fissure; Posterior inferior cerebellum |
| Human Skull | Biology and Human Anatomy | `biology/human_skull.glb` | 2172 | skull, cranium, cranium bone | Rounded neurocranium vault; Bilateral orbital eye sockets |
| Human Lungs | Biology and Human Anatomy | `biology/human_lungs.glb` | 1512 | lungs, respiratory system, pulmonary organs | Central vertical cartilaginous trachea; Bilateral spongeous pulmonary lobes |
| Human Eye | Biology and Human Anatomy | `biology/human_eye.glb` | 1656 | eyeball, oculus, eye model | Spherical white sclera eyeball; Anterior curved cornea and colored iris |
| Human Tooth | Biology and Human Anatomy | `biology/human_tooth.glb` | 156 | molar, tooth, incisor | White enamelled occlusal crown; Cervical line anatomical neck |
| Red Blood Cell | Biology and Human Anatomy | `biology/red_blood_cell.glb` | 1536 | erythrocyte, RBC, blood cell | Biconcave circular disc profile; Indented central dimple on both faces |
| Neuron | Biology and Human Anatomy | `biology/neuron.glb` | 1080 | nerve cell, brain neuron, multipolar neuron | Cell soma with central nucleus; Branching radiating dendrite tree |
| Animal Cell | Biology and Human Anatomy | `biology/animal_cell.glb` | 2880 | eukaryotic animal cell, typical animal cell, cell model | Flexible spherical plasma membrane; Distinct spherical nucleus with genetic material |
| Plant Cell | Biology and Human Anatomy | `biology/plant_cell.glb` | 2892 | eukaryotic plant cell, plant cell model | Rigid rectangular cellulose cell wall; Large central fluid-filled vacuole |
| Bacterium | Biology and Human Anatomy | `biology/bacterium.glb` | 2456 | bacteria, bacillus, prokaryote | Cylindrical rod-shaped bacillus capsule; Hemispherical polar ends |
| Virus | Biology and Human Anatomy | `biology/virus.glb` | 596 | bacteriophage, phage, viral particle | Icosahedral protein capsid head; Contractile cylindrical tail sheath |
| Mitochondrion | Biology and Human Anatomy | `biology/mitochondrion.glb` | 1664 | mitochondria, cellular powerhouse, chondriosome | Oval rod-shaped outer membrane; Extensively folded inner cristae convolutions |
| Chloroplast | Biology and Human Anatomy | `biology/chloroplast.glb` | 1008 | chloroplast organelle, plastid, photosynthetic organelle | Biconvex lens outer envelope; Stacked disc-like thylakoid grana columns |
| Chromosome | Biology and Human Anatomy | `biology/chromosome.glb` | 864 | sister chromatids, human chromosome, genetic chromosome | Characteristic X-shaped sister chromatid arms; Central constricted centromere junction |
| Leaf | Biology and Human Anatomy | `biology/leaf.glb` | 792 | plant leaf, foliage, tree leaf | Flat elliptical lamina blade; Central longitudinal midrib axis |
| Flower | Biology and Human Anatomy | `biology/flower.glb` | 5112 | blossom, angiosperm flower, floral bloom | Central circular floral disc; Six symmetric radiating floral petals |
| Tree | Biology and Human Anatomy | `biology/tree.glb` | 792 | deciduous tree, forest tree, timber tree | Tapered woody trunk cylinder with bark texture; Elevated globular photosynthetic canopy crown |
| Mushroom | Biology and Human Anatomy | `biology/mushroom.glb` | 152 | toadstool, fungus, mushroom fruiting body | Conical umbrella-shaped pileus cap; Vertical fleshy stipe stalk |
| Bone | Biology and Human Anatomy | `biology/bone.glb` | 2232 | femur, thigh bone, human bone | Long cylindrical cortical diaphysis shaft; Angled proximal femoral head sphere |
| DNA Double Helix | Biology and Human Anatomy | `biology/dna_double_helix.glb` | 2752 | DNA, deoxyribonucleic acid, double helix | Antiparallel intertwined double helices; Connecting transverse base pair rungs |
| Ribosome | Biology and Human Anatomy | `biology/ribosome.glb` | 1440 | ribosomal complex, protein factory, 70S/80S ribosome | Asymmetric large ribosomal subunit; Interlocking small ribosomal subunit |
| Stomach | Biology and Human Anatomy | `biology/stomach.glb` | 720 | human stomach, gastric organ, ventriculus | C-shaped greater and lesser curvatures; Superior cardiac esophageal inlet |
| Kidney | Biology and Human Anatomy | `biology/kidney.glb` | 864 | human kidney, renal organ, ren | Characteristic reniform bean silhouette; Medial concave hilum indentation |
| Skeleton Torso | Biology and Human Anatomy | `biology/skeleton_torso.glb` | 2964 | ribcage, thoracic cage, skeleton chest | Posterior segmented vertebral column; Curved bilateral thoracic ribs |
| Dog | Animals and Wildlife | `animals/dog.glb` | 1812 | domestic dog, canine, canis familiaris | Quadruped mammalian anatomy; Elongated canine snout and jaw |
| Cat | Animals and Wildlife | `animals/cat.glb` | 1944 | domestic cat, feline, felis catus | Compact feline quadruped body; Upright pointed triangular ears |
| Horse | Animals and Wildlife | `animals/horse.glb` | 1800 | equine, stallion, equus caballus | Equine arched neck carriage; Slender athletic ungulate legs |
| Cow | Animals and Wildlife | `animals/cow.glb` | 468 | bovine, cattle, bos taurus | Heavy barrel torso with black-and-white patch markings; Bilateral cranial horns |
| Lion | Animals and Wildlife | `animals/lion.glb` | 2448 | panthera leo, king of beasts, male lion | Prominent dark masculine facial mane ruff; Tawny golden coat coloration |
| Tiger | Animals and Wildlife | `animals/tiger.glb` | 1740 | panthera tigris, bengal tiger, siberian tiger | Reddish-orange ground coloration; Transverse black camouflage stripes |
| Elephant | Animals and Wildlife | `animals/elephant.glb` | 1944 | pachyderm, loxodonta, african elephant | Prehensile cranial proboscis trunk; Bilateral curved ivory tusks |
| Giraffe | Animals and Wildlife | `animals/giraffe.glb` | 1800 | giraffa camelopardalis, tallest mammal, spotted giraffe | Extraordinarily elongated cervical neck; Tall slender stilt-like limbs |
| Zebra | Animals and Wildlife | `animals/zebra.glb` | 1740 | equus quagga, wild horse, striped zebra | Striking black and white striped patterning; Equid quadruped morphology |
| Bear | Animals and Wildlife | `animals/bear.glb` | 1740 | grizzly bear, brown bear, ursus arctos | Stocky robust muscular barrel torso; Broad round ursid head and snout |
| Penguin | Animals and Wildlife | `animals/penguin.glb` | 888 | emperor penguin, spheniscidae, antarctic penguin | Upright bipedal tuxedo countershading; Rigid paddle-like flipper wings |
| Eagle | Animals and Wildlife | `animals/eagle.glb` | 1536 | bald eagle, raptor, bird of prey | Wide outstretched aerofoil wingspan; Contrasting white cranial plumage |
| Owl | Animals and Wildlife | `animals/owl.glb` | 2232 | strigiformes, barn owl, nocturnal bird | Upright compact cylindrical posture; Prominent forward-facing binocular eyes |
| Dolphin | Animals and Wildlife | `animals/dolphin.glb` | 804 | delphinus, bottlenose dolphin, marine cetacean | Hydrodynamic fusiform marine body; Falcate curved dorsal fin |
| Shark | Animals and Wildlife | `animals/shark.glb` | 804 | selachimorpha, great white shark, predatory fish | Fusiform torpedo hydrodynamic body; Prominent sharp triangular dorsal fin |
| Whale | Animals and Wildlife | `animals/whale.glb` | 732 | blue whale, cetacean, humpback whale | Massive colossal cetacean fuselage; Wide horizontal caudal fluke blades |
| Octopus | Animals and Wildlife | `animals/octopus.glb` | 1296 | octopod, cephalopod, eight-armed mollusc | Bulbous cranial mantle sac; Eight radial flexible grasping tentacles |
| Butterfly | Animals and Wildlife | `animals/butterfly.glb` | 1512 | lepidoptera, monarch butterfly, winged insect | Slender segmented insect body; Broad paired membranous aerofoil wings |
| Bee | Animals and Wildlife | `animals/bee.glb` | 744 | honeybee, bumblebee, apis mellifera | Yellow and black striped abdomen; Translucent vibrating flight wings |
| Turtle | Animals and Wildlife | `animals/turtle.glb` | 4320 | testudines, sea turtle, tortoise | Domed protective bony carapace; Projecting cranial neck and head |
| Football | Sports Equipment and Balls | `sports/football.glb` | 840 | soccer ball, association football, match ball | Spherical leather surface with white hexagonal ground; Distinct black pentagonal patch distribution |
| Basketball | Sports Equipment and Balls | `sports/basketball.glb` | 1872 | nba ball, hoop ball, basketball sphere | Pebbled orange high-grip composite leather; Contrasting inset black channel seam ribs |
| Tennis Ball | Sports Equipment and Balls | `sports/tennis_ball.glb` | 1296 | tennis sphere, felt ball, yellow tennis ball | Optic yellow fluorescent fibrous felt cover; Continuous curved vulcanized white rubber seam |
| Cricket Ball | Sports Equipment and Balls | `sports/cricket_ball.glb` | 1296 | leather cricket ball, red ball, match cricket ball | Polished red leather hemispherical quarters; Raised equatorially stitched string seam |
| Baseball | Sports Equipment and Balls | `sports/baseball.glb` | 1872 | hardball, mlb baseball, stitched baseball | White genuine cowhide leather panels; Double-curved raised red waxed thread stitches |
| Golf Ball | Sports Equipment and Balls | `sports/golf_ball.glb` | 1872 | dimpled ball, pga golf ball, golf sphere | Aerodynamic dimpled surface indentations; Pure white ionomer/urethane protective cover |
| Table Tennis Paddle | Sports Equipment and Balls | `sports/table_tennis_paddle.glb` | 168 | ping pong paddle, table tennis bat, ping pong racket | Circular laminated wooden blade with rubber sheet; Vivid competition red rubber surface |
| Shuttlecock | Sports Equipment and Balls | `sports/shuttlecock.glb` | 784 | birdie, badminton shuttlecock, feather shuttle | Hemispherical white leather-covered cork nose; Flared aerodynamic conical feather skirt |
| Tennis Racket | Sports Equipment and Balls | `sports/tennis_racket.glb` | 816 | tennis racquet, strung racket, graphite racket | Oval strung head hoop frame; Open aerodynamic bifurcated throat |
| Cricket Bat | Sports Equipment and Balls | `sports/cricket_bat.glb` | 84 | willow bat, cricket paddle, english willow | Flat front striking face with rear spine ridge; Natural English willow wood grain finish |
| Baseball Bat | Sports Equipment and Balls | `sports/baseball_bat.glb` | 144 | wooden bat, ash bat, baseball club | Tapered cylindrical contour from thin handle to thick barrel; Rounded end cap hitting surface |
| Hockey Stick | Sports Equipment and Balls | `sports/hockey_stick.glb` | 24 | field hockey stick, ice hockey stick, hockey shaft | Long straight composite rectangular shaft; Sharp perpendicular curved lower blade |
| Dumbbell | Sports Equipment and Balls | `sports/dumbbell.glb` | 120 | hand weight, free weight, hex dumbbell | Central knurled steel hand grip bar; Twin hexagonal rubber-encased weight heads |
| Barbell | Sports Equipment and Balls | `sports/barbell.glb` | 216 | olympic barbell, weightlifting bar, weight plate bar | Long straight knurled steel Olympic bar; Large circular Olympic bumper weight plates |
| Sports Helmet | Sports Equipment and Balls | `sports/sports_helmet.glb` | 732 | protective helmet, cycling helmet, baseball helmet | Curved impact-resistant outer polycarbonate shell; Anterior sun visor / protective brim |
| Formula 1 Car | Vehicles and Transportation | `vehicles/formula_1_car.glb` | 964 | f1 car, grand prix racer, formula one | Open-wheel racing configuration with 4 exposed slick tires; Wide front and high rear downforce aerofoil wings |
| Sports Car | Vehicles and Transportation | `vehicles/sports_car.glb` | 324 | supercar, fast car, sports coupe | Low-profile aerodynamic sports coupe silhouette; Glossy automotive paintwork finish |
| Sedan | Vehicles and Transportation | `vehicles/sedan.glb` | 312 | passenger car, saloon car, family car | Three-box classic automobile profile; Elevated four-door passenger compartment |
| Truck | Vehicles and Transportation | `vehicles/truck.glb` | 456 | semi truck, lorry, cargo truck | Upright forward commercial operator cabin; Long rear heavy-duty cargo haulage bed |
| Bus | Vehicles and Transportation | `vehicles/bus.glb` | 312 | city bus, transit bus, coach | Long rectangular mass transit monocoque body; Continuous horizontal panoramic window glazing |
| Ambulance | Vehicles and Transportation | `vehicles/ambulance.glb` | 336 | emergency vehicle, paramedic van, medical transport | White emergency medical van coachwork; Roof-mounted emergency strobe lightbar |
| Police Car | Vehicles and Transportation | `vehicles/police_car.glb` | 336 | police cruiser, patrol car, cop car | High-contrast black-and-white police livery; Roof emergency dual red and blue strobe lightbar |
| Motorcycle | Vehicles and Transportation | `vehicles/motorcycle.glb` | 948 | motorbike, cycle, motor racer | Two in-line road wheels on single track; Central internal combustion engine block |
| Bicycle | Vehicles and Transportation | `vehicles/bicycle.glb` | 1308 | bike, pushbike, cycle | Tubular diamond geometric bicycle frame; Two slender spoked pneumatic rim wheels |
| Airplane | Vehicles and Transportation | `vehicles/airplane.glb` | 168 | passenger jet, airliner, commercial aircraft | Long cylindrical pressurized passenger fuselage; Wide aerofoil lift wings |
| Fighter Jet | Vehicles and Transportation | `vehicles/fighter_jet.glb` | 804 | supersonic fighter, combat aircraft, jet fighter | Sleek aerodynamic supersonic delta wings; Tinted bubble pilot cockpit canopy |
| Helicopter | Vehicles and Transportation | `vehicles/helicopter.glb` | 804 | chopper, rotary aircraft, whirlybird | Aerodynamic aerodynamic cabin fuselage; Large horizontal overhead main rotor blades |
| Drone | Vehicles and Transportation | `vehicles/drone.glb` | 204 | quadcopter, UAV, multirotor | Central flight controller avionics hub; Four radial cross arms in X-configuration |
| Hot Air Balloon | Vehicles and Transportation | `vehicles/hot_air_balloon.glb` | 732 | aerostat, hot air aerostat, thermal balloon | Large tear-shaped thermal air envelope; Suspended passenger wicker gondola basket |
| Speedboat | Vehicles and Transportation | `vehicles/speedboat.glb` | 96 | motorboat, powerboat, runabout | Sharp hydrodynamic V-hull entry bow; Tinted aerodynamic cockpit windshield |
| Sailboat | Vehicles and Transportation | `vehicles/sailboat.glb` | 804 | sailing yacht, monohull, sail vessel | Sleek marine displacement monohull; Tall vertical aluminum/carbon mast spar |
| Submarine | Vehicles and Transportation | `vehicles/submarine.glb` | 804 | naval sub, underwater vessel, submersible | Elongated cylindrical pressure hull fuselage; Elevated conning tower sail with periscope mast |
| Bullet Train | Vehicles and Transportation | `vehicles/bullet_train.glb` | 96 | shinkansen, high-speed train, maglev | Aerodynamic elongated needle-nose front taper; Smooth streamlined high-speed train carriage |
| Cargo Ship | Vehicles and Transportation | `vehicles/cargo_ship.glb` | 48 | container ship, freighter, merchant vessel | Broad heavy displacement merchant hull; Multi-tiered stern navigation bridge deck |
| Locomotive | Vehicles and Transportation | `vehicles/locomotive.glb` | 96 | train engine, diesel locomotive, railroad engine | Heavy diesel-electric hood engine body; Elevated forward engineer observation cab |
| Robotic Arm | Engineering, Machines and Technology | `engineering/robotic_arm.glb` | 240 | industrial robot, articulated arm, 6-axis robot | Articulated multi-axis robotic joints; Vivid safety industrial orange finish |
| Jet Engine | Engineering, Machines and Technology | `engineering/jet_engine.glb` | 216 | turbofan, aircraft engine, gas turbine | Aerodynamic cylindrical bypass nacelle casing; Forward rotating aerodynamic spinner cone |
| Planetary Gear | Engineering, Machines and Technology | `engineering/planetary_gear.glb` | 280 | epicyclic gear, gear train, sun and planet gear | Central drive sun gear cylinder; Three symmetrically meshed planet gears |
| CPU | Engineering, Machines and Technology | `engineering/cpu.glb` | 96 | microprocessor, central processing unit, processor chip | Square multi-layer PCB carrier substrate; Metallic nickel-plated integrated heat spreader (IHS) |
| Circuit Board | Engineering, Machines and Technology | `engineering/circuit_board.glb` | 168 | PCB, printed circuit board, motherboard | Green solder mask laminate substrate plate; Surface mount black IC chip packages |
| 3D Printer | Engineering, Machines and Technology | `engineering/3d_printer.glb` | 36 | additive manufacturing, fdm printer, rapid prototyper | Cartesian axis aluminum gantry frame; Square horizontal heated build plate |
| Wind Turbine | Engineering, Machines and Technology | `engineering/wind_turbine.glb` | 192 | wind generator, wind mill, renewable energy turbine | Tall slender tapered tubular steel tower; Overhead generator nacelle pod |
| Solar Panel | Engineering, Machines and Technology | `engineering/solar_panel.glb` | 24 | photovoltaic module, pv panel, solar module | Extruded anodized aluminum perimeter frame; Deep blue anti-reflective crystalline silicon cell matrix |
| Electric Motor | Engineering, Machines and Technology | `engineering/electric_motor.glb` | 156 | induction motor, AC motor, electromotor | Cylindrical ribbed stator housing; Protruding hardened steel drive rotor shaft |
| Hydraulic Cylinder | Engineering, Machines and Technology | `engineering/hydraulic_cylinder.glb` | 144 | hydraulic actuator, ram, pneumatic piston | Heavy seamless steel cylinder barrel tube; Extending polished hard-chrome plated piston rod |
| Water Pump | Engineering, Machines and Technology | `engineering/water_pump.glb` | 720 | centrifugal pump, fluid pump, impeller pump | Spiral hydrodynamic volute casing; Central axial suction eye inlet |
| Steam Turbine | Engineering, Machines and Technology | `engineering/steam_turbine.glb` | 144 | power turbine, rotary steam engine, turbogenerator | Heavy cast steel split casing shell; Concentric high-speed rotor shaft axis |
| Camera | Engineering, Machines and Technology | `engineering/camera.glb` | 96 | dslr, digital camera, slr camera | Ergonomic matte black magnesium chassis body; Top pentaprism optical viewfinder hump |
| Microphone | Engineering, Machines and Technology | `engineering/microphone.glb` | 792 | mic, dynamic mic, studio condenser | Cylindrical die-cast zinc handheld handle; Spherical woven steel mesh acoustic capsule grille |
| Speaker | Engineering, Machines and Technology | `engineering/speaker.glb` | 804 | loudspeaker, studio monitor, audio speaker | Solid acoustic cabinet enclosure; Large cone-shaped low frequency woofer driver |
| Chair | Everyday Objects and Furniture | `everyday/chair.glb` | 312 | wooden chair, dining chair, seat | Horizontal supportive seat cushion pan; Four vertical ground-contact legs |
| Table | Everyday Objects and Furniture | `everyday/table.glb` | 300 | dining table, wooden table, desk table | Broad flat rectangular work tabletop; Four perimeter vertical supporting legs |
| Sofa | Everyday Objects and Furniture | `everyday/sofa.glb` | 48 | couch, davenport, living room sofa | Upholstered multi-person seat bench cushion; Padded upright backrest support |
| Bed | Everyday Objects and Furniture | `everyday/bed.glb` | 36 | mattress bed, single bed, sleeping bed | Horizontal sleep mattress box; Elevated vertical wooden headboard |
| Desk | Everyday Objects and Furniture | `everyday/desk.glb` | 36 | office desk, workstation, computer desk | Spacious rectangular desktop surface; Side pedestal filing drawer unit |
| Coffee Mug | Everyday Objects and Furniture | `everyday/coffee_mug.glb` | 648 | mug, coffee cup, tea cup | Cylindrical glazed ceramic beverage vessel; Lateral curved D-shaped finger grip handle |
| Water Bottle | Everyday Objects and Furniture | `everyday/water_bottle.glb` | 216 | bottle, drink bottle, flask | Cylindrical liquid containment body; Tapered shoulder narrowing to neck |
| Plate | Everyday Objects and Furniture | `everyday/plate.glb` | 112 | dinner plate, dish, ceramic plate | Circular shallow ceramic disc; Raised concave outer rim edge |
| Spoon | Everyday Objects and Furniture | `everyday/spoon.glb` | 792 | soup spoon, eating utensil, tablespoon | Concave oval scoop bowl; Elongated flat handle shank |
| Fork | Everyday Objects and Furniture | `everyday/fork.glb` | 84 | dinner fork, table fork, eating utensil | Parallel piercing tines / prongs; Ergonomic slender handle shank |
| Knife | Everyday Objects and Furniture | `everyday/knife.glb` | 24 | table knife, dinner knife, butter knife | Straight steel cutting blade with bevel edge; Solid ergonomic hand grip scales |
| Smartphone | Everyday Objects and Furniture | `everyday/smartphone.glb` | 96 | phone, mobile phone, cell phone | Slim rectangular rounded-corner chassis slab; Full-face glossy touchscreen glass display |
| Laptop | Everyday Objects and Furniture | `everyday/laptop.glb` | 36 | notebook computer, laptop computer, macbook | Clamshell folding hinge design; Keyboard and trackpad lower base deck |
| Keyboard | Everyday Objects and Furniture | `everyday/keyboard.glb` | 24 | computer keyboard, typing keyboard, qwerty keyboard | Low-profile rectangular base deck; Grid of typewriter keycaps |
| Computer Mouse | Everyday Objects and Furniture | `everyday/computer_mouse.glb` | 792 | mouse, optical mouse, wireless mouse | Ergonomic palm-contoured dome shell; Split primary left/right clicking buttons |
| Headphones | Everyday Objects and Furniture | `everyday/headphones.glb` | 720 | headset, earphones, audio headphones | Padded flexible cranial headband arch; Twin circumaural acoustic earcups |
| Watch | Everyday Objects and Furniture | `everyday/watch.glb` | 96 | wristwatch, timepiece, chronometer | Circular metallic watchcase with dial face; Twin flexible leather wrist attachment straps |
| Backpack | Everyday Objects and Furniture | `everyday/backpack.glb` | 24 | rucksack, knapsack, bookbag | Voluminous zippered main canvas compartment; Protruding front utility storage pocket |
| Book | Everyday Objects and Furniture | `everyday/book.glb` | 24 | hardcover book, novel, textbook | Stiff protective cloth/board hardcover binding; Squared bound paper leaf edge block |
| Pen | Everyday Objects and Furniture | `everyday/pen.glb` | 156 | ballpoint pen, biro, ink pen | Slender cylindrical plastic/metal grip barrel; Tapered metallic writing point nib |
| Desk Lamp | Everyday Objects and Furniture | `everyday/desk_lamp.glb` | 216 | table lamp, task light, reading lamp | Stable weighted circular counterbase; Articulated elevating support gooseneck column |
| Fan | Everyday Objects and Furniture | `everyday/fan.glb` | 732 | desk fan, table fan, cooling fan | Circular protective wire guard grille; Internal rotating multi-fin airflow impeller |
| Clock | Everyday Objects and Furniture | `everyday/clock.glb` | 712 | wall clock, analog clock, timepiece | Circular analog dial face with contrast bezel; Coaxial rotating hour and minute indicator hands |
| Plant Pot | Everyday Objects and Furniture | `everyday/plant_pot.glb` | 216 | flowerpot, planter, indoor plant | Tapered conical terracotta pottery body; Dark interior nutrient soil bed surface |
| Umbrella | Everyday Objects and Furniture | `everyday/umbrella.glb` | 712 | parasol, rain umbrella, brolly | Domed protective waterproof fabric canopy; Central supporting metal runner shaft |
| Globe | Geography, Earth Science and Environment | `geography/globe.glb` | 1368 | terrestrial globe, world globe, earth globe | Spherical planetary representation; Standard 23.5-degree axial tilt angle |
| Mountain Peak | Geography, Earth Science and Environment | `geography/mountain_peak.glb` | 128 | mountain, alpine peak, summit | Steep conical elevated rocky summit; High-albedo perennial snow/ice cap |
| Volcano | Geography, Earth Science and Environment | `geography/volcano.glb` | 144 | active volcano, stratovolcano, volcano cone | Conical basaltic volcanic edifice; Summit caldera summit crater depression |
| Earth Layers | Geography, Earth Science and Environment | `geography/earth_layers.glb` | 2160 | internal earth structure, earth core, geosphere layers | Concentric spherical geological shells; Dense solid iron-nickel inner core |
| Coral Reef | Geography, Earth Science and Environment | `geography/coral_reef.glb` | 200 | marine reef, coral ecosystem, barrier reef | Calcium carbonate substrate bedrock; Diverse pigmented branching coral polyp structures |
| Glacier | Geography, Earth Science and Environment | `geography/glacier.glb` | 24 | valley glacier, alpine glacier, ice stream | High-density compacted blue ice tongue; Glaciated mountain valley trough |
| Tectonic Plates | Geography, Earth Science and Environment | `geography/tectonic_plates.glb` | 24 | plate tectonics, subduction zone, lithospheric plates | Thick continental lithospheric slab; Subducting dense oceanic crustal slab |
| Crystal | Geography, Earth Science and Environment | `geography/crystal.glb` | 204 | quartz crystal, mineral cluster, crystalline prism | Hexagonal columnar prismatic crystal habit; Pointed six-sided pyramidal apex terminations |
| Waterfall | Geography, Earth Science and Environment | `geography/waterfall.glb` | 96 | cascade, falls, plunge waterfall | Vertical rock cliff escarpment ledge; Cascading vertical fluid water sheet |
| Cloud | Geography, Earth Science and Environment | `geography/cloud.glb` | 2160 | cumulus, cumulus cloud, fluffy cloud | Billowing cauliflower-like globular tops; Flat horizontal condensation cloud base |
| Tornado | Geography, Earth Science and Environment | `geography/tornado.glb` | 80 | twister, funnel cloud, cyclone | Inverted narrow vortex funnel cone; Extreme rotational updraft column |
| Water Cycle | Geography, Earth Science and Environment | `geography/water_cycle.glb` | 744 | hydrologic cycle, hydrological cycle, evaporation cycle | Integrated ocean reservoir and terrestrial landscape; Atmospheric condensation cloud phase |
| Fossil | Geography, Earth Science and Environment | `geography/fossil.glb` | 944 | ammonite, fossil shell, paleontology specimen | Planispiral logarithmic shell coiling; Ribbed chambered shell ornamentation |
| Sand Dune | Geography, Earth Science and Environment | `geography/sand_dune.glb` | 64 | dune, desert dune, barchan dune | Asymmetric windward slope and steep slip face; Golden desert aeolian sand coloration |
| Canyon | Geography, Earth Science and Environment | `geography/canyon.glb` | 36 | gorge, river canyon, ravine | Steep stratified sedimentary canyon walls; Deep erosional incision gorge profile |
| Pyramid | History, Architecture and Culture | `history/pyramid.glb` | 32 | great pyramid, giza pyramid, egyptian pyramid | Square monumental stone masonry foundation; Four converging equilateral triangular sloping faces |
| Column | History, Architecture and Culture | `history/column.glb` | 104 | greek column, corinthian column, classical pillar | Fluted vertical cylindrical marble shaft; Classical architectural proportion with entasis taper |
| Castle Tower | History, Architecture and Culture | `history/castle_tower.glb` | 136 | keep, castle turret, medieval tower | Heavy cylindrical stone ashlar masonry core; Projecting corbelled machicolation crown |
| Colosseum | History, Architecture and Culture | `history/colosseum.glb` | 184 | roman amphitheatre, amphitheatrum flavium, flavian amphitheatre | Elliptical multi-tiered arcade monument; Central gladiatorial arena combat floor |
| Obelisk | History, Architecture and Culture | `history/obelisk.glb` | 32 | monolithic obelisk, egyptian obelisk, stone monument | Monolithic four-sided tapering granite pillar; Gilded summit pyramidion apex |
| Amphora | History, Architecture and Culture | `history/amphora.glb` | 1944 | greek amphora, ancient vase, terracotta vessel | Ovoid tapered terracotta ceramic body; Two vertical curved arching handles |
| Stone Monument | History, Architecture and Culture | `history/stone_monument.glb` | 36 | stonehenge, trilithon, megalith | Two vertical upright megalithic stone pillars; Horizontal spanning lintel capstone beam |
| Handaxe | History, Architecture and Culture | `history/handaxe.glb` | 32 | biface, paleolithic tool, flint tool | Bifacial flaked teardrop silhouette; Sharp worked lateral cutting perimeter edges |
| Astrolabe | History, Architecture and Culture | `history/astrolabe.glb` | 180 | planispheric astrolabe, navigational instrument, brass astrolabe | Circular brass graduated mater disc rim; Rotating openwork star-pointer rete disc |
| Arch | History, Architecture and Culture | `history/arch.glb` | 612 | roman arch, triumphal arch, stone archway | Two vertical supporting stone masonry piers; Semicircular voussoir curved arch vault |

## Attribution & Licensing

All assets in this starter pack are released under **Creative Commons Zero 1.0 Universal (CC0-1.0)**.
They are free for educational, commercial, and research use without restrictions.

**Author / Curator:** Magic Mirror Spatial AI Team  
**Release Date:** 2026-11-22  
**Repository:** [Magic Mirror NLP-Visualisation](https://github.com/Yash-P29/Magic_Mirror_NLP-Visualisation)  

## Continuous Discovery and Expansion Backlog (Section 16)

In accordance with Section 16 of the Contribution Guide, the starter pack forms a launchpad for continuous discovery across all fields of knowledge. See [DISCOVERY_BACKLOG.md](file:///d:/Sem_5/EDI/assets/starter_pack/DISCOVERY_BACKLOG.md) for the active expansion backlog detailing proposed canonical names, technical difficulty, educational purpose, and priorities across quantum physics, medical biomechanics, AI silicon hardware, and mechanical engineering.
