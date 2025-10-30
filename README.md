### 🪨 PyPetrelGPM

Python-based Geological Process Modeling Toolkit for PETREL GSLIB Data

---

📘 Overview

PyPetrelGPM is a Python-based software designed for automated import, preprocessing, and visualization of geological models exported from PETREL in GSLIB format.
It enables seamless conversion of structural and physical information from PETREL models into Python environments for further data analysis, generative AI training, and geological storage simulations.

The program supports:

- Structural and property data conversion from PETREL

- Visualization for model verification and quality control

- Missing-value correction, interpolation, and normalization

- Generation of AI-ready training datasets for reservoir or CO₂ storage modeling

---

🔬 Application Fields

- Geological modeling

- Geophysical exploration

- Reservoir engineering

- Geotechnical and subsurface resource studies
---

💡 Key Features

1️⃣ Automatic GSLIB Import and Grid Conversion

- Automatically recognizes PETREL GSLIB files and reconstructs them into 3D grid-based structures (x, y, z or i, j, k coordinates).

- Separates lithofacies proportions (Sand, Silt, Clay) into independent arrays for facies-specific analysis.

2️⃣ Visualization and Model Review

- Provides 3D visualization of lithofacies proportions in spatial coordinates.

- Supports cross-sectional views and depth-distribution plots for examining structural continuity and facies transitions.

3️⃣ Preprocessing and AI-Ready Dataset Generation

- Detects and trims grid domains containing valid data only.

- Performs missing-value interpolation and feature scaling to normalize lithofacies proportions.

- Outputs high-quality training datasets ready for use in AI-based geological modeling or CO₂ storage simulation workflows.

---

⚙️ Program Structure

PyPetrelGPM consists of three core modules, each corresponding to a key stage in the data workflow:

| Step                                      | Script                 | Description                                                                                               |
| ----------------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------- |
| **1. PETREL Data Import & Visualization** | `petrel_to_numpy.py`   | Converts GSLIB text files into 3D grid structures and visualizes facies distributions (Sand, Silt, Clay). |
| **2. Preprocessing**                      | `gpm_preprocessing.py` | Handles missing values, performs interpolation and scaling, and prepares datasets for AI training.        |
| **3. Visualization of Processed Data**    | `gpm_viz.py`           | Produces 3D visualizations and cross-sectional plots of preprocessed lithofacies data.                    |

---

🧩 Usage

All scripts accept configuration files in YAML (.yml) format for parameter control.

▶️ 1. PETREL to Numpy Conversion


$ python software/petrel_to_numpy.py params/params_petrel.yml


- Loads PETREL-exported GSLIB files

- Converts them to (x, y, z / i, j, k) grid structures

- Visualizes lithofacies proportions (Sand, Silt, Clay)

▶️ 2. Preprocessing and Normalization

$ python software/gpm_preprocessing.py params/params_preproc.yml


- Performs missing-value detection, interpolation, and scaling

- Generates normalized data for machine learning or simulation input

▶️ 3. Visualization of Preprocessed Results

$ python software/gpm_viz.py sections params/params_viz_sections.yml
$ python software/gpm_viz.py geometry3d params/params_viz_geometry3d.yml

---
🧑‍💻 Author

Eun-sil Park
M.S. Candidate, Energy Resources Engineering, Inha University
Advisor: Prof. Hong-geun Jo