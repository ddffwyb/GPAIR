# GPAIR: Gaussian-Kernel-Based Ultrafast 3D Photoacoustic Iterative Reconstruction

<p align="center">
  <strong>Yibing Wang<sup>1</sup>, Shuang Li<sup>1</sup>, Tingting Huang<sup>1</sup>, Yu Zhang<sup>1</sup>, Chulhong Kim<sup>2-6</sup>, Seongwook Choi<sup>2-6</sup>, Changhui Li<sup>1,7,*</sup></strong>
</p>

<p align="center">
  <sup>1</sup>Department of Biomedical Engineering, College of Future Technology, Peking University, Beijing, China<br>
  <sup>2</sup>Department of Electrical Engineering, Pohang University of Science and Technology, Pohang, Republic of Korea<br>
  <sup>3</sup>Department of Convergence IT Engineering, POSTECH, Pohang, Republic of Korea<br>
  <sup>4</sup>Department of Mechanical Engineering, POSTECH, Pohang, Republic of Korea<br>
  <sup>5</sup>Department of Medical Science and Engineering, POSTECH, Pohang, Republic of Korea<br>
  <sup>6</sup>Medical Device Innovation Center, POSTECH, Pohang, Republic of Korea<br>
  <sup>7</sup>National Biomedical Imaging Center, Peking University, Beijing, China<br>
  <sup>*</sup>Corresponding author: <a href="mailto:chli@pku.edu.cn">chli@pku.edu.cn</a>
</p>

---

## Overview

This repository contains the official source code for the paper:

> **GPAIR: Gaussian-Kernel-Based Ultrafast 3D Photoacoustic Iterative Reconstruction**

GPAIR is an ultrafast iterative reconstruction (IR) framework for three-dimensional photoacoustic computed tomography (3D PACT). By replacing traditional discrete voxel representations with continuous isotropic Gaussian kernels and deriving closed-form analytical solutions for the forward and adjoint operators, GPAIR achieves **orders-of-magnitude acceleration** (up to **872×** speedup) compared to existing IR methods while delivering **superior reconstruction fidelity**.

### Key Highlights

- **Sub-second 3D reconstruction**: Achieves **< 0.82 s** reconstruction for in vivo datasets with **8.4 million voxels** and 1024 detectors, enabling near-real-time large-scale 3D PA imaging.
- **Continuous-to-discrete paradigm**: Replaces the traditional "discrete-to-discrete" voxel model with continuous Gaussian-kernel-based source representation, eliminating discretization artifacts.
- **Closed-form analytical solution**: Derives an exact analytical expression for the PA pressure wave from Gaussian sources, bypassing computationally expensive numerical time-stepping.
- **GPU-native Triton kernels**: Implements high-performance forward and adjoint operators using Triton GPU kernels with Split-K parallelization, SoA memory layout, and hardware-accelerated atomic operations.
- **Adaptive Supersampling Alignment (ASSA)**: Dynamically upsamples the temporal grid to suppress discretization misalignment errors while maintaining computational efficiency.
- **Vessel Continuity Regularization (VCR)**: Combines Hessian-based curvature priors with Total Variation (TV) regularization to preserve vascular connectivity and suppress artifacts.
- **Nonnegative Parameterization Constraint (NPC)**: Ensures physically meaningful non-negative pressure reconstruction via smooth reparameterization.

---

## Method

<p align="center">
  <em>Schematic overview of the GPAIR framework</em>
</p>

GPAIR integrates three coupled components:

1. **Gaussian-Kernel-Based Discretization (GKD)**: Models the initial pressure field as a superposition of continuous isotropic Gaussian kernels, ensuring spatial differentiability and eliminating voxel boundary artifacts.

2. **Differentiable Physics Modeling with ASSA**: Constructs a fully differentiable forward operator with a closed-form analytical PA wave expression. The ASSA strategy adaptively upsamples the temporal grid to reconcile continuous time-of-flight with discrete sampling.

3. **Constrained Optimization with VCR**: Formulates reconstruction as gradient-based iterative optimization with the Adam optimizer and cosine annealing warm restarts learning rate scheduling. NPC ensures non-negativity, and VCR enforces vessel morphological priors.

The standalone computational modules — `SignalSimulator` (forward operator) and `ImageReconstructor` (adjoint operator) — are also provided as reusable high-performance tools.

---

## Repository Structure

```
GPAIR/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── signal_simulator.py                # Forward operator (A) - Triton-based signal simulation module
├── image_reconstructor.py             # Adjoint operator (A^T) - Triton-based backprojection module
├── data/                              # Experimental data
│   ├── sensor_Vessel_data_matrix_64.mat       # Simulated vessel signal data (64 detectors)
│   ├── sensor_Vessel_data_matrix_256.mat      # Simulated vessel signal data (256 detectors)
│   ├── sensor_Vessel_data_matrix_1024.mat     # Simulated vessel signal data (1024 detectors)
│   ├── sensor_Vessel_location_64.mat          # Planar array detector locations (64 detectors)
│   ├── sensor_Vessel_location_256.mat         # Planar array detector locations (256 detectors)
│   ├── sensor_Vessel_location_1024.mat        # Planar array detector locations (1024 detectors)
│   ├── sensor_Sphere_data_matrix_1024.mat     # Simulated vessel signal data (hemispherical, 1024 detectors)
│   ├── sensor_Sphere_location_1024.mat        # Hemispherical array detector locations (1024 detectors)
│   ├── sensor_Liver_data_matrix.mat           # In vivo rat liver signal data (1024 detectors)
│   ├── sensor_Liver_location.mat              # Hemispherical array detector locations for liver
│   ├── sphere_integral_table.mat              # Precomputed sphere integral lookup table (for MB-PD)
│   └── sphere_integral_gradd_table.mat        # Precomputed sphere integral gradient lookup table (for MB-PD)
├── gpair_v7_Vessel_64.ipynb           # GPAIR reconstruction - Planar vessel, 64 detectors
├── gpair_v7_Vessel_256.ipynb          # GPAIR reconstruction - Planar vessel, 256 detectors
├── gpair_v7_Vessel_1024.ipynb         # GPAIR reconstruction - Planar vessel, 1024 detectors
├── gpair_v7_Sphere_64.ipynb           # GPAIR reconstruction - Hemispherical vessel, 64 detectors
├── gpair_v7_Sphere_256.ipynb          # GPAIR reconstruction - Hemispherical vessel, 256 detectors
├── gpair_v7_Sphere_1024.ipynb         # GPAIR reconstruction - Hemispherical vessel, 1024 detectors
├── gpair_v7_Liver.ipynb               # GPAIR reconstruction - In vivo rat liver
└── MB-PD/                             # Baseline comparison: Model-Based Point-Detector algorithm
    ├── mbpd_Vessel_64.ipynb           # MB-PD reconstruction - Planar vessel, 64 detectors
    ├── mbpd_Vessel_256.ipynb          # MB-PD reconstruction - Planar vessel, 256 detectors
    ├── mbpd_Vessel_1024.ipynb         # MB-PD reconstruction - Planar vessel, 1024 detectors
    ├── mbpd_Sphere_64.ipynb           # MB-PD reconstruction - Hemispherical vessel, 64 detectors
    ├── mbpd_Sphere_256.ipynb          # MB-PD reconstruction - Hemispherical vessel, 256 detectors
    ├── mbpd_Sphere_1024.ipynb         # MB-PD reconstruction - Hemispherical vessel, 1024 detectors
    ├── mbpd_Liver.ipynb               # MB-PD reconstruction - In vivo rat liver
    └── sphere_integration.ipynb       # Sphere integration computation for MB-PD
```

---

## Requirements

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (tested on NVIDIA RTX 5090)
- **GPU Memory**: ≥ 8 GB recommended (depends on reconstruction grid size)

### Software

- Python ≥ 3.9
- PyTorch ≥ 2.0 (with CUDA support)
- Triton ≥ 3.0
- GAPAT
- NumPy
- SciPy
- Matplotlib

### Installation

1. Clone this repository:

```bash
git clone https://github.com/ddffwyb/GPAIR.git
cd GPAIR
```

2. Install dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install triton numpy scipy matplotlib gapat
```

> **Note**: Please ensure your PyTorch installation matches your CUDA version. Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for detailed installation instructions.
>
> **Note**: GAPAT is our open-source photoacoustic toolbox for visualization and post-processing. GitHub: [https://github.com/ddffwyb/GAPAT](https://github.com/ddffwyb/GAPAT) | PyPI: [https://pypi.org/project/gapat/](https://pypi.org/project/gapat/)

---

## Quick Start

### Running GPAIR Reconstruction

Each experiment is provided as a self-contained Jupyter notebook. To reproduce the results:

```bash
jupyter notebook gpair_v7_Liver.ipynb
```

The notebooks follow a unified workflow:

1. **Data loading**: Load detector signal data and detector locations from `.mat` files.
2. **Parameter configuration**: Set reconstruction grid dimensions, speed of sound, sampling frequency, etc.
3. **Kernel generation**: Compute the adaptive convolution kernel and upsampling parameters (ASSA).
4. **Forward & Adjoint operators**: Define Triton-accelerated GPU kernels for the forward operator $\mathcal{A}$ and adjoint operator $\mathcal{A}^T$.
5. **Iterative reconstruction**: Run gradient-based optimization with Adam optimizer, NPC, VCR, and cosine annealing warm restarts.
6. **Visualization**: Display reconstruction results as maximum amplitude projections (MAPs) and slice views.

### Using Standalone Modules

The forward and adjoint operators are also packaged as reusable Python modules:

#### Signal Simulator (Forward Operator)

```python
from signal_simulator import SignalSimulator

simulator = SignalSimulator(
    x_range=[-12.80e-3, 12.80e-3],
    y_range=[-12.80e-3, 12.80e-3],
    z_range=[-6.40e-3, 6.40e-3],
    res=0.10e-3,           # Spatial resolution (m)
    vs=1510.0,             # Speed of sound (m/s)
    fs=8.333333e6,         # Sampling frequency (Hz)
    detector_locations=detector_locs,  # [num_detectors, 3]
    num_times=512,
)

# voxel_data: [num_x, num_y, num_z] numpy/torch array
simulated_signal = simulator.simulate(voxel_data)
# Returns: [num_detectors, num_times]
```

#### Image Reconstructor (Adjoint Operator)

```python
from image_reconstructor import ImageReconstructor

reconstructor = ImageReconstructor(
    x_range=[-12.80e-3, 12.80e-3],
    y_range=[-12.80e-3, 12.80e-3],
    z_range=[-6.40e-3, 6.40e-3],
    res=0.10e-3,
    vs=1510.0,
    fs=8.333333e6,
    detector_locations=detector_locs,  # [num_detectors, 3]
    num_times=512,
)

# signal_data: [num_detectors, num_times] numpy/torch array
reconstructed_image = reconstructor.reconstruct(signal_data)
# Returns: [num_x, num_y, num_z]
```

---

## Experiments

### Simulation Experiments

Synthetic 3D vascular network (512 × 512 × 256 voxels) with PA signals generated by k-Wave:

| System        | Detectors | Algorithm | 3D PSNR (dB) |  3D SSIM   | Time (s)  | Speedup  |
| ------------- | --------- | --------- | :----------: | :--------: | :-------: | :------: |
| Planar        | 64        | MB-PD     |    26.27     |   0.5229   |  1207.8   |    —     |
| Planar        | 64        | **GPAIR** |  **36.49**   | **0.9932** | **3.92**  | **308×** |
| Planar        | 256       | MB-PD     |    29.53     |   0.5670   |  4535.8   |    —     |
| Planar        | 256       | **GPAIR** |  **38.71**   | **0.9970** | **5.52**  | **822×** |
| Planar        | 1024      | MB-PD     |    33.93     |   0.7238   |  17301.2  |    —     |
| Planar        | 1024      | **GPAIR** |  **39.32**   | **0.9985** | **19.85** | **872×** |
| Hemispherical | 64        | MB-PD     |    26.85     |   0.5477   |   985.0   |    —     |
| Hemispherical | 64        | **GPAIR** |  **36.96**   | **0.9945** | **4.46**  | **221×** |
| Hemispherical | 256       | MB-PD     |    30.19     |   0.6012   |  3689.6   |    —     |
| Hemispherical | 256       | **GPAIR** |  **38.43**   | **0.9982** | **6.44**  | **573×** |
| Hemispherical | 1024      | MB-PD     |    34.57     |   0.7505   |  13907.1  |    —     |
| Hemispherical | 1024      | **GPAIR** |  **38.98**   | **0.9987** | **23.57** | **590×** |

### In Vivo Experiments

In vivo data (256 × 256 × 128 voxels, ~8.4 million voxels) acquired using a 1024-element hemispherical array:

| Sample      | Algorithm |    CNR     | Time (s) | Speedup  |
| ----------- | --------- | :--------: | :------: | :------: |
| Mouse Brain | MB-PD     |   68.47    |  237.2   |    —     |
| Mouse Brain | **GPAIR** | **220.61** | **0.82** | **289×** |
| Rat Kidney  | MB-PD     |   46.87    |  238.9   |    —     |
| Rat Kidney  | **GPAIR** | **871.08** | **0.80** | **299×** |
| Rat Liver   | MB-PD     |   85.25    |  251.2   |    —     |
| Rat Liver   | **GPAIR** | **203.30** | **0.79** | **318×** |

> All experiments were performed on an NVIDIA RTX 5090 GPU.

---

## Data Availability

The data supporting the findings of this study are provided within the article.

**Simulated vascular data**: The raw data for the simulated vessel experiments have been archived and are available in this repository at `data/` (including planar and hemispherical array configurations with 64, 256, and 1024 detectors).

**In vivo animal experimental data**: The in vivo animal experimental data generated in Professor Chulhong Kim's laboratory are subject to access restrictions due to confidentiality considerations (only the rat liver dataset is provided in this repository). The use of the provided liver data also requires permission from Professor Kim. Researchers interested in obtaining these data may apply for access under a data sharing agreement by contacting Professor Kim (email: [chulhong@postech.edu](mailto:chulhong@postech.edu)).

---

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{wang2025gpair,
  title={GPAIR: Gaussian-Kernel-Based Ultrafast 3D Photoacoustic Iterative Reconstruction},
  author={Wang, Yibing and Li, Shuang and Huang, Tingting and Zhang, Yu and Kim, Chulhong and Choi, Seongwook and Li, Changhui},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Changhui Li acknowledges support from the National Key R&D Program of China (2023YFC2411700, 2017YFE0104200) and the Beijing Natural Science Foundation (7232177).
- Chulhong Kim acknowledges support from the Basic Science Research Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Education (2020R1A6A1A03047902).
- This work is supported by the Biomedical Computing Platform of the National Biomedical Imaging Center, Peking University.

---

## Contact

For questions or issues, please open an issue on this repository or contact the corresponding author:

- **Changhui Li** — [chli@pku.edu.cn](mailto:chli@pku.edu.cn)
