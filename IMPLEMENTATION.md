# NVU Three-Phase PINN Model - Implementation Guide

## Overview

This repository contains a comprehensive Physics-Informed Neural Network (PINN) implementation for modeling glucose transport in a Neurovascular Unit (NVU), based on the paper by Nartsissov YR (2022).

## Files

- `nvu_pinn_model.ipynb` - Main Jupyter notebook with complete implementation
- `requirements.txt` - Python dependencies
- `README.md` - Original problem specification and parameters
- `IMPLEMENTATION.md` - This file

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Model

```bash
jupyter notebook nvu_pinn_model.ipynb
```

Then run all cells in order.

## Implementation Details

### Issue 1: Domain Decomposition Architecture ✓

**Problem**: Original code used a single neural network for all domains.

**Solution**: Implemented domain-decomposed architecture with:
- Shared encoder for feature extraction
- Separate branches for Blood, BBB, and Brain domains
- Domain masks to apply correct physics
- Each domain has appropriate outputs:
  - Blood: (u_r, u_z, p, c)
  - BBB: (c)
  - Brain: (u_r, u_z, p, c)

### Issue 2: Complete Physics Implementation ✓

**Implemented all physics from the paper:**

#### 2.1 Navier-Stokes in Cylindrical Coordinates
- Full momentum equations for r and z components
- Carreau viscosity model: `μ = μ∞ + (μ0-μ∞)[1+(λγ̇)²]^((n-1)/2)`
- Continuity equation: `∇·u = 0`
- Proper cylindrical coordinate formulation

#### 2.2 Time-Dependent CBF Changes
```python
f_shift(t) = {
    1,           if t < t_0
    1 ± δ(t),    if t ≥ t_0
}
```
- Smooth transition using sigmoid over Δt = 0.5s
- Supports CBF increase and decrease scenarios

#### 2.3 GLUT1 Boundary Conditions
Three interfaces with GLUT1 transporters:
1. Blood-Endothelium (luminal): `N_GLUT1 = 1.0×10³ 1/μm²`
2. Endothelium-Basal Lamina (abluminal): `N_GLUT1 = 1.0×10³ 1/μm²`
3. Basal Lamina-Brain (end-feet): `N_GLUT1 = 0.018×10³ 1/μm²`

Flux equation: `Flux = N·k_cat·c/(N_A·(K_m+c))`

#### 2.4 Proper Boundary Conditions
- **Inlet (Danckwerts)**: `-n·(J + u·c) = -n·(u·c_inlet)`
- **Outlet**: `-n·(D·∇c) = 0`
- **Wall**: No-slip `u = 0`
- **Symmetry**: `u_r = 0` and `∂u_z/∂r = 0` at r=0

#### 2.5 Variable End-Feet Coverage
- Configurable coverage: 20-86% (default: 50%)
- Weighted flux condition at brain interface
- Models realistic astrocyte end-feet distribution

#### 2.6 Dispersion Tensor
For porous media transport in brain:
```
D_eff = D_Glc·α_ISF/τ + D_Glc/τ
```

### Issue 3: Numerical Stability ✓

#### 3.1 Singularity at r=0
**Fixed by**:
- Using `r_safe = r + 1e-6` instead of `r + 1e-10`
- Proper handling of 1/r terms in cylindrical coordinates
- L'Hôpital's rule implicit in formulation

#### 3.2 Interface Handling
- Continuous predictions enforced by network architecture
- Flux discontinuities properly modeled via GLUT1 transporters
- Domain masks ensure smooth transitions

#### 3.3 Adaptive Loss Weighting
```python
class AdaptiveLossWeighting:
    - Balances physics, boundary, and initial condition terms
    - Updates weights every 100 iterations
    - Uses exponential moving average for stability
```

### Issue 4: Validation Suite ✓

#### Expected Values from Paper

| Metric | Expected Value | Range |
|--------|----------------|-------|
| Blood velocity | 1.28 mm/s | 0.99-2.03 mm/s |
| Brain glucose | 1.7 mM | 1.03-2.2 mM |
| ISF velocity | 4.5×10⁻⁷ m/s | Order of 10⁻⁷ |

#### Validation Functions
- `validate_model()`: Compares predictions to experimental ranges
- Automatic PASS/FAIL indicators
- Mass conservation checks

#### Comprehensive Visualization
9-subplot figure showing:
1. Radial glucose profile
2. Blood velocity profile
3. Pressure distribution
4. Temporal evolution
5. Glucose gradient
6. Domain decomposition
7. 2D concentration field
8. 2D velocity field
9. Validation comparison

## Physical Parameters

All parameters from Nartsissov YR (2022):

### Geometry
- Capillary radius: 7 μm
- Endothelium thickness: 1 μm
- Basal lamina thickness: 0.1 μm
- Brain tissue thickness: 25 μm
- Capillary length: 25 μm

### Blood (Carreau Model)
- μ∞ = 3.45×10⁻³ Pa·s
- μ0 = 5.6×10⁻² Pa·s
- λ = 3.131 s
- n = 0.3568
- ρ = 1070 kg/m³

### Diffusion Coefficients
- Blood: 3.1×10⁻¹⁰ m²/s
- Endothelium: 3.1×10⁻¹⁰ m²/s
- Basal lamina: 1.6×10⁻¹⁰ m²/s
- Brain: 8.7×10⁻¹⁰ m²/s

### GLUT1 Parameters
- K_m = 8.0 mM
- k_cat = 1.166×10³ 1/s

### Brain Tissue (Darcy Flow)
- κ = 6.5×10⁻¹⁵ m²
- μ_ISF = 7×10⁻⁴ Pa·s
- α_ISF = 0.36 (porosity)
- τ = 1.635 (tortuosity)

## Usage Examples

### Basic Training

```python
# Default configuration (steady state)
history = train_pinn(model, X_domain, X_boundary, X_initial, params,
                     epochs=1000, lr=1e-3, cbf_type='none')
```

### Simulating CBF Decrease

```python
# 30% decrease in CBF at t=2s
params.cbf_change_amplitude = 0.3
history = train_pinn(model, X_domain, X_boundary, X_initial, params,
                     epochs=5000, lr=1e-3, cbf_type='decrease')
```

### Varying End-Feet Coverage

```python
# Study effect of low end-feet coverage (20%)
params.delta_endfeet = 0.20
model = DomainDecomposedNN(params)
# ... continue with training
```

## Expected Performance

- **Training time**: ~5-15 minutes for 1000 epochs (CPU)
- **Final loss**: Should converge to < 10⁻³
- **Memory**: ~500 MB for default configuration
- **Validation**: All metrics should be within experimental ranges

## Notebook Structure

1. **Cell 1**: Imports and setup
2. **Cell 2-3**: Physical parameters class
3. **Cell 4-5**: Domain-decomposed neural network
4. **Cell 6-7**: Physics functions (Carreau, f_shift, GLUT1, consumption)
5. **Cell 8-9**: Physics residuals (Navier-Stokes, Darcy, transport)
6. **Cell 10-11**: Boundary conditions (all interfaces)
7. **Cell 12-13**: Training infrastructure (adaptive weighting, points)
8. **Cell 14-15**: Loss function
9. **Cell 16-17**: Training loop
10. **Cell 18-19**: Validation function
11. **Cell 20-21**: Visualization function
12. **Cell 22-25**: Execution cells (train, validate, visualize)
13. **Cell 26**: Summary and documentation

## Troubleshooting

### Low Blood Velocity
- Check pressure boundary conditions
- Verify Carreau viscosity parameters
- Increase weight for Navier-Stokes residuals

### High Brain Glucose
- Verify GLUT1 flux at interfaces
- Check consumption parameters
- Adjust end-feet coverage

### Training Instability
- Reduce learning rate
- Increase adaptive weighting update frequency
- Add more collocation points

### Slow Convergence
- Increase number of collocation points
- Use learning rate schedule
- Train for more epochs (5000-10000)

## Customization

### Adding New Species
1. Add diffusion coefficient to `NVUParameters`
2. Add concentration output to neural network
3. Add transport equation to `compute_physics_residuals`
4. Add boundary conditions

### Different Geometries
1. Modify geometry parameters in `NVUParameters`
2. Update domain masks in `DomainDecomposedNN.call()`
3. Adjust collocation point generation

### Alternative Blood Flow Models
1. Replace `carreau_viscosity()` with new model
2. Update Navier-Stokes residuals accordingly
3. Adjust viscosity parameters

## Citations

If you use this code, please cite:

```bibtex
@article{nartsissov2022,
  title={Application of a multicomponent model of convectional reaction-diffusion to description of glucose gradients in a neurovascular unit},
  author={Nartsissov, Yaroslav R},
  journal={Frontiers in Physiology},
  volume={13},
  pages={843473},
  year={2022},
  publisher={Frontiers Media SA},
  doi={10.3389/fphys.2022.843473}
}
```

## License

This implementation is provided for research and educational purposes.

## Contact

For questions or issues, please open an issue in the GitHub repository.

---

**Implementation Date**: December 2024  
**Python Version**: 3.8+  
**TensorFlow Version**: 2.10+
