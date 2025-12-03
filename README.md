# Neurovascular Unit (NVU) Three-Phase Model
## Mathematical Specifications from Research Paper

---

## Overview

This document contains all equations, parameters, and boundary conditions needed to implement a three-phase computational model of glucose transport in a neurovascular unit (NVU).

### The Three Phases:
1. **Phase 1 (Ω_I)**: Blood (Vascular Compartment) - Capillary lumen
2. **Phase 2 (Ω_E, Ω_BL)**: Blood-Brain Barrier (Endothelial Layer + Basal Lamina)
3. **Phase 3 (Ω_II)**: Brain Tissue (Porous Medium / Interstitial Space)

---

## 1. GEOMETRY PARAMETERS

### Structural Dimensions

| Symbol | Description | Value | Units |
|--------|-------------|-------|-------|
| L_capillary | Length of capillary segment | 25 | μm |
| R_0 | Capillary lumen radius | 7 | μm |
| h_end | Endothelial cell layer thickness | 1 | μm |
| h_bl | Basal lamina thickness | 100 | nm (0.1 μm) |
| L_surround | Brain tissue thickness (radial) | 25 | μm |

### Domain Definitions
- **Ω_I**: Capillary lumen (cylinder: radius R_0, length L_capillary)
- **Ω_E**: Endothelial cells (cylindrical shell: inner radius R_0, thickness h_end)
- **Ω_BL**: Basal lamina (cylindrical shell: thickness h_bl)
- **Ω_II**: Brain parenchyma (cylindrical shell: thickness L_surround)

### Boundary Surfaces
- **S'_∂Ω_I, S''_∂Ω_I**: Inlet and outlet transverse surfaces of capillary
- **H_inner_∂Ω_E**: Luminal (inner) surface of endothelium
- **H_outer_∂Ω_E**: Abluminal (outer) surface of endothelium
- **H_inner_∂Ω_II**: Inner surface of brain tissue (contacts basal lamina)
- **H_outer_∂Ω_II**: Outer surface of brain tissue
- **S_end-feet**: Area of astrocyte end-feet on H_inner_∂Ω_II

---

## 2. PHASE 1: BLOOD (CAPILLARY LUMEN - Ω_I)

### 2.1 Governing Equations - Blood Flow

**Non-steady-state Navier-Stokes equation** for incompressible non-Newtonian fluid:

```
ρ ∂u/∂t + ρ(u·∇)u = -∇p + ∇(μ(|γ̇|)(∇u + (∇u)ᵀ))
```

**Incompressibility condition:**
```
∇·u = 0
```

**Shear rate tensor:**
```
γ̇ = 2ε = ∇u + (∇u)ᵀ
|γ̇| = √(2(ε:ε))
```

Where:
- u = velocity vector [m/s]
- ρ = blood density [kg/m³]
- p = pressure [Pa]
- μ = dynamic viscosity [Pa·s]
- γ̇ = shear rate tensor [1/s]

### 2.2 Carreau Model for Blood Viscosity

```
μ(|γ̇|) = μ_∞ + (μ_0 - μ_∞)[1 + (λγ̇)²]^((n-1)/2)
```

| Parameter | Description | Value | Units |
|-----------|-------------|-------|-------|
| μ_∞ | Limit viscosity (Newtonian) | 3.45 × 10⁻³ | Pa·s |
| μ_0 | Zero-shear rate viscosity | 5.6 × 10⁻² | Pa·s |
| λ | Time constant | 3.131 | s |
| n | Power-law index | 0.3568 | - |
| ρ_blood | Blood density | 1,070 | kg/m³ |

### 2.3 Pressure Conditions

**Initial pressure:**
```
p_0 = 18.5 mmHg = 2,466 Pa
```

**Inlet/Outlet pressures with time-dependent changes:**
```
p_inlet = (p_0 + Δp) · f_shift(t)
p_outlet = (p_0 - Δp) · f_shift(t)
```

**Step-like function for CBF changes:**
```
f_dec_shift(t) = {
  1,           if t < t_0
  1 - δ(t),    if t ≥ t_0
}

f_inc_shift(t) = {
  1,           if t < t_0
  1 + δ(t),    if t ≥ t_0
}
```

Where:
- t_0 = 2 s (time of CBF change initiation)
- δ(t) ∈ [0, a], with 0.5 ≤ a < 1
- Δt = 0.5 s (smoothing transition zone)

### 2.4 Glucose Transport in Blood (Convection-Diffusion-Reaction)

```
∂c/∂t = ∇·(D·∇c) - u·∇c + f_con(c,r)
```

Where:
- c = glucose concentration [mol/m³ or mM]
- D = diffusion tensor [m²/s]
- u = velocity field from Navier-Stokes
- f_con = consumption rate function

**Diffusion coefficient in blood:**
```
D_Glc_CBF = 3.1 × 10⁻¹⁰ m²/s
```

**Diffusion tensor (isotropic in blood):**
```
D = D_Glc_CBF · I
```
Where I is the identity matrix.

### 2.5 Glucose Consumption in Blood

```
f_con(c,r) = -ε_Glc/ATP^(Ω_I) · c/(c + K_Glc)
```

Where:
```
ε_Glc/ATP^(Ω_I) = k_Glc^(Ω_I) · c_ATP^(Ω_I) · [1 + (c_ATP^(Ω_I)/K_I,ATP)^(nH)]^(-1)
```

**Note**: ε_Glc/ATP^(Ω_I) ≪ ε_Glc/ATP^(Ω_E) (lower in blood than tissue)

---

## 3. PHASE 2: BLOOD-BRAIN BARRIER (Ω_E, Ω_BL)

### 3.1 Endothelial Layer (Ω_E)

**Governing equation (no convection in endothelium):**
```
∂c/∂t = ∇·(D·∇c) + f_con(c,r)
```

**Diffusion coefficient:**
```
D_Glc_end = D_Glc_CBF = 3.1 × 10⁻¹⁰ m²/s
```

**Glucose consumption:**
```
f_con(c,r) = -ε_Glc/ATP^(Ω_E) · c/(c + K_Glc)
```

Where:
```
ε_Glc/ATP^(Ω_E) = k_Glc^(Ω_E) · c_ATP^(Ω_E) · [1 + (c_ATP^(Ω_E)/K_I,ATP)^(nH)]^(-1)
```

### 3.2 Basal Lamina (Ω_BL)

**Governing equation:**
```
∂c/∂t = ∇·(D·∇c)
```
(No consumption in basal lamina)

**Diffusion coefficient:**
```
D_Glc_BL = 1.6 × 10⁻¹⁰ m²/s
```

### 3.3 Glucose Transporter (GLUT1) Boundary Conditions

**Flux discontinuity at membranes** (H_inner_∂Ω_E, H_outer_∂Ω_E):

```
-n·(J + u·c) = N_GLUT1 · k_cat^1 · c / (N_A · (K_m^1 + c))
```

Where:
- J = -D·∇c (diffusive flux)
- N_GLUT1 = number of transporters per μm²
- k_cat^1 = turnover number
- K_m^1 = Michaelis constant
- N_A = Avogadro's number

**GLUT1 Parameters:**

| Parameter | Description | Value | Units |
|-----------|-------------|-------|-------|
| K_m^1 | GLUT1 affinity constant | 8 | mmol/L |
| k_cat^1 | GLUT1 turnover rate at 37°C | 1.166 × 10³ | 1/s |
| N_GLUT1^(H_inner) | Luminal membrane density | 1.0 × 10³ | 1/μm² |
| N_GLUT1^(H_outer) | Abluminal membrane density | 1.0 × 10³ | 1/μm² |

---

## 4. PHASE 3: BRAIN TISSUE (Ω_II)

### 4.1 Interstitial Fluid Flow (Darcy's Law)

**Velocity field in porous medium:**
```
u = -(κ/μ_ISF)∇p
```

Where:
- κ = Darcy's permeability [m²]
- μ_ISF = interstitial fluid viscosity [Pa·s]

**Parameters:**

| Parameter | Description | Value | Units |
|-----------|-------------|-------|-------|
| κ | Darcy's permeability | 6.5 × 10⁻¹⁵ | m² |
| μ_ISF | ISF viscosity | 7 × 10⁻⁴ | Pa·s |
| ρ_ISF | ISF density | 1,000 | kg/m³ |

### 4.2 Glucose Transport in Brain Parenchyma

**Governing equation (porous media with dispersion):**
```
∂/∂t[α_ICF · c] = ∇·((D_d + α_ICF/τ · D)·∇c) - u·∇c + f_con(c,r)
```

Where:
- α_ICF = porosity (volume fraction of ISF)
- τ = tortuosity
- D_d = dispersion tensor

**Porous Medium Parameters:**

| Parameter | Description | Value | Units |
|-----------|-------------|-------|-------|
| α_ISF | Porosity (volume fraction) | 0.36 | - |
| τ | Interstitial tortuosity | 1.635 | - |

### 4.3 Dispersion Tensor

**For low ISF velocities, dispersion simplifies to:**
```
D_d^L = D_d^T = D_c/τ
```

Where:
- D_d^L = longitudinal dispersivity
- D_d^T = transverse dispersivity
- D_c = molecular diffusion coefficient

### 4.4 Diffusion Tensor in Brain Tissue

**Anisotropic diffusion:**
```
D = D_Glc · diag(σ_xx, σ_yy, σ_zz)
```

**Parameters:**

| Parameter | Description | Value | Units |
|-----------|-------------|-------|-------|
| D_Glc | Glucose diffusion in brain at 37°C | 8.7 × 10⁻¹⁰ | m²/s |
| σ_xx | Axial anisotropy coefficient | 1.0 | - |
| σ_yy = σ_zz | Transverse anisotropy coefficient | 0.33 | - |

### 4.5 Glucose Consumption in Brain Tissue

**Michaelis-Menten kinetics with ATP regulation:**
```
f_con(c,r) = -ε_Glc/ATP^(Ω_II) · c/(c + K_Glc)
```

Where:
```
ε_Glc/ATP^(Ω_II) = k_Glc^(Ω_II) · c_ATP^(Ω_II) · [1 + (c_ATP^(Ω_II)/K_I,ATP)^(nH)]^(-1)
```

**Consumption Parameters:**

| Parameter | Description | Value | Units |
|-----------|-------------|-------|-------|
| k_Glc | Kinetic constant | 120 × 10⁻³ | 1/s |
| K_Glc | Glucose affinity constant | 0.05 | mM |
| K_I,ATP | ATP inhibition constant | 1 | mM |
| nH | Hill coefficient | 4 | - |
| c_ATP | ATP concentration | 1 | mM |

**Note**: ε_Glc/ATP^(Ω_E) ≈ ε_Glc/ATP^(Ω_II) (similar in endothelium and brain tissue)

### 4.6 Astrocyte End-Feet Boundary

**GLUT1 transporters on end-feet surface:**
```
-n·(J + u·c)|_S_end-feet = N_GLUT1^(end-feet) · k_cat^1 · c / (N_A · (K_m^1 + c))
```

**Free diffusion in clefts between end-feet:**
```
-n·[(J + u·c)|_H_inner_Ω_II - (J + u·c)|_H_outer_Ω_BL] = 0
```
for r ∈ H_inner_∂Ω_II \ S_end-feet

**End-feet parameters:**

| Parameter | Description | Value | Units |
|-----------|-------------|-------|-------|
| N_GLUT1^(end-feet) | End-feet GLUT1 density | 0.018 × 10³ | 1/μm² |
| δ_end-feet | End-feet coverage ratio | Variable (20-86%) | % |

**Coverage ratio:**
```
δ_end-feet = (S_end-feet / S_total) × 100%
```

---

## 5. BOUNDARY CONDITIONS

### 5.1 Phase 1 (Blood) Boundaries

**Inlet (S''_∂Ω_I) - Danckwerts condition:**
```
-n·(J + u·c) = -n·(u·c̃(t))
```
Where c̃(t) is the inlet concentration.

**Outlet (S'_∂Ω_I) - Outflow condition:**
```
-n·(D·∇c) = 0
```

### 5.2 Phase 2 (BBB) Boundaries

**Transverse surfaces (no flux):**
```
-n·J = 0    for r ∈ S'_∂Ω_E/BL, S''_∂Ω_E/BL
```

**Or fixed concentration (for long capillaries):**
```
c(r,t) = c_0(t)    for r ∈ S'_∂Ω_E/BL, S''_∂Ω_E/BL
```

### 5.3 Phase 3 (Brain Tissue) Boundaries

**Outer boundary (H_outer_∂Ω_II):**
```
-n·(D·∇c) = 0
```

### 5.4 Interface Boundaries (GLUT1 Transport)

See sections 3.3 and 4.6 for flux discontinuity conditions at:
- H_inner_∂Ω_E (luminal endothelial surface)
- H_outer_∂Ω_E (abluminal endothelial surface)
- S_end-feet ⊂ H_inner_∂Ω_II (astrocyte end-feet)

---

## 6. INITIAL CONDITIONS

**Initial glucose concentrations:**
```
c(r,t)|_{t=0} = ⟨c⟩|_{I/E/BL/II}
```

**Typical values:**
- ⟨c⟩|_I = 5 mM (blood)
- ⟨c⟩|_E = 1 mM (endothelium)
- ⟨c⟩|_BL = 1 mM (basal lamina)
- ⟨c⟩|_II = 1 mM (brain tissue)
- c̃(t) = 5 mM (inlet concentration)

**Expected tissue concentration:**
- Average in Ω_II: 1.03-2.2 mM (depending on δ_end-feet)
- Experimental value: 1.7 ± 0.9 mM

---

## 7. NUMERICAL IMPLEMENTATION NOTES

### Time Parameters
- Total simulation time: 4 s
- CBF change time: t_0 = 2 s
- Analysis window: 1.5 s ≤ t ≤ 2.5 s
- Transition smoothing: Δt = 0.5 s

### Expected Blood Flow Velocities
- Normal CBF: ⟨u⟩ ≈ 1.28 ± 0.06 mm/s
- Experimental range: 0.99-2.03 mm/s
- ISF velocity: ≈ 4-5 × 10⁻⁷ m/s

### Mesh Requirements
- Method: Finite Element Method (FEM)
- Mesh: User-controlled extra-fine
- Minimum angle: 240°
- Element size scaling: 0.1

---

## 8. KEY PHYSICAL CONSTANTS

| Constant | Symbol | Value |
|----------|--------|-------|
| Avogadro's number | N_A | 6.022 × 10²³ mol⁻¹ |
| Universal gas constant | R | 8.314 J/(mol·K) |
| Temperature | T | 310 K (37°C) |

---

## 9. COUPLING BETWEEN PHASES

### Phase 1 → Phase 2
- Blood flow creates pressure gradient
- Velocity field u drives convective flux
- GLUT1 at H_inner_∂Ω_E mediates glucose entry

### Phase 2 → Phase 3
- GLUT1 at H_outer_∂Ω_E mediates glucose exit from endothelium
- Diffusion through basal lamina (Ω_BL)
- GLUT1 at end-feet (partial coverage) and free diffusion in clefts

### Feedback Effects
- CBF changes alter glucose delivery
- Glucose consumption affects local gradients
- End-feet coverage (δ_end-feet) modulates brain glucose levels

---

## 10. MODEL VALIDATION

Expected outputs should match:
1. CBF velocity: 1-2 mm/s range
2. ISF velocity: ~5 × 10⁻⁷ m/s
3. Brain glucose: 1.7 ± 0.9 mM
4. Decreased CBF with low δ_end-feet → increased brain glucose
5. Increased CBF → always increased brain glucose

---

## References

Nartsissov YR (2022). Application of a multicomponent model of convectional reaction-diffusion to description of glucose gradients in a neurovascular unit. Front. Physiol. 13:843473. doi: 10.3389/fphys.2022.843473
