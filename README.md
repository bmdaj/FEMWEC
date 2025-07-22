# FEMWEC

**Finite Element Waveguide Eigenmode Calculation (FEMWEC)**

---

FEMWEC is a powerful tool designed to solve finite element problems for waveguide cross sections to accurately compute eigenmodes. It employs the finite element method (FEM) using vector elements, enabling precise and flexible modeling of complex waveguide structures.

### Key Features

- Calculates **eigenmodes** of waveguides with high accuracy.
- Supports **lossy eigenmodes** computation, including modes above the cutoff frequency in waveguides with lossy materials.
- Robust and efficient numerical implementation tailored for photonics and waveguide design.

---

### Applications

FEMWEC has been successfully used as part of the thesis for **topology optimization of phase shifters**, as demonstrated in the work:

> Beñat Martinez de Aguirre Jokisch, Rasmus Ellebæk Christiansen, and Ole Sigmund,  
> "Topology optimization framework for designing efficient thermo-optical phase shifters,"  
> *Journal of the Optical Society of America B*, 41(2), A18-A31 (2024).  
> [Read the paper](https://doi.org/10.1364/JOSAB.499979)

---

### Example

An example case is provided in the notebook [`example_tutorial.ipynb`](./example_tutorial.ipynb) solving a homogeneous waveguide.  
This example computes the waveguide eigenvalue expressed as the **complex effective index** of the mode, along with the eigenmodes’ electric field components (x, y, z) and their norm.

---

### Citation

If you use FEMWEC in your research or projects, please cite the following PhD thesis:

```bibtex
@phdthesis{99585b680429404f8f9a5667c0cec354,
  title     = {Multiphysics topology optimization in nanophotonics},
  author    = {Jokisch, {Benat Martinez de Aguirre}},
  year      = {2025},
  doi       = {10.11581/99585b68-0429-404f-8f9a-5667c0cec354},
  language  = {English},
  publisher = {Technical University of Denmark},
}
