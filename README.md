# DeconvolutionEFR - A large eddy simulation model for the incompressible flow with moderate large Reynolds numbers
Implementation of the Leray-alpha-EFR model for incompressible flows. 

This package implements a Large Eddy Simulation (LES) model based on the Leray model for the incompressible flow with moderate large Reynolds numbers (at the range of a few thousands). The Leray model was originally proposed as a regularization of the Navier-Stokes equations and more recently attracted attention as a large eddy simulation model  This package implements a deconvolution-based Leray model using a Evolve-Filter-Relax scheme. Not only does the model have localized stabilization with high order of accuracy, but also it is able to stabilize the backflow instability appear in certain flow applications, such as hemodynamics, with the appropriate choice of model parameters, see  for details of the model and for its applications.

This package builds on the parallel finite element library FEniCS for the discretization of the partial differential equations, FENaPack \cite{Blechta2016Fenapack} for preconditioning and PETSc for scalable linear solvers. This package supports structured/unstructured (2D/3D) meshes in XDMF fromat, a YAML based input file format for user settings and the XDMF file format for visualization.

The effectiveness of the filter in the Leray models is the key to the success of the model. This package provides the implementation of the Van Cittert-Helmholtz deconvolution filter  with arbitrarily high orders of the approximated deconvolution. 

Another critical aspect of Leray models for the LES of incompressible flows at moderately large Reynolds number is the selection of the filter radius. This drives the effective regularization of the filtering procedure, and its selection is a trade-off between stability (the larger, the better) and accuracy (the smaller, the better). This package also provides the implementation of the adjoint equations of the model with respect to the filter radius for the local sensitivity analysis.


## Reference

* J. Leray. Sur le mouvement d’un liquide visqueux emplissant l’espace. Acta mathematica, 63(1):193–248, 1934.
* W. Layton, C. C. Manica, M. Neda, and L. G. Rebholz. Numerical analysis and computational testing of a high accuracy leray-deconvolution model of turbulence. Numerical Methods for Partial Differential Equations: An International Journal, 24(2):555–582, 2008.
* L. Bertagna, A. Quaini, and A. Veneziani. Deconvolution-based nonlinear filtering for incompressible flows at moderately large reynolds numbers. International Journal for Numerical Methods in Fluids, 2015.
* H. Xu, M. Piccinelli, B. G. Leshnower, A. Lefieux, W. R. Taylor, and A. Veneziani. Coupled morphological–hemodynamic computational analysis of type b aortic dissection: A longitudinal study. Annals of Biomedical Engineering, pages 1–13, 2018.
* L. Bertagna, A. Quaini, L. G. Rebholz, and A. Veneziani. On the sensitivity to the filtering radius in leray models of incompressible flow. In Contributions to Partial Differential Equations and Applications, pages 111–130. Springer, 2019.
* H. Xu, D. Baroli, F. Massimo, A. Quaini, A. Veneziani. Backflow Stabilization by Deconvolution-based Large Eddy Simulation Modeling. Journal of Computational Physics (In press), 2019.
* H. P. Langtangen and A. Logg. Solving PDEs in Python: The FEniCS Tutorial I. Springer, 2016.
* J. Blechta, M. Řehoř, Fenapack2016, http://dx.doi.org/10.5281/zenodo.56754.

## Acknowledgment
The work is supported by the National Science Foundation through grants NSF-DMS1620406/162038 (PIs: Alessandro Veneziani and Annalisa Quaini) and TG-ASC160069 (PI: Alessandro Veneziani).
