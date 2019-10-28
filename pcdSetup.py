from fenapack import PCDKrylovSolver, PCDAssembler
from dolfin import *
import pdb

def build_nullspace(V):
    """Function to build null space"""
    xf = Function(V)
    null_vector = Vector(xf.vector())
    V.sub(1).dofmap().set(null_vector, 1.0)
    null_vector *= 1.0/null_vector.norm("l2")
    null_vector.apply("insert")
    nullspace_basis = VectorSpaceBasis([null_vector])
    return nullspace_basis


def create_pcd_solver(comm, pcd_variant, ls, mumps_debug=False):
    #Author M. Rehor (to add in paper)

    prefix = "po_"

    # Set up linear solver (GMRES with right preconditioning using Schur fact)
    linear_solver = PCDKrylovSolver(comm=comm)
    linear_solver.set_options_prefix(prefix)
    linear_solver.parameters["relative_tolerance"] = 1e-10
    PETScOptions.set(prefix+"ksp_gmres_restart", 150)

    # Set up subsolvers
    PETScOptions.set(prefix+"fieldsplit_p_pc_python_type", "fenapack.PCDRPC_" + pcd_variant)
    if ls == "iterative":
        PETScOptions.set(prefix+"fieldsplit_u_ksp_type", "richardson")
        PETScOptions.set(prefix+"fieldsplit_u_ksp_max_it", 1)
        PETScOptions.set(prefix+"fieldsplit_u_pc_type", "hypre") # "gamg"
        PETScOptions.set(prefix+"fieldsplit_u_pc_hypre_type", "boomeramg")

        PETScOptions.set(prefix+"fieldsplit_p_PCD_Rp_ksp_type", "richardson")
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Rp_ksp_max_it", 1)
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Rp_pc_type", "hypre") # "gamg"
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Rp_pc_hypre_type", "boomeramg")

        PETScOptions.set(prefix+"fieldsplit_p_PCD_Ap_ksp_type", "richardson")
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Ap_ksp_max_it", 1)
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Ap_pc_type", "hypre") # "gamg"
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Ap_pc_hypre_type", "boomeramg")

        PETScOptions.set(prefix+"fieldsplit_p_PCD_Mp_ksp_type", "chebyshev")
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Mp_ksp_max_it", 5)
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Mp_ksp_chebyshev_eigenvalues", "0.5, 2.0")
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Mp_pc_type", "jacobi")
    elif ls == "direct":
        # Debugging MUMPS
        if mumps_debug:
            PETScOptions.set(prefix+"fieldsplit_u_mat_mumps_icntl_4", 2)
            PETScOptions.set(prefix+"fieldsplit_p_PCD_Ap_mat_mumps_icntl_4", 2)
            PETScOptions.set(prefix+"fieldsplit_p_PCD_Mp_mat_mumps_icntl_4", 2)
    else:
        assert False

    # Apply options
    linear_solver.set_from_options()

    return linear_solver