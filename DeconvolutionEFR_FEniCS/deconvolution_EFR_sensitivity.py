from dolfin import *
import pdb
from Indicator_pybind11 import *
import numpy as np

class deconvolution_EFR():

    def __init__(self, **kwargs):
        r"""        Args:
            **kwargs:
                vel: velocity not filtred
				N : order of Filter
				delta : parameter of Filter
        """
        # Eval settings

        self.N = kwargs['N']
        self.delta = kwargs['delta']
        self.vel = kwargs['velocity']
        self.boundary = kwargs['boundaryMesh']
        self.comm = kwargs['comm']
        self.DFu =[]
        for i in range(self.N+1):
            u = Function(self.vel.function_space())
            self.DFu.append(u)
        self.Fu =[]
        for i in range(1):
            u = Function(self.vel.function_space())
            self.Fu.append(u)
        # Do vel,_ = mixedVariable.split(True) and not split(mixedVariable).
        # assert(isinstance(self.vel,Function))
   
    @staticmethod ###static method in python doesn't receive the first implicit argument
    def build_nullspace(V):
       """Function to build null space"""
       # Create list of vectors for null space
       xf = Function(V)
       x = xf.vector()
       tdim= V.mesh().topology().dim() # or number of component ?
       nullspace_basis = [x.copy() for i in range(tdim)]
       # Build translational null space basis
       for idx in range(0,tdim):
           V.sub(idx).dofmap().set(nullspace_basis[idx], 1.0)
       # Define Null Space or Dirichlet Datum.
       for x in nullspace_basis:
            x.apply("insert")
       # Create vector space basis and orthogonalize
       basis = VectorSpaceBasis(nullspace_basis)
       basis.orthonormalize()
       return basis

    

    def applyHelmholtzFilter(self,v,datum='Neumann'):
        # It is actually I - F_H implemented here
        if MPI.rank(self.comm) == 0:
            print("applyHelmholtzFilter ==============================================")
        u = Function(v.function_space())   ###memory leak??
        u_hat = Function(v.function_space())
        du = TrialFunction(v.function_space())
        ddu = TestFunction(v.function_space())
        tdim = v.function_space().mesh().topology().dim()
        # Define helmholtz filter
        f = + self.delta ** 2 * inner(grad(du),grad(ddu))*dx \
            + inner(du,ddu)*dx \
            - inner(v,ddu)*dx
        if datum=='Neumann':
            # add the nullspace 
            null_space=self.build_nullspace(v.function_space())
            bcs=[] 
        #TODO add 1 with lagrange multiplier
        else:
            # mf = MeshFunction('size_t',self.vel.function_space().mesh(), tdim-1)
            # boundary = CompiledSubDomain('on_boundary').mark(mf,2)
            # pdb.set_trace()

            with XDMFFile(MPI.comm_world, "output/boundary_helmholtz.xdmf") as xdmf_handler:
                # xdmf_handler.write(boundary)
                xdmf_handler.write(self.boundary)
            # bcs_h =     [DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 5 )]
            # bcs_h.append(DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 2 )) 
            # bcs_h.append(DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 4 ))
            # bcs_h.append(DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 6 )) 
            # bcs_h.append(DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 7 )) 
            # bcs_h.append(DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 8 )) 
            # bcs_h.append(DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 9 ))   
            # bcs_h.append(DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 3 ))   

            bcs_h =     [DirichletBC(self.vel.function_space(), self.vel, self.boundary, 5 )]
            bcs_h.append(DirichletBC(self.vel.function_space(), self.vel, self.boundary, 2 )) 
            bcs_h.append(DirichletBC(self.vel.function_space(), self.vel, self.boundary, 4 ))
            bcs_h.append(DirichletBC(self.vel.function_space(), self.vel, self.boundary, 6 )) 
            bcs_h.append(DirichletBC(self.vel.function_space(), self.vel, self.boundary, 7 )) 
            bcs_h.append(DirichletBC(self.vel.function_space(), self.vel, self.boundary, 8 )) 
            bcs_h.append(DirichletBC(self.vel.function_space(), self.vel, self.boundary, 9 ))   
            bcs_h.append(DirichletBC(self.vel.function_space(), self.vel, self.boundary, 3 )) 

            # bcs_h =     [DirichletBC(v.function_space(), v, self.boundary, 5 )]
            # bcs_h.append(DirichletBC(v.function_space(), v, self.boundary, 2 )) 
            # bcs_h.append(DirichletBC(v.function_space(), v, self.boundary, 4 ))
            # bcs_h.append(DirichletBC(v.function_space(), v, self.boundary, 6 )) 
            # bcs_h.append(DirichletBC(v.function_space(), v, self.boundary, 7 )) 
            # bcs_h.append(DirichletBC(v.function_space(), v, self.boundary, 8 )) 
            # bcs_h.append(DirichletBC(v.function_space(), v, self.boundary, 9 ))   
            # bcs_h.append(DirichletBC(v.function_space(), v, self.boundary, 3 ))
            # bcs = [DirichletBC(self.vel.function_space(), self.vel, mf, 10)] 

        A, b = assemble_system(lhs(f), rhs(f),bcs_h)
        if datum == 'Neumann':
            as_backend_type(A).set_nullspace(null_space)

        pc = PETScPreconditioner("petsc_amg")
        # Use Chebyshev smoothing for multigrid
        PETScOptions.set("mg_levels_ksp_type", "chebyshev")
        PETScOptions.set("mg_levels_pc_type", "jacobi")

        # Improve estimate of eigenvalues for Chebyshev smoothing
        PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
        PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)

        # Create CG Krylov solver and turn convergence monitoring on
        solver = PETScKrylovSolver("cg", pc)
        solver.parameters["monitor_convergence"] = True

        # Set matrix operator
        solver.set_operator(A);

        # Compute solution
        solver.solve(u.vector(), b);
        # solve(lhs(f)==rhs(f), u, bcs_h)
        # u_hat.vector().set_local(v.vector().get_local() - u.vector().get_local())
        # with XDMFFile(MPI.comm_world, "output/u-Fu.xdmf") as xdmf_handler:
                # xdmf_handler.write(u_hat)
        # return u.assign(self.vel - u)
        # file_helm = XDMFFile(MPI.comm_world,'output/testHelmholz.xdmf')
        # u.rename('Fu','Fu')
        # file_helm.write(u)
        return u

    def applyDeconvolution(self):
        ## Deconvolution is computed using Richardson method for F(u_D) = b, where b = F(u) 
        ## u_N+1 = u_N + (b - F(u_N))

        self.DFu[0].assign(self.Fu[0])
        DFu_tmp=Function(self.DFu[0].function_space())
        for n in range(self.N):
            # DFu_tmp = self.applyHelmholtzFilter(self.DFu[n],'Dirichlet')
            # pdb.set_trace()
            DFu_tmp = self.applyHelmholtzFilter(self.DFu[n],'Dirichlet')
            self.DFu[n+1].assign(self.DFu[n] +  self.DFu[0] - DFu_tmp)
            # self.DFu.append(self.DFu[n] +  self.DFu[0] - DFu_tmp)
            # self.DFu[n+1] = self.DFu[n] +  self.DFu[0] -  self.DFu[n+1]

        # return  DFu



    def computeIndicator(self, option='DG'):
        # with Yoshida regolarization
        # a_N(u) = [delta^2N+2 | ([delta^(-2)(I-F_H)]^(N+1) u|)
        # We need to do the Loop and set the indicator.
        # Fu=Function(self.vel.function_space())
        # Fu.assign(self.vel)
        indicatorVector     = Function(self.vel.function_space())
        indicatorVector_tmp = Function(self.vel.function_space())
        # indicatorVector.assign(self.vel)
        self.Fu[0].assign(self.applyHelmholtzFilter(self.vel,'Dirichlet'))
        # self.Fu.append(indicatorVector_tmp2)
        # self.Fu.append(self.applyHelmholtzFilter(self.vel,'Dirichlet'))
        # file_Fu = XDMFFile(MPI.comm_world,'output/testFu.xdmf')
        # self.Fu[0].rename('Fu[0]','Fu[0]')
        # file_Fu.write(self.Fu[0])
        self.applyDeconvolution()
        # file_DFu = XDMFFile(MPI.comm_world,'output/testDFu.xdmf')
        # self.DFu[self.N].rename('DFu[N]','DFu[N]')
        # file_DFu.write(self.DFu[self.N])
        # DFu = self.applyDeconvolution(Fu)
        indicatorAssigner = FunctionAssigner(indicatorVector.function_space(), self.DFu[self.N].function_space())
        indicatorAssigner.assign(indicatorVector_tmp, self.DFu[self.N])
        indicatorVector.assign(self.vel - indicatorVector_tmp)
        # indicatorVector.rename('indicatorVec','indicatorVec')
        # file_indicator = XDMFFile(MPI.comm_world,'output/testIndicatorVector.xdmf')
        # file_indicator.write(indicatorVector)

        ## compute F_H(u)
        # Fu.vector().set_local(self.applyhelmotzfilter(v= self.vel,datum='Dirichlet').vector().get_local())

        # indicatorVector_tmp=Function(self.vel.function_space())
        # indicatorVector_tmp.assign(self.vel)

        # indicator_2norm_out=Function(self.vel.function_space().sub(0).collapse())
        # indicator.vector().set_local(self.vel.vector().get_local())
        # for n in range(self.N+1):
            # pdb.set_trace()
            # indicatorVector.vector().set_local(self.applyhelmotzfilter(v= indicatorVector_tmp,datum='Dirichlet').vector().get_local())
            # indicatorVector_tmp.assign(indicatorVector)
        # indicator_2norm = indicator.vector().get_local()*indicator.vector().get_local()
        # indicator_2norm_out.vector().set_local(indicator_2norm)
        # return indicator_2norm
        # indicator_norm = norm(indicator.vector(),'linf')
        ## TODO if dim ==3 then...
        # pdb.set_trace()
        if option=='l2Local':
            if self.vel.function_space().mesh().topology().dim()==2:
                # pdb.set_trace()
                indicator_0, indicator_1 = indicatorVector.split(True)
                indicator_norm = Function(indicator_0.function_space())
                # indicator_norm.vector().set_local((indicator_0.vector().get_local()**2 + indicator_1.vector().get_local()**2)**0.5)
                indicator_norm.vector().set_local(indicator_0.vector().get_local()**2 + indicator_1.vector().get_local()**2)    #no sqrt
                # return indicator
            elif self.vel.function_space().mesh().topology().dim()==3:
                indicator_0, indicator_1, indicator_2 = indicatorVector.split(True)
                indicator_norm = Function(indicator_0.function_space())
                indicator_norm.vector().set_local((indicator_0.vector().get_local()**2 \
                                                 + indicator_1.vector().get_local()**2 \
                                                 + indicator_2.vector().get_local()**2 \
                                                 )**0.5)
            indicator_norm.rename('indicator_l2Local', 'indicator_l2Local')
            indicator_norm.vector().update_ghost_values()
            return indicator_norm
        elif option=='DG':

            DG = FunctionSpace(self.vel.function_space().mesh(),FiniteElement("DG",self.vel.function_space().mesh().ufl_cell(),0))
            ind = Indicator(indicatorVector)
            indicator_DG = project(ind,DG)
            indicator_DG.rename("indicator_DG","indicator_DG")
            indicator_DG.vecotr().update_ghost_values()
            return indicator_DG
          # indicator. assign(self.applyhelmotzfilter(indicator,'Dirichlet'))
        # indicator_tmp =  interpolate(Expression(("x[0]/sqrt(pow(x[0], 2) + pow(x[1], 2))",\
                                                 # "x[1]/sqrt(pow(x[0], 2) + pow(x[1], 2))"),degree=2), self.vel.function_space())
        # indicator_2norm = project(sqrt(inner(indicator_tmp,indicator_tmp)), self.vel.function_space().sub(0))
        # indicator_2norm=Function(self.vel.function_space().sub(0).collapse())
        # dv = TrialFunction(self.vel.function_space().sub(0).collapse())
        # ddv = TestFunction(self.vel.function_space().sub(0).collapse())
        # F_indicator = inner(ddv,dv) - inner(indicator*indicator, ddv)
        # solve(lhs(F_indicator)==rhs(F_indicator),indicator_2norm)
    def applyFilterSensitivity(self,u, datum ='Neumann'):
        if MPI.rank(self.comm) == 0:
            print("applyFilterSensitivity ==============================================")
               # It is actually I - F_H implemented here
        # u = Function(v.function_space())   ###memory leak??
        u_delta = Function(u.function_space())
        du = TrialFunction(u.function_space())
        ddu = TestFunction(u.function_space())
        tdim = u.function_space().mesh().topology().dim()
        # Define helmholtz filter
        f = + self.delta ** 2 * inner(grad(du),grad(ddu))*dx \
            + inner(du,ddu)*dx \
            - inner(u,ddu)*dx\
            - inner( 2 * self.delta * grad(u),grad(ddu))*dx
        if datum=='Neumann':
            # add the nullspace 
            null_space=self.build_nullspace(u.function_space())
            bcs=[] 
        #TODO add 1 with lagrange multiplier
        else:
            # mf = MeshFunction('size_t',self.vel.function_space().mesh(), tdim-1)
            # boundary = CompiledSubDomain('on_boundary').mark(mf,2)
            # pdb.set_trace()

            with XDMFFile(MPI.comm_world, "output/boundary_helmholtz.xdmf") as xdmf_handler:
                # xdmf_handler.write(boundary)
                xdmf_handler.write(self.boundary)
            

            # bcs_h =     [DirichletBC(u.function_space(), u, self.boundary, 5 )]
            # bcs_h.append(DirichletBC(u.function_space(), u, self.boundary, 2 )) 
            # bcs_h.append(DirichletBC(u.function_space(), u, self.boundary, 4 ))
            # bcs_h.append(DirichletBC(u.function_space(), u, self.boundary, 6 )) 
            # bcs_h.append(DirichletBC(u.function_space(), u, self.boundary, 7 )) 
            # bcs_h.append(DirichletBC(u.function_space(), u, self.boundary, 8 )) 
            # bcs_h.append(DirichletBC(u.function_space(), u, self.boundary, 9 ))   
            # bcs_h.append(DirichletBC(u.function_space(), u, self.boundary, 3 ))
            bcs_h =     [DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 5 )]
            bcs_h.append(DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 2 )) 
            bcs_h.append(DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 4 ))
            bcs_h.append(DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 6 )) 
            bcs_h.append(DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 7 )) 
            bcs_h.append(DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 8 )) 
            bcs_h.append(DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 9 ))   
            bcs_h.append(DirichletBC(self.vel.function_space(), Constant(tuple([0.] * tdim)), self.boundary, 3 )) ## should be Neumann???delta**2 * grad(u_delta)*n =0 

        # solve(lhs(f)==rhs(f), u_delta, bcs_h)
        A, b = assemble_system(lhs(f), rhs(f),bcs_h)
        if datum == 'Neumann':
            as_backend_type(A).set_nullspace(null_space)

        pc = PETScPreconditioner("petsc_amg")
        # Use Chebyshev smoothing for multigrid
        PETScOptions.set("mg_levels_ksp_type", "chebyshev")
        PETScOptions.set("mg_levels_pc_type", "jacobi")

        # Improve estimate of eigenvalues for Chebyshev smoothing
        PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
        PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)

        # Create CG Krylov solver and turn convergence monitoring on
        solver = PETScKrylovSolver("cg", pc)
        solver.parameters["monitor_convergence"] = True

        # Set matrix operator
        solver.set_operator(A);

        # Compute solution
        solver.solve(u_delta.vector(), b);
    
        return u_delta
    def computeIndicatorlocalSensitivity(self, u_delta):
        ## TODO compute indicator_delta
        ## indicator_delta = 2 * (u - DFu[N]) * (u_delta - DFu_delta[N]) 
        Fu_n_tmp = Function(self.Fu[0].function_space())   ###memory leak??
        DFu_delta_tmp = Function(self.Fu[0].function_space())  
        # pdb.set_trace()
        DFu_delta =[self.applyFilterSensitivity(u = self.Fu[0], datum = 'Dirichlet')]
        # DFu_delta.append([])
        for n in range(self.N):
            Fu_n_tmp.assign(self.DFu[n] + self.DFu[0] - self.DFu[n+1])
            DFu_delta_tmp.assign(DFu_delta[n]+ DFu_delta[0] - self.applyFilterSensitivity(Fu_n_tmp,datum = 'Dirichlet'))
            DFu_delta.append(DFu_delta_tmp )
            # u_n_delta.append(u_n_delta[0])
        # pdb.set_trace()

        tmp = Function(self.vel.function_space()) 
        DFu_delta_0, _   = DFu_delta[self.N].split(True)
        indicator_delta1 = Function(self.vel.function_space())
        indicator_delta2 = Function(self.vel.function_space())
        indicatorAssigner = FunctionAssigner(self.vel.function_space(), self.DFu[self.N].function_space())
        indicatorAssigner2 = FunctionAssigner(u_delta.function_space(), DFu_delta[self.N].function_space())
        tmp2 = Function(u_delta.function_space()) 

        indicatorAssigner.assign(tmp ,self.DFu[self.N])
        indicator_delta1.assign(self.vel - tmp)
        indicatorAssigner2.assign(tmp2 ,DFu_delta[self.N])
        indicator_delta2.assign(u_delta - tmp2)
        indicator_delta1_0, indicator_delta1_1 = indicator_delta1.split(True)
        indicator_delta2_0, indicator_delta2_1 = indicator_delta2.split(True)
        indicator_delta  = Function(indicator_delta1_0.function_space())  
        indicator_delta.vector().set_local(2*(indicator_delta1_0.vector().get_local()* indicator_delta2_0.vector().get_local() \
                                          + indicator_delta1_1.vector().get_local()*indicator_delta2_1.vector().get_local()))
        # indicator_delta.vector().apply('insert')
        indicator_delta.vector().update_ghost_values()
        # file_indicator_delta = XDMFFile(MPI.comm_world,'output/testIndicator_delta.xdmf')
        # indicator_delta.rename('indicator_delta','indicator_delta')
        # file_indicator_delta.write(indicator_delta)
        # return DFu_delta_0
        return indicator_delta
