import numpy as np
from dolfin import *
# from dolfin import *
from fenapack import PCDKrylovSolver, PCDAssembler
import pdb
from pcdSetup import *
from dragLift import *
from deconvolution_EFR_sensitivity import *


def NS_leray_localSensitivity(comm, meshPath, boundaryMeshPath,dt, NT, rho, vis, indicatorOption, deconOrder, filterOn, sensitivityOn, chi):


    file_handler_press = XDMFFile(comm,'outputN0/press_NS_BDF2_pcd.xdmf')
    file_handler_vel   = XDMFFile(comm,'output/vel_NS_BDF2_pcd.xdmf')
    csvDir = "csv/NS_BDF2_PCD/"
    liftFile =csvDir + "lift_simu_semiImplicitBDF2_pcd.csv"
    dragFile =csvDir + "drag_simu_semiImplicitBDF2_pcd.csv"
    pressDiffFile =csvDir + "pressDiff_simu_semiImplicitBDF2_pcd.csv"
    file_handler_indicator = XDMFFile(comm,'output/indicator_'+indicatorOption+'_N'+str(deconOrder)+'.xdmf')

    if filterOn:
        file_handler_vel_leray     = XDMFFile(comm,'output/vel_leray_BDF2_pcd.xdmf')
        file_handler_press_leray   = XDMFFile(comm,'output/press_leray_BDF2_pcd.xdmf')
        file_handler_vel_relax   = XDMFFile(comm,'output/vel_relax_BDF2_pcd.xdmf')
        file_handler_press_relax = XDMFFile(comm,'output/press_relax_BDF2_pcd.xdmf')
        if sensitivityOn:
            file_handler_vel_delta   = XDMFFile(comm,'output/vel_delta_BDF2_pcd.xdmf')
            file_handler_press_delta = XDMFFile(comm,'output/press_delta_BDF2_pcd.xdmf')
            file_handler_vel_leray_delta = XDMFFile(comm,'output/vel_leray_delta_BDF2_pcd.xdmf')
            file_handler_press_leray_delta = XDMFFile(comm,'output/press_leray_delta_BDF2_pcd.xdmf')
            file_indicator_delta = XDMFFile(MPI.comm_world,'output/Indicator_delta.xdmf')

    ## Mesh===============================================================
    ## read mesh file-----------------------------------------------------
    mesh = Mesh(comm)
    with XDMFFile(comm, meshPath) as xdmf:
         xdmf.read(mesh)
    boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    with XDMFFile(comm, boundaryMeshPath) as xdmf:
         xdmf.read(boundaries)
    ## create toy mesh inside for debugging purpose-------------------------
    # nx=100
    # ny=50
    # mesh = RectangleMesh(Point(0.0, 0.0), Point(10., 1.0), nx, ny,"right")
    # boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    # boundaries.set_all(0)
    # Top = CompiledSubDomain("x[1]==side && on_boundary",side=1.)
    # Top.mark(boundaries,1)
    # Bottom = CompiledSubDomain("x[1]==side && on_boundary",side=0.)
    # Bottom.mark(boundaries,2)
    # Left = CompiledSubDomain("x[0]==side && on_boundary",side=0.)
    # Left.mark(boundaries,3)
    # Right = CompiledSubDomain("x[0]==side && on_boundary",side=10.)
    # Right.mark(boundaries,4)
    ## define measure based on mesh---------------------------------------
    info("hmin= %e, hmax = %e" % (mesh.hmin() , mesh.hmax()))

    dx = Measure('dx',domain=mesh)
    ds = Measure('ds',domain=mesh, subdomain_data=boundaries)

    ## Basis and function space============================================
    Dim = mesh.topology().dim()
    Vel = VectorElement("Lagrange", mesh.ufl_cell(), 2) #P2 
    P   = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #P1
    element = MixedElement([Vel, P])

    M_ns = FunctionSpace(mesh, element)
    dm_ns = TrialFunction(M_ns)
    ddm_ns= TestFunction(M_ns)
    # tuple Trial Function of velocity and pressure = split(m)
    (du,dp) = split(dm_ns)
    # tuple Test Function of velocity and pressure = split(dm)
    (ddv,ddp) = split(ddm_ns)

    m_ns   = Function(M_ns)
    # Define the Zero initial condition
    m0_ns  = Function(M_ns)
    # for second order time discretization
    m00_ns =Function (M_ns)

    m_relax   = Function(M_ns)   ##end of step velocities
    m0_relax  = Function(M_ns)
    m00_relax = Function(M_ns)
    ## Define Leray Problem now for filtering ===============================
        # pdb.set_trace()
    # delta = 0.1
    delta = mesh.hmin() ## filter radius

    # if filterOn:
    M_leray = FunctionSpace(mesh,element)
    # Trial and test function for leray
    dm_leray   = TrialFunction(M_leray)
    ddm_leray  = TestFunction(M_leray)
    # tuple Trial Function of velocity and pressure = split(m)
    (dv_leray,dp_leray)   = split(dm_leray)
    # tuple Test Function of velocity and pressure =split(dm)
    (ddv_leray,ddp_leray) = split(ddm_leray)

    m_l     = Function(M_leray)
    m0_l    = Function(M_leray)

        # if sensitivityOn:
    m_ns_delta      = Function(M_ns)
    m0_ns_delta     = Function(M_ns)
    m00_ns_delta    = Function(M_ns)

    m_leray_delta   = Function(M_leray)
    m0_leray_delta  = Function(M_leray)
    m00_leray_delta = Function(M_leray)

    # initizalize numpy array for lift and drag 
    lift_array  = np.empty((0,2),float)
    drag_array  = np.empty((0,2),float)
    pDiff_array = np.empty((0,2),float)

    # Strain Rate
    D = lambda v : (grad(v).T + grad(v)) / 2
    # Spin tensor
    Spin = lambda v: (grad(v) - grad(v).T)/2

    ## Boundary conditions===============================================
    t = 0.0
    inflow_profile = Expression(('4*Um*(x[1]*(ymax-x[1]))*sin(pi*t/8.0)/(ymax*ymax)', '0'), ymax=0.41,Um = 1.5, t = t, degree=2)


    bcs = [DirichletBC(M_ns.sub(0), inflow_profile  , boundaries, 5 )]
    bcs.append(DirichletBC(M_ns.sub(0),Constant(tuple([0.] * Dim)),boundaries, 2 )) 
    bcs.append(DirichletBC(M_ns.sub(0),Constant(tuple([0.] * Dim)),boundaries, 4 ))
    bcs.append(DirichletBC(M_ns.sub(0),Constant(tuple([0.] * Dim)),boundaries, 6 )) 
    bcs.append(DirichletBC(M_ns.sub(0),Constant(tuple([0.] * Dim)),boundaries, 7 )) 
    bcs.append(DirichletBC(M_ns.sub(0),Constant(tuple([0.] * Dim)),boundaries, 8 )) 
    bcs.append(DirichletBC(M_ns.sub(0),Constant(tuple([0.] * Dim)),boundaries, 9 ))

    # if filterOn:
    bcs_l = [DirichletBC(M_leray.sub(0), inflow_profile  , boundaries, 5 )]
    bcs_l.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 2 )) 
    bcs_l.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 4 ))
    bcs_l.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 6 )) 
    bcs_l.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 7 )) 
    bcs_l.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 8 )) 
    bcs_l.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 9 ))
        # if sensitivityOn:
    bcs_delta =     [DirichletBC(M_ns.sub(0),Constant(tuple([0.] * Dim)),boundaries, 5 )]
    bcs_delta.append(DirichletBC(M_ns.sub(0),Constant(tuple([0.] * Dim)),boundaries, 2 )) 
    bcs_delta.append(DirichletBC(M_ns.sub(0),Constant(tuple([0.] * Dim)),boundaries, 4 ))
    bcs_delta.append(DirichletBC(M_ns.sub(0),Constant(tuple([0.] * Dim)),boundaries, 6 )) 
    bcs_delta.append(DirichletBC(M_ns.sub(0),Constant(tuple([0.] * Dim)),boundaries, 7 )) 
    bcs_delta.append(DirichletBC(M_ns.sub(0),Constant(tuple([0.] * Dim)),boundaries, 8 )) 
    bcs_delta.append(DirichletBC(M_ns.sub(0),Constant(tuple([0.] * Dim)),boundaries, 9 ))

    bcs_l_delta =     [DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 5 )]
    bcs_l_delta.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 2 )) 
    bcs_l_delta.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 4 ))
    bcs_l_delta.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 6 )) 
    bcs_l_delta.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 7 )) 
    bcs_l_delta.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 8 )) 
    bcs_l_delta.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 9 ))

    ## Weak forms=====================================================================================

    def NSsemiImplicit_BDF2(m0_ns,m00_ns,m0_relax,m00_relax,dt):

        v0, _ = split(m0_ns) 
        v00,_ = split(m00_ns)  ## time discretization is computed from previous NS velocities 

        vEnd0,_ = split(m0_relax) 
        vEnd00,_= split(m00_relax)
        v_star= 2*vEnd0 - vEnd00   ##convection filed is computed from previous end-of-step velocities

        # Residual form
        a_00 = + 1.5/dt * inner( ddv, rho  * du               ) * dx \
               + inner( ddv         , rho  * grad(du) * v_star) * dx \
               + inner( D(ddv)      , vis  * 2*D(du)          ) * dx

        a_01 = - inner( div(ddv)    , dp                      ) * dx

        a_10 = + inner( ddp         , div(du)                 ) * dx

        L = + 2. /dt * inner( ddv, rho * v0 ) * dx \
            - 0.5/dt * inner( ddv, rho * v00) * dx

        f = a_00 + a_01 + a_10 - L

        # PCD forms - need elaboration!
        mu = 1.5/dt * inner( ddv, rho * du ) * dx
        ap = inner(grad(ddp), grad(dp)) * dx
        mp = 1./vis * ddp * dp * dx
        kp = rho/vis * ddp * dot(grad(dp), v_star) * dx
        gp = a_01

        return f, mu, ap, mp, kp, gp

    def lerayFilter(m_ns, indicator):
        u_ns, _ = split(m_ns)
        leray_00  = + inner(D(ddv_leray)  , (delta**2)*indicator*2*D(dv_leray))*dx \
                    + inner(ddv_leray     , dv_leray                          )*dx

        leray_01  = - inner(div(ddv_leray), dp_leray                          )*dx 

        leray_10  = + inner(ddp_leray     , div(dv_leray)                     )*dx

        leray_rhs = + inner(ddv_leray     ,u_ns)*dx

        f_leray   = leray_00 + leray_01 + leray_10 - leray_rhs

        # # mu_leray = inner(ddv_leray, dv_leray)*dx 
        # ap_leray = inner(grad(ddp_leray), grad(dp_leray)) * dx
        # mp_leray = 1./((delta**2)*indicator) * ddp * dp * dx
        # kp_leray = rho/((delta**2)*indicator)* ddp * dot(grad(dp), u_ns) * dx
        # gp_leray = leray_01
        p_leray = + inner(grad(ddv_leray), grad(dv_leray))*dx\
                  + ddp_leray*dp_leray*dx
        # return f_leray, mu_leray, ap_leray, mp_leray, kp_leray, gp_leray
        return f_leray, p_leray

    def localSensitivityDelta_NS(m_ns, m0_relax, m00_relax, m0_ns_delta, m00_ns_delta, m0_leray_delta, m00_leray_delta, dt):
        p_leray = inner(grad(ddv_leray), grad(dv_leray))    ## same functionSpace used for sensitivity as forward NS
    ## TODO: reuse the matrix from NS, only change assemble the L(rhs)
        v_ns,_  = split(m_ns)
        v0, _   = split(m0_relax)  ## end-of-step velocity
        v00, _  = split(m00_relax)
        v_star  = 2 * v0 - v00  
        # pdb.set_trace()
        v0_delta, _  = split(m0_ns_delta) # deepcopy.
        v00_delta, _ = split(m00_ns_delta)
        # vf_star      = 2 * v0_delta - v00_delta

        vf0_delta, _  = split(m0_leray_delta) # deepcopy.
        vf00_delta, _ = split(m00_leray_delta)
        vf_delta_star = 2 * vf0_delta - vf00_delta
        # Residual form ## need to double check the convection term
        a_00 = + 1.5/dt * inner( ddv, rho * du                ) * dx \
               + inner( ddv         , rho * grad(du) * v_star ) * dx \
               + inner( D(ddv)      , vis * 2*D(du)           ) * dx  

        a_01 = - inner( div(ddv)    , dp                   ) * dx

        a_10 = + inner( ddp         , div(du)              ) * dx

        rhs_delta = + 2./dt  * inner( ddv, rho * v0_delta          ) * dx \
                    - 0.5/dt * inner( ddv, rho * v00_delta         ) * dx \
                    - inner( ddv , rho * grad(v_ns) * vf_delta_star) * dx  ## double check


        f_delta = a_00 + a_01 + a_10 - rhs_delta
        # PCD forms - need elaboration!
        # mu = 1.5/dt * inner( ddv, rho * du ) * dx
        # ap = inner(grad(ddp), grad(dp)) * dx
        # mp = 1./nu * ddp * dp * dx
        # kp = rho/nu * ddp * dot(grad(dp), v_star) * dx
        # gp = a_01
        return f_delta
        # return f_delta, mu, ap, mp, kp, gp
        # return rhs_delta

    def localSensitivityDelta_leray(m_ns, m_ns_delta, indicator, indicator_delta, m_l):
    ## same functionSpace used for sensitivity as forward NS
        u, _       = split(m_ns)
        u_delta, _ = split(m_ns_delta)
        uf, _      = split(m_l)

        leray_00   = + inner(D(ddv_leray), (delta**2) * indicator* 2*D(dv_leray)) * dx \
                     + inner(ddv_leray,dv_leray)*dx  

        leray_01   = - inner(div(ddv_leray),dp_leray)*dx 
        leray_10   = + inner(ddp_leray, div(dv_leray))*dx

        rhs_leray_delta  = + inner(ddv_leray,u_delta) * dx\
                           - inner(D(ddv_leray), 2* delta * indicator * 2*D(uf)) * dx\
                           - inner(D(ddv_leray), delta**2 * indicator_delta * 2*D(uf)) * dx\
                           - inner(D(ddv_leray), delta*2 * indicator * 2*D(uf)) * ds(3) \
                           - inner(D(ddv_leray), delta**2 * indicator_delta * 2*D(uf)) * ds(3) ## neumann bc (assume the NS has zero-stress on the outlet)

        f_leray_delta = leray_00 + leray_01 + leray_10 - rhs_leray_delta
        mu_leray = inner(ddv_leray,dv_leray)*dx 
        # ap_leray = inner(grad(ddp_leray), grad(dp_leray)) * dx
        # mp_leray = 1./nu * ddp * dp * dx
        # kp_leray = rho/nu * ddp * dot(grad(dp), v_star) * dx
        # gp_leray = leray_01
        # return rhs_leray_delta
        return f_leray_delta

    ## forms ====================================================================================
    F, mu, ap, mp, kp, gp = NSsemiImplicit_BDF2(m0_ns,m00_ns,m0_relax,m00_relax,dt)
    a, R = lhs(F), rhs(F)


    v, _ = m_ns.split(True)

    deconvolution = deconvolution_EFR(N=deconOrder, delta=delta, velocity=v, boundaryMesh = boundaries,comm = comm)
    indicator = deconvolution.computeIndicator(option=indicatorOption)

    # F_leray = lerayFilter(m_ns, indicator)
    # F_leray, mu_leray, ap_leray, mp_leray, kp_leray, gp_leray = lerayFilter(m_ns, indicator)
    # F_leray, p_form_leray = lerayFilter(m_ns, indicator)
    # a_leray, R_leray = lhs(F_leray), rhs(F_leray)
    #         # if sensitivityOn:
    F_delta = localSensitivityDelta_NS(m_ns, m0_relax, m00_relax, m0_ns_delta, m00_ns_delta, m0_leray_delta, m00_leray_delta, dt)
    a_delta, R_delta = lhs(F_delta), rhs(F_delta)
    # v_delta,_ = m_ns_delta.split(True)
    # indicator_delta = deconvolution.computeIndicatorlocalSensitivity(v_delta)
    # F_leray_delta = localSensitivityDelta_leray(m_ns, m_ns_delta, indicator, indicator_delta, m_l)

    ## setup linear solver for generalized stokes====================================================================
    if has_krylov_solver_method("minres"):
        krylov_method = "minres"
    solver_stokes = KrylovSolver(krylov_method, "amg")
    solver_stokes.parameters["monitor_convergence"] = True
    solver_stokes.parameters["relative_tolerance"] = 1e-9

    ## setup linear solver====================================================================
    null_space = build_nullspace(M_ns)
    solver = create_pcd_solver(mesh.mpi_comm(), "BRM1", "direct")
    # Add another options for pcd solver if you wish
    prefix = solver.get_options_prefix()
    PETScOptions.set(prefix+"ksp_monitor")
    solver.set_from_options()
    # bc for pcd??????
    bcs_pcd_ns = DirichletBC(M_ns.sub(1), 0.0, boundaries, 5)
    bcs_pcd_leray = DirichletBC(M_ns.sub(1), 0.0, boundaries, 5)
    bcs_pcd_delta = DirichletBC(M_ns.sub(1), 0.0, boundaries, 5)

    pcd_assembler_NS = PCDAssembler(
        a,
        R,
        bcs,
        gp=gp,
        ap=ap,
        kp=kp,
        mp=mp,
        mu=mu,
        bcs_pcd=bcs_pcd_ns)

    # pcd_assembler_leray = PCDAssembler(
    #     a_leray,
    #     R_leray,
    #     bcs_l,
    #     gp=gp_leray,
    #     ap=ap_leray,
    #     kp=kp_leray,
    #     mp=mp_leray,
    #     mu=mu_leray,
    #     bcs_pcd=bcs_pcd_leray)
    # if sensitivityOn:
    pcd_assembler_NS_delta = PCDAssembler(
        a_delta,
        R_delta,
        bcs_delta,
        gp=gp,
        ap=ap,
        kp=kp,
        mp=mp,
        mu=mu,
        bcs_pcd=bcs_pcd_delta)


    ## time interation====================================================
    _init_pcd_flag = False
    for i in range(NT):
        t = i*dt
        if MPI.rank(comm) == 0:
            print("time =", t,"computing NS==============================================")
        inflow_profile.t = t
        A, b = PETScMatrix(mesh.mpi_comm()), PETScVector(mesh.mpi_comm())
        pcd_assembler_NS.system_matrix(A)
        pcd_assembler_NS.rhs_vector(b)
        #solve(A,m_ns.vector(),b)
        P = A # you have the possibility to use P != A
        solver.set_operators(A, P)
        if not _init_pcd_flag:
            solver.init_pcd(pcd_assembler_NS)
            _init_pcd_flag = True
        solver.solve(m_ns.vector(), b)

        # extract solution of NS
        v, p = m_ns.split(True) # deepcopy.

        # TODO: put parameter for do not save mesh at each time step.
        # write output======================================================
        v.rename('vel', 'vel')
        p.rename('press', 'press')
        # TODO: put parameter for do not save mesh at each time step.
        file_handler_press.write(p, float(t))
        file_handler_vel.write(  v,  float(t))
        
        ## compute indicator function
        deconvolution.vel = v
        indicator = deconvolution.computeIndicator(option=indicatorOption)  ## memory leak?
        file_handler_indicator.write(indicator, float(t))


        if filterOn:
            if MPI.rank(comm) == 0:
                print("time =", t,"computing leray========================================")
            F_leray, p_form_leray = lerayFilter(m_ns, indicator)
            # F_leray = lerayFilter(m_ns, indicator)
            # A_leray, b_leray = PETScMatrix(mesh.mpi_comm()), PETScVector(mesh.mpi_comm())
            # pcd_assembler_leray.system_matrix(A_leray)
            # pcd_assembler_leray.rhs_vector(b_leray)
            ##solve(A,m_ns.vector(),b)
            A_leray, b_leray = assemble_system(lhs(F_leray), rhs(F_leray),bcs_l)
            P_leray = A_leray # you have the possibility to use P != A
            solver_stokes.set_operators(A_leray, P_leray)
            # if not _init_pcd_flag:
            #     solver.init_pcd(pcd_assembler_leray)
            #     _init_pcd_flag = True
            # solver.solve(m_l.vector(), b_leray)
            solver_stokes.solve(m_l.vector(), b_leray)
            # solve(lhs(F_leray) == rhs(F_leray), m_l, bcs_l ) 
            v_leray, p_leray = m_l.split(True) # deepcopy.

            v_leray.rename('vel_leray'  , 'vel_leray')
            p_leray.rename('press_leray', 'press_leray')
            # TODO: put parameter for do not save mesh at each time step.
            file_handler_press_leray.write(p_leray,float(t))
            file_handler_vel_leray.write(v_leray,  float(t))
     
            assignerVel = FunctionAssigner(v.function_space(), v_leray.function_space())   ## memory leak?
            assignerPress = FunctionAssigner(p.function_space(), p_leray.function_space())
            v_tmp = Function(v.function_space())
            p_tmp = Function(p.function_space())
            assignerVel.assign(v_tmp, v_leray)
            assignerPress.assign(p_tmp, p_leray)

            v_relax = Function(m_relax.sub(0).function_space().collapse())
            p_relax = Function(m_relax.sub(1).function_space().collapse())
            v_relax.assign((1 - chi) * v + chi * v_tmp)
            p_relax.assign(p + 1.5 * chi * p_tmp)
            assign(m_relax,[v_relax, p_relax])
            # assign(m_relax,[v, p])
            v_relax.rename('vel_relax', 'vel_relax')
            p_relax.rename('press_relax', 'press_relax')
            file_handler_press_relax.write(p_relax,float(t))
            file_handler_vel_relax.write(v_relax,  float(t))

            if sensitivityOn:
                if MPI.rank(comm) == 0:
                    print("time =", t, "  computing sensitivity of NS======================================================")
                # F_delta = localSensitivityDelta_NS(m_ns, m0_relax, m00_relax, m0_ns_delta, m00_ns_delta, m0_leray_delta, m00_leray_delta, dt)
                # a_delta, R_delta = lhs(F_delta), rhs(F_delta)

                # solve(lhs(F_delta) == rhs(F_delta), m_ns_delta, bcs_delta ) 
                A_delta, b_delta = PETScMatrix(mesh.mpi_comm()), PETScVector(mesh.mpi_comm())
                pcd_assembler_NS_delta.system_matrix(A_delta)
                pcd_assembler_NS_delta.rhs_vector(b_delta)
                P_delta = A_delta # you have the possibility to use P != A
                solver.set_operators(A_delta, P_delta)
                if not _init_pcd_flag:
                    solver.init_pcd(pcd_assembler_NS_delta)
                    _init_pcd_flag = True
                solver.solve(m_ns_delta.vector(), b_delta)

                v_delta, p_delta = m_ns_delta.split(True)
                v_delta.rename('vel_delta','vel_delta')
                p_delta.rename('pressure_delta','pressure_delta')
                file_handler_press_delta.write(p_delta,float(t))
                file_handler_vel_delta.write(v_delta,  float(t))
                if MPI.rank(comm) == 0:
                    print("norm of v_delta =",norm(v_delta))   ##checking if the sensitivity is zero

                if MPI.rank(comm) == 0:
                    print("time =", t, "  computing sensitivity of leray=====================================================")
                indicator_delta = deconvolution.computeIndicatorlocalSensitivity(v_delta) 
                indicator_delta.rename('indicator_delta','indicator_delta')
                file_indicator_delta.write(indicator_delta,float(t))

                F_leray_delta = localSensitivityDelta_leray(m_ns, m_ns_delta, indicator, indicator_delta, m_l)
                # solve(lhs(F_leray_delta) == rhs(F_leray_delta), m_leray_delta, bcs_l_delta) 
                A_leray_delta, b_leray_delta = assemble_system(lhs(F_leray_delta), rhs(F_leray_delta),bcs_l_delta)
                P_leray_delta = A_leray_delta # you have the possibility to use P != A
                solver_stokes.set_operators(A_leray_delta, P_leray_delta)
                solver_stokes.solve(m_leray_delta.vector(), b_leray_delta)


                vf_delta, pf_delta = m_leray_delta.split(True)
                vf_delta.rename('vel_leray_delta','vel_leray_delta')
                pf_delta.rename('pressure_leray_delta','pressure_leray_delta')
                file_handler_press_leray_delta.write(pf_delta,float(t))
                file_handler_vel_leray_delta.write(vf_delta,  float(t))
                

        
        ## finalize the timestep

        if filterOn:
            m00_ns.assign(m0_ns)
            m0_ns.assign(m_ns)
            m00_relax.assign(m0_relax)
            m0_relax.assign(m_relax)

            if sensitivityOn:
                m00_ns_delta.assign(m0_ns_delta)
                m0_ns_delta.assign(m_ns_delta)
                m00_leray_delta.assign(m0_leray_delta)
                m0_leray_delta.assign(m_leray_delta)
                

            # lift, drag, p_diff = computeDragLift( mesh, ds, v, p, vis)

        else:
            m00_ns.assign(m0_ns)
            m0_ns.assign(m_ns)
            m00_relax.assign(m0_ns)
            m0_relax.assign(m_ns)
            # lift, drag, p_diff = computeDragLift( mesh, ds, v, p, vis)

        del assignerVel, assignerPress
        del F_leray, F_leray_delta
        # info("drag= %e, lift= %e, p_diff = %e" % (drag , lift, p_diff))
        # lift_array = np.vstack((lift_array,np.array([float(t),lift])))
        # drag_array = np.vstack((drag_array,np.array([float(t),drag])))
        # pDiff_array = np.vstack((pDiff_array,np.array([t,p_diff])))
        ## update solution for next time step
if __name__ == "__main__":
    comm = MPI.comm_world
    dt  = 0.0025
    NT  = 5
    rho = 1.
    vis  = 0.001    ##dynamic viscousity
    indicatorOption = 'l2Local' #'DG'or 'l2Local'
    deconOrder = 0
    filterOn = True
    chi = 1.
    sensitivityOn = True
    meshPath="mesh/mesh_fine.xdmf"
    boundaryMeshPath="mesh/facet_fine.xdmf"
    NS_leray_localSensitivity(comm,meshPath, boundaryMeshPath,dt, NT, rho, vis, indicatorOption, deconOrder, filterOn,sensitivityOn, chi)
    list_timings(TimingClear.clear, [TimingType.wall])
    cpp.common.monitor_memory_usage()
   # np.savetxt(liftFile,lift_array,fmt='%1.5f,%1.7e',header="time,lift_simu",comments='')
   # np.savetxt(dragFile,drag_array,fmt='%1.5f,%1.7e',header="time,drag_simu",comments='')
   # np.savetxt(pressDiffFile,pDiff_array,fmt='%1.5f,%1.7e',header="time,pressDiff_simu",comments='')

