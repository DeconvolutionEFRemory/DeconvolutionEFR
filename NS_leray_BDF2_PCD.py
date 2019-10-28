import numpy as np
from dolfin import *
from fenapack import PCDKrylovSolver, PCDAssembler
import pdb
from pcdSetup import *
from dragLift import *
from deconvolution_EFR import *

comm = MPI.comm_world
## basic setup======================================================
dt  = 0.0025
NT  = 3201
rho = 1.
nu  = 0.001
lift_array  = np.empty((0,2),float)
drag_array  = np.empty((0,2),float)
pDiff_array = np.empty((0,2),float)


indicatorOption = 'l2Local' #'DG'or 'l2Local'
deconOrder = 0
filterOn = True

file_handler_press = XDMFFile(comm,'output/press_NS_BDF2_pcd.xdmf')
file_handler_vel   = XDMFFile(comm,'output/vel_NS_BDF2_pcd.xdmf')
csvDir = "csv/NS_BDF2_PCD/"
liftFile =csvDir + "lift_simu_semiImplicitBDF2_pcd.csv"
dragFile =csvDir + "drag_simu_semiImplicitBDF2_pcd.csv"
pressDiffFile =csvDir + "pressDiff_simu_semiImplicitBDF2_pcd.csv"
file_handler_indicator = XDMFFile(comm,'output/indicator_'+indicatorOption+'_N'+str(deconOrder)+'.xdmf')

if filterOn:
    file_handler_vel_bar   = XDMFFile(comm,'output/vel_bar_leray_BDF2_pcd.xdmf')
    file_handler_press_bar   = XDMFFile(comm,'output/press_bar_leray_BDF2_pcd.xdmf')
    file_handler_vel_relax   = XDMFFile(comm,'output/vel_relax_BDF2_pcd.xdmf')
    file_handler_press_relax   = XDMFFile(comm,'output/press_relax_BDF2_pcd.xdmf')
## Mesh===============================================================
## read mesh file-----------------------------------------------------
meshPath="mesh/mesh_fine.xdmf"
boundaryMeshPath="mesh/facet_fine.xdmf"
mesh = Mesh(comm)
with XDMFFile(comm, meshPath) as xdmf:
     xdmf.read(mesh)
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
with XDMFFile(comm, boundaryMeshPath) as xdmf:
     xdmf.read(boundaries)
## create toy mesh inside-------------------------------------------
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
P = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #P1
element = MixedElement([Vel, P])

M_ns = FunctionSpace(mesh, element)
dm_ns = TrialFunction(M_ns)
ddm_ns= TestFunction(M_ns)
# tuple Trial Function of velocity and pressure = split(m)
(du,dp) = split(dm_ns)
# tuple Test Function of velocity and pressure = split(dm)
(ddv,ddp) = split(ddm_ns)

m_ns = Function(M_ns)
# Define the Zero initial condition
m0_ns = Function(M_ns)
# for second order time discretization
m00_ns =Function (M_ns)

## Define Leray Problem now for filtering ===============================
    # pdb.set_trace()
# delta = 0.1
delta = mesh.hmin()

if filterOn:
 
    N = 0
    chi = 0.1
    m_relax = Function(M_ns)
    M_leray = FunctionSpace(mesh,element)
    dm_l = TrialFunction(M_leray)
    ddm_l= TestFunction(M_leray)
    m_l  = Function(M_leray)
    # Trial and test function for leray
    # tuple Trial Function of velocity and pressure = split(m)
    (dv_bar,dp_bar) = split(dm_l)
    # tuple Test Function of velocity and pressure =split(dm)
    (ddv_bar,ddp_bar) = split(ddm_l)



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

if filterOn:
    bcs_l = [DirichletBC(M_leray.sub(0), inflow_profile  , boundaries, 5 )]
    bcs_l.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 2 )) 
    bcs_l.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 4 ))
    bcs_l.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 6 )) 
    bcs_l.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 7 )) 
    bcs_l.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 8 )) 
    bcs_l.append(DirichletBC(M_leray.sub(0),Constant(tuple([0.] * Dim)),boundaries, 9 ))

#BDF2 NS Unsteady

def NSsemiImplicit_BDF2(m0_ns,m00_ns,dt):

    v0, _ = split(m0_ns) # deepcopy.
    v00,_ = split(m00_ns)
    v_star= 2*v0 - v00

    # Residual form
    a_00 = + 1.5/dt * inner( ddv, rho * du ) * dx \
           + inner( ddv         , rho * grad(du) * v_star   ) * dx \
           + inner( D(ddv)      , 2*nu * D(du)         ) * dx

    a_01 = - inner( div(ddv)    , dp                   ) * dx

    a_10 = + inner( ddp         , div(du)              ) * dx

    L = + 2./dt * inner( ddv, rho * v0 ) * dx \
        - 0.5/dt * inner( ddv, rho * v00 ) * dx

    f = a_00 + a_01 + a_10 - L

    # PCD forms - need elaboration!
    mu = 1.5/dt * inner( ddv, rho * du ) * dx
    ap = inner(grad(ddp), grad(dp)) * dx
    mp = 1./nu * ddp * dp * dx
    kp = rho/nu * ddp * dot(grad(dp), v_star) * dx
    gp = a_01

    return f, mu, ap, mp, kp, gp

def lerayFilter(m_ns, indicator):
    u, _ = split(m_ns)
    leray_00  = + inner(D(ddv_bar), 2*delta**2*indicator*D(dv_bar))*dx \
                + inner(ddv_bar,dv_bar)*dx 
    leray_01  = - inner(div(ddv_bar),dp_bar)*dx 
    leray_10  = + inner(ddp_bar, div(dv_bar))*dx

    leray_rhs = + inner(ddv_bar,u)*dx

    f_leray = leray_00 + leray_01 + leray_10 - leray_rhs

    # mu_leray = inner(ddv_bar,dv_bar)*dx 
    # ap_leray = inner(grad(ddp_bar), grad(dp_bar)) * dx
    # mp_leray = 1./nu * ddp * dp * dx
    # kp_leray = rho/nu * ddp * dot(grad(dp), v_star) * dx
    # gp_leray = leray_01
    return f_leray

## forms and linear system=========================================
F, mu, ap, mp, kp, gp = NSsemiImplicit_BDF2(m0_ns,m00_ns,dt)
a, L = lhs(F), rhs(F)

## setup linear solver=============================================
null_space = build_nullspace(M_ns)
solver = create_pcd_solver(mesh.mpi_comm(), "BRM1", "direct")
# Add another options for pcd solver if you wish
prefix = solver.get_options_prefix()
PETScOptions.set(prefix+"ksp_monitor")
solver.set_from_options()
# bc for pcd??????
bcs_pcd = DirichletBC(M_ns.sub(1), 0.0, boundaries, 5)
pcd_assembler = PCDAssembler(
    a,
    L,
    bcs,
    gp=gp,
    ap=ap,
    kp=kp,
    mp=mp,
    mu=mu,
    bcs_pcd=bcs_pcd)
v, _ = m_ns.split(True)
deconvolution = deconvolution_EFR(N=deconOrder, delta=delta, velocity=v, boundaryMesh = boundaries)
indicator = deconvolution.compute_indicator(option=indicatorOption)

if filterOn:

    F_leray = lerayFilter(m_ns, indicator)

# if filterOn:

v, p = m_ns.split(True) # deepcopy.
# v_bar, p_bar = m_l.split(True) # deepcopy.


v_relax=Function(v.function_space())
p_relax=Function(p.function_space())

## time interation====================================================
_init_pcd_flag = False
for i in range(NT):
    t = i*dt
    print("time =", t,"======================================================")
    inflow_profile.t = t
    A, b = PETScMatrix(mesh.mpi_comm()), PETScVector(mesh.mpi_comm())
    pcd_assembler.system_matrix(A)
    pcd_assembler.rhs_vector(b)

    #solve(A,m_ns.vector(),b)
    P = A # you have the possibility to use P != A
    solver.set_operators(A, P)
    if not _init_pcd_flag:
        solver.init_pcd(pcd_assembler)
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
    indicator = deconvolution.compute_indicator(option=indicatorOption)  ## memory leak?
    file_handler_indicator.write(indicator, float(t))


    if filterOn:
        # deconvolution.vel = v
        # indicator = deconvolution.compute_indicator()  ## memory leak?
        # indicator.rename('a_deconv', 'a_deconv')
        # file_handler_indicator.write(indicator, float(t))

        solve(lhs(F_leray) == rhs(F_leray), m_l, bcs_l ) 
        v_bar, p_bar = m_l.split(True) # deepcopy.

        v_bar.rename('vel_bar', 'vel_bar')
        p_bar.rename('press_bar', 'press_bar')
        # TODO: put parameter for do not save mesh at each time step.
        file_handler_press_bar.write(p_bar,float(t))
        file_handler_vel_bar.write(v_bar,  float(t))
 
        assignerVel = FunctionAssigner(v.function_space(), v_bar.function_space())   ## memory leak?
        assignerPress = FunctionAssigner(p.function_space(), p_bar.function_space())
        v_tmp = Function(v.function_space())
        p_tmp = Function(p.function_space())
        assignerVel.assign(v_tmp, v_bar)
        assignerPress.assign(p_tmp, p_bar)

        # v_relax = Function(m_relax.sub(0).function_space().collapse())
        v_relax.assign((1 - chi) * v + chi * v_tmp)
        p_relax.assign(p + 1.5 * chi * p_tmp)
        assign(m_relax,[v_relax, p_relax])
        # assign(m_relax,[v, p])
        v_relax.rename('vel_relax', 'vel_relax')
        p_relax.rename('press_relax', 'press_relax')
        file_handler_press_relax.write(p_relax,float(t))
        file_handler_vel_relax.write(v_relax,  float(t))

    
    m00_ns.assign(m0_ns)

    if filterOn:
        m0_ns.assign(m_relax)
        lift, drag, p_diff = computeDragLift( mesh, ds, v_relax, p_relax, nu)

    else:
        m0_ns.assign(m_ns)
        lift, drag, p_diff = computeDragLift( mesh, ds, v, p, nu)

    info("drag= %e, lift= %e, p_diff = %e" % (drag , lift, p_diff))
    lift_array = np.vstack((lift_array,np.array([float(t),lift])))
    drag_array = np.vstack((drag_array,np.array([float(t),drag])))
    pDiff_array = np.vstack((pDiff_array,np.array([t,p_diff])))
    ## update solution for next time step
if __name__ == "__main__":
   list_timings(TimingClear.clear, [TimingType.wall])
   np.savetxt(liftFile,lift_array,fmt='%1.5f,%1.7e',header="time,lift_simu",comments='')
   np.savetxt(dragFile,drag_array,fmt='%1.5f,%1.7e',header="time,drag_simu",comments='')
   np.savetxt(pressDiffFile,pDiff_array,fmt='%1.5f,%1.7e',header="time,pressDiff_simu",comments='')

