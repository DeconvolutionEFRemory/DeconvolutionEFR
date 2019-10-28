from dolfin import *
import pdb

__all__=['Indicator']

_indicator_cpp = """
#include <pybind11/pybind11.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/FunctionSpace.h>
using namespace dolfin;
class Indicator : public Expression
{
  public:
  std::shared_ptr<GenericFunction> velocity;
  Indicator() : Expression() { }
  void eval(Array<double>& values, const Array<double>& x, const ufc::cell& c) const
  {
    // Get dolfin cell and its diameter
    dolfin_assert(velocity->function_space());
    const std::shared_ptr<const Mesh> mesh = velocity->function_space()->mesh();
    const Cell cell(*mesh, c.index);
    double h = cell.h();
    // Compute l2 norm of wind
    double wind_norm = 0.0, velocity_norm = 0.0;
    // FIXME: Avoid dynamic allocation
    Array<double> v(velocity->value_size());
    velocity->eval(v, x, c);
    for (uint i = 0; i < v.size(); ++i)
        wind_norm += v[i]*v[i];
    velocity_norm = sqrt(wind_norm);
    values[0] = velocity_norm;

  }
};
PYBIND11_MODULE(SIGNATURE, m)
{
  pybind11::class_<Indicator,
             std::shared_ptr<Indicator>,
             Expression> 
             (m, "Indicator")
    .def(pybind11::init<>())
    .def_readwrite("velocity", &Indicator::velocity);
}
"""

_expr = compile_cpp_code(_indicator_cpp).Indicator



def Indicator(velocity):
    """Returns a subclass of :py:class:`dolfin.Expression` representing
    the indicator
    *Arguments*
        velocity (:py:class:`dolfin.GenericFunction`)
            A vector field determining convective velocity.
    """
    mesh = velocity.function_space().mesh()
    element = FiniteElement("DG", mesh.ufl_cell(), 0)
    indicator = CompiledExpression(_expr(), element=element, domain=mesh) ## bug: return nonType
    # pdb.set_trace()
    indicator.velocity = velocity._cpp_object

    # file_handler_press = XDMFFile('output/testExpression.xdmf')
    # file_handler_press.write(indicator)
    return indicator
