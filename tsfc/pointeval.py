from __future__ import absolute_import

import collections
import itertools

import numpy
import sympy
from singledispatch import singledispatch

from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.classes import Coefficient, FormArgument, GeometricQuantity

import gem
from gem.node import Memoizer

from tsfc.modified_terminals import analyse_modified_terminal
from tsfc import geometric, ufl2gem
from tsfc.ufl_utils import (CollectModifiedTerminals,
                            ModifiedTerminalMixin, simplify_abs)

from tsfc.fem import _tabulate


def tabulate(ufl_element, order):
    X, Y, Z = map(sympy.Symbol, ['X', 'Y', 'Z'])
    dim = ufl_element.cell().topological_dimension()
    points = numpy.array([(X, Y, Z)[:dim]], dtype=object)
    for c, D, table in _tabulate(ufl_element, order, points):
        table, = table
        yield c, D, table


class TabulationManager(object):

    def __init__(self):
        self.evaluate_basis = {}

    def tabulate(self, ufl_element, max_deriv):
        """Prepare the tabulations of a finite element up to a given
        derivative order.

        :arg ufl_element: UFL element to tabulate
        :arg max_deriv: tabulate derivatives up this order
        """
        for c, D, table in tabulate(ufl_element, max_deriv):
            assert len(table.shape) == 1
            self.evaluate_basis[(ufl_element, c, D)] = table

    def __getitem__(self, key):
        return self.evaluate_basis[key]


class Translator(MultiFunction, ModifiedTerminalMixin, ufl2gem.Mixin):
    """Contains all the context necessary to translate UFL into GEM."""

    def __init__(self, tabulation_manager, reference_point, coefficient_map, index_cache):
        MultiFunction.__init__(self)
        ufl2gem.Mixin.__init__(self)
        self.tabulation_manager = tabulation_manager
        self.reference_point = reference_point
        self.coefficient_map = coefficient_map
        self.index_cache = index_cache

        XYZ = map(sympy.Symbol, ['X', 'Y', 'Z'])
        dim, = reference_point.shape
        bindings = dict(zip(XYZ, [gem.Indexed(reference_point, (d,))
                                  for d in range(dim)]))
        self.sympy2gem = Memoizer(sympy2gem)
        self.sympy2gem.bindings = bindings

    def modified_terminal(self, o):
        """Overrides the modified terminal handler from
        :class:`ModifiedTerminalMixin`."""
        mt = analyse_modified_terminal(o)
        return translate(mt.terminal, mt, self)


def iterate_shape(mt, callback):
    """Iterates through the components of a modified terminal, and
    calls ``callback`` with ``(ufl_element, c, D)`` keys which are
    used to look up tabulation matrix for that component.  Then
    assembles the result into a GEM tensor (if tensor-valued)
    corresponding to the modified terminal.

    :arg mt: analysed modified terminal
    :arg callback: callback to get the GEM translation of a component
    :returns: GEM translation of the modified terminal

    This is a helper for translating Arguments and Coefficients.
    """
    ufl_element = mt.terminal.ufl_element()
    dim = ufl_element.cell().topological_dimension()

    def flat_index(ordered_deriv):
        return tuple((numpy.asarray(ordered_deriv) == d).sum() for d in range(dim))

    ordered_derivs = itertools.product(range(dim), repeat=mt.local_derivatives)
    flat_derivs = map(flat_index, ordered_derivs)

    result = []
    for c in range(ufl_element.reference_value_size()):
        for flat_deriv in flat_derivs:
            result.append(callback((ufl_element, c, flat_deriv)))

    shape = mt.expr.ufl_shape
    assert len(result) == numpy.prod(shape)

    if shape:
        return gem.ListTensor(numpy.asarray(result).reshape(shape))
    else:
        return result[0]


@singledispatch
def sympy2gem(node, self):
    raise AssertionError("sympy node expected, got %s" % type(node))


@sympy2gem.register(sympy.Expr)
def sympy2gem_expr(node, self):
    raise NotImplementedError("no handler for sympy node type %s" % type(node))


@sympy2gem.register(sympy.Add)
def sympy2gem_add(node, self):
    args = [self(arg) for arg in node.args]
    assert len(args) >= 2
    if len(args) == 2:
        return gem.Sum(*args)
    else:
        result = gem.Sum(args[0], args[1])
        for i in range(2, len(args)):
            result = gem.Sum(result, args[i])
        return result


@sympy2gem.register(sympy.Mul)
def sympy2gem_mul(node, self):
    args = [self(arg) for arg in node.args]
    assert len(args) >= 2
    if len(args) == 2:
        return gem.Product(*args)
    else:
        result = gem.Product(args[0], args[1])
        for i in range(2, len(args)):
            result = gem.Product(result, args[i])
        return result


@sympy2gem.register(sympy.Pow)
def sympy2gem_pow(node, self):
    return gem.Power(*[self(arg) for arg in node.args])


@sympy2gem.register(sympy.Integer)
def sympy2gem_integer(node, self):
    return gem.Literal(int(node))


@sympy2gem.register(sympy.Float)
def sympy2gem_float(node, self):
    return gem.Literal(float(node))


@sympy2gem.register(sympy.Symbol)
def sympy2gem_symbol(node, self):
    return self.bindings[node]


@singledispatch
def translate(terminal, mt, params):
    """Translates modified terminals into GEM.

    :arg terminal: terminal, for dispatching
    :arg mt: analysed modified terminal
    :arg params: translator context
    :returns: GEM translation of the modified terminal
    """
    raise AssertionError("Cannot handle terminal type: %s" % type(terminal))


@translate.register(GeometricQuantity)
def translate_geometricquantity(terminal, mt, params):
    return geometric.translate(terminal, mt, params)


@translate.register(Coefficient)
def translate_coefficient(terminal, mt, params):
    kernel_arg = params.coefficient_map[terminal]

    if terminal.ufl_element().family() == 'Real':
        assert mt.local_derivatives == 0
        return kernel_arg

    def callback(key):
        tabulation = params.tabulation_manager[key]
        basis_values = gem.ListTensor(map(params.sympy2gem, tabulation))

        r = params.index_cache[terminal.ufl_element()]
        return gem.IndexSum(gem.Product(gem.Indexed(basis_values, (r,)),
                                        gem.Indexed(kernel_arg, (r,))), r)

    return iterate_shape(mt, callback)


def process(expression, tabulation_manager, reference_point, coefficient_map, index_cache):
    # Abs-simplification
    expression = simplify_abs(expression)

    # Collect modified terminals
    modified_terminals = []
    map_expr_dag(CollectModifiedTerminals(modified_terminals), expression)

    # Collect maximal derivatives that needs tabulation
    max_derivs = collections.defaultdict(int)

    for mt in map(analyse_modified_terminal, modified_terminals):
        if isinstance(mt.terminal, FormArgument):
            ufl_element = mt.terminal.ufl_element()
            max_derivs[ufl_element] = max(mt.local_derivatives, max_derivs[ufl_element])

    # Collect tabulations for all components and derivatives
    for ufl_element, max_deriv in max_derivs.items():
        if ufl_element.family() != 'Real':
            tabulation_manager.tabulate(ufl_element, max_deriv)

    # Translate UFL to Einstein's notation,
    # lowering finite element specific nodes
    translator = Translator(tabulation_manager, reference_point,
                            coefficient_map, index_cache)
    return map_expr_dag(translator, expression)
