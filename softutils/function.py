#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 2021-11-29
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from threading import RLock
import typing as typ
import inspect
import numpy as np

ADDITION       = '+'
SUBTRACTION    = '-'
R_SUBTRACTION  = 'r-'
MULTIPLICATION = '*'
DIVISION       = '/'
R_DIVISION     = '\\'
SUPERPOSITION  = '@'
FROM_FUNC      = 'simple_func'
FROM_VAR       = 'var'
FROM_CONST     = 'const'

BINARY_OPERATORS = {ADDITION, SUBTRACTION, MULTIPLICATION, DIVISION,
                    R_DIVISION, R_SUBTRACTION}

UNARY_OPERATORS = {FROM_FUNC, FROM_VAR, FROM_CONST}

SPECIAL_OPERATORS = {SUPERPOSITION, }

OPERATORS = BINARY_OPERATORS | UNARY_OPERATORS | SPECIAL_OPERATORS

BINARY_DICT = {
    ADDITION:       lambda x, y: x + y,
    SUBTRACTION:    lambda x, y: x - y,
    MULTIPLICATION: lambda x, y: x * y,
    DIVISION:       lambda x, y: x / y,
    R_DIVISION:     lambda x, y: y / x,
    R_SUBTRACTION:  lambda x, y: y - x,
}

SIMPLE_DERIVATIVES = {
    ADDITION:       lambda u, v, u_der, v_der: u_der + v_der,
    SUBTRACTION:    lambda u, v, u_der, v_der: u_der - v_der,
    MULTIPLICATION: lambda u, v, u_der, v_der: u_der * v + u * v_der,
    DIVISION:       lambda u, v, u_der, v_der: (u_der * v - u * v_der) / (v ** 2),
    R_DIVISION:     lambda u, v, u_der, v_der: (v_der * u - v * u_der) / (u ** 2),
    R_SUBTRACTION:  lambda u, v, u_der, v_der: v_der - u_der
}

inf = float('inf')


def get_power(n):

    def _res(x):
        return x ** n

    return _res


ELEMENTARY_FUNCTIONS = {
    'sin':  (np.sin, -inf, inf),
    'cos':  (np.cos, -inf, inf),
    'sqrt': (np.sqrt, 0, inf),
    'tan':  (np.tan, -np.pi/2, np.pi/2),
    'log':  (np.log, 0, inf),
    'pow':  (get_power, -inf, inf),
    'exp':  (np.exp, -inf, inf),
    'sinh': (np.sinh, -inf, inf),
    'cosh': (np.cosh, -inf, inf),
}

LOCK = RLock()


class WTFError(Exception):

    def __str__(self):
        return "Such situation couldn't happen."


class NotInitialisedVarError(TypeError):

    def __init__(self, name):
        super(NotInitialisedVarError, self).__init__()
        self.name = name

    def __str__(self):
        return f"Call for not initialised variable: {self.name}."


class Var:

    __instances = {}

    def __new__(cls, name: str, left=float('-inf'), right=float('inf')):
        if name not in cls.__instances:
            cls.__instances[name] = super().__new__(cls)
        return cls.__instances[name]

    def __init__(self, name: str, left=float('-inf'), right=float('inf')):
        self.name = name
        self._value = None
        self._left = left
        self._right = right

    @property
    def value(self):
        return self()

    @value.setter
    def value(self, value: typ.Union[float, int]):
        self.set_value(value)

    def set_value(self, value: typ.Union[float, int]):
        with LOCK:
            self._value = value

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, left: typ.Union[float, int]):
        with LOCK:
            self._left = left

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, right: typ.Union[float, int]):
        with LOCK:
            self._right = right

    def __hash__(self):
        return hash(self.name)

    def __call__(self):
        if self._value is None:
            raise NotInitialisedVarError(self.name)
        return self._value

    def __eq__(self, other):
        if isinstance(other, Var):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            return False

    def __add__(self, other):
        return other.__radd__(self)

    def __mul__(self, other):
        return other.__radd__(self)

    def __sub__(self, other):
        return other.__rsup__(self)

    def __truediv__(self, other):
        return other.__rtruediv__(self)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Var('{self.name}')"


# TODO:
#      - add parser
#      - add domains for elementary functions
#      -

class Function:

    delta_x = 10e-5

    def __init__(self, main_op: str, variables: typ.Set[Var],
                 *sons: typ.Union[typ.Callable, float, int],
                 name='', str_repr='', check_signature=True, **superpos_sons):
        assert main_op in OPERATORS, 'bad operation'
        assert all(isinstance(var, Var) for var in variables), 'bad variable'
        assert not (len(sons) > 1 and main_op in UNARY_OPERATORS), 'bad amount of sons'
        assert not (len(sons) != 2 and main_op in BINARY_OPERATORS), 'bad amount of sons'
        assert not (main_op in BINARY_OPERATORS and
                    any(not isinstance(son, Function) for son in sons)), 'bad type of sons'

        assert not (len(sons) != 1 and main_op == SUPERPOSITION
                    and len(superpos_sons) == 0 ), 'bad superposition format'

        self._main_op = main_op
        self._vars = variables

        self._superpos_sons = superpos_sons
        self._sons = sons
        self._name = name
        self._str_repr = str_repr
        if self._main_op == FROM_FUNC and check_signature:
            self._check_function()

        if self._main_op == SUPERPOSITION:
            self._check_superposition()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name: str):
        with LOCK:
            self._name = new_name

    def set_str_repr(self, value: str):
        with LOCK:
            self._str_repr = value

    def _check_superposition(self):
        given_vars = set(self._superpos_sons.keys())
        func = self._sons[0]    # type: Function
        if given_vars != func._vars:
            raise ValueError(f"Bad set of variables are given: "
                             f"expected {func._vars}, got {self._superpos_sons}")

        sons_vars = set()
        for el in self._superpos_sons.values():
            if not isinstance(el, Function):
                raise TypeError(f"Element with type for superposition: {el}")
            sons_vars |= el._vars

        if sons_vars != self._vars:
            raise ValueError(f'Conflict between given and substitutions'
                             f'variables: given: {self._vars}, subs: {sons_vars}')

        if sons_vars & func._vars:
            raise ValueError(f"Super func should be free from substitutions' "
                             f"variables. Conflicts: {sons_vars & func._vars}")

    def _check_function(self):
        func = self._sons[0]    # type: typ.Callable
        signature = inspect.getfullargspec(func)
        if self._vars != set(signature.args):
            raise ValueError(f"Error checking function signature: "
                             f"expected {self._vars}, got {signature}")

    def _binary(self, other, operation_type):
        assert isinstance(other, Function) or isinstance(other, Var) \
            or isinstance(other, int) or isinstance(other, float), 'bad operand'
        assert operation_type in BINARY_OPERATORS, 'bad operation'

        if isinstance(other, Var):
            res_other = Function(FROM_VAR, {other}, other)
        elif isinstance(other, float) or isinstance(other, int):
            res_other = Function(FROM_CONST, set(), other)
        else:
            res_other = other
        return Function(operation_type, self._vars | res_other._vars,
                        self, res_other)

    def __add__(self, other):
        return self._binary(other, ADDITION)

    def __sub__(self, other):
        return self._binary(other, SUBTRACTION)

    def __mul__(self, other):
        return self._binary(other, MULTIPLICATION)

    def __truediv__(self, other):
        return self._binary(other, DIVISION)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self._binary(other, R_DIVISION)

    def __rsub__(self, other):
        return self._binary(other, R_SUBTRACTION)

    def substitute(self, **others):
        """ return f(...) = self(other(...))
        """
        given_pars = set(others.keys())
        if self._vars - given_pars:
            raise ValueError(f"Not enough parameters: expected {self._vars}, "
                             f"got {given_pars}")

        res_kwargs = {}
        res_vars = set()
        for var in self._vars:
            value = others[var.name]
            if isinstance(value, Function):
                res_kwargs[var.name] = value
            elif isinstance(value, Var):
                res_kwargs[var.name] = from_var_factory(value)
            elif isinstance(value, int) or isinstance(value, float):
                res_kwargs[var.name] = from_const_factory(value)
            else:
                raise TypeError(f"Element with unknown type for superposition: {value}")

            res_vars |= res_kwargs[var.name]._vars

        return Function(SUPERPOSITION, res_vars, self, name='', **res_kwargs)

    def __call__(self, **kwargs):
        if self._main_op == FROM_VAR:
            if self._sons[0] in kwargs:
                # noinspection PyTypeChecker
                return kwargs[self._sons[0]]
            if not kwargs:
                return self._sons[0]()

        elif self._main_op == FROM_CONST:
            return self._sons[0]

        elif self._main_op == FROM_FUNC:
            given_keys = set(kwargs.keys())
            if not kwargs:
                kwargs = {var.name: var() for var in self._vars}

            elif self._vars - given_keys != set():
                raise ValueError(f"Not enough parameters: expected {self._vars}, "
                                 f"got {given_keys}")

            res_kwargs = {el.name: kwargs[el.name] for el in self._vars}
            return self._sons[0](**res_kwargs)

        elif self._main_op == SUPERPOSITION:
            res_kwargs = {}
            for var, func in self._superpos_sons.items():
                if isinstance(var, Var):
                    res_kwargs[var.name] = func(**kwargs)
                elif isinstance(var, str):
                    res_kwargs[var] = func(**kwargs)
                else:
                    raise WTFError()
            return self._sons[0](**res_kwargs)

        elif self._main_op in BINARY_OPERATORS:
            return BINARY_DICT[self._main_op](self._sons[0](**kwargs),
                                              self._sons[1](**kwargs))

        else:
            raise WTFError()

    def _partial_complex_der(self, var: Var):
        assert var in self._vars
        assert self._main_op == SUPERPOSITION
        res = from_const_factory(0)
        F = self._sons[0]         # type: Function

        str_rep = ''
        # sum of ( dF / d _sub_var ) * (d _sub_var / d var)
        for _sub_var in F._vars:
            dFdf = F.partial_derivative(_sub_var)
            dFdf = dFdf.substitute(**self._superpos_sons)
            f = self._superpos_sons[_sub_var.name]  # type: Function
            dfdx = f.partial_derivative(var)
            add = dFdf * dfdx
            res += add
            str_rep += str(add)

        res.set_str_repr(str_rep)
        return res

    def _simple_derivative(self, var: Var):
        assert self._main_op == FROM_FUNC
        assert var in self._vars
        func = self._sons[0]    # type: callable
        var_name = var.name

        def res_func(**kwargs):
            with LOCK:
                kwargs[var_name] += self.delta_x
                f1 = func(**kwargs)
                kwargs[var_name] -= self.delta_x * 2
                f2 = func(**kwargs)
                kwargs[var_name] += self.delta_x
            return (f1 - f2) / (self.delta_x * 2)

        # TODO: add name
        return Function(FROM_FUNC, self._vars, res_func, check_signature=False,
                        str_repr=f'd/d{var} ({self})')

    def partial_derivative(self, var: Var):

        if var not in self._vars:
            return from_const_factory(0)

        if self._main_op == SUPERPOSITION:
            return self._partial_complex_der(var)

        elif self._main_op == FROM_FUNC:
            return self._simple_derivative(var)

        elif self._main_op == FROM_VAR:
            return from_const_factory(1)

        elif self._main_op == FROM_CONST:
            return from_const_factory(0)

        elif self._main_op in BINARY_OPERATORS:
            return self._binary_op_derivative(var)
        else:
            raise WTFError()

    def _binary_op_derivative(self, var: Var):
        assert var in self._vars
        u = self._sons[0]   # type: Function
        v = self._sons[1]   # type: Function
        u_der = u.partial_derivative(var)
        v_der = v.partial_derivative(var)

        res = SIMPLE_DERIVATIVES[self._main_op](u, v, u_der, v_der)  # type: Function
        return res

    def partial_derivative_n(self, *variables: Var):
        res = self
        for var in variables:
            res = res.partial_derivative(var)
        return res

    def get_vars(self):
        return self._vars

    def __pow__(self, power, modulo=None):
        if modulo:
            raise NotImplementedError()

        if isinstance(power, int):
            tmp_var = Var("___tmp_var___")
            tmp_func = ElementaryFunction('pow', tmp_var, n=power)
            return tmp_func.substitute(___tmp_var___=self)
        else:
            raise NotImplementedError()

    def __str__(self):
        if self._str_repr:
            return self._str_repr

        variables = ', '.join([str(var) for var in self._vars])
        if self._name:
            return f'{self._name}({variables})'
        else:
            if self._main_op == FROM_FUNC:
                func = self._sons[0]     # type: callable
                doc = f'{func.__name__}({variables})'
                return doc
            elif self._main_op in {FROM_VAR, FROM_CONST}:
                return str(self._sons[0])

            elif self._main_op == SUPERPOSITION:
                func = self._sons[0]  # type: Function
                res_doc = str(func)
                for el in func.get_vars():
                    res_doc = res_doc.replace(str(el), str(self._superpos_sons[el]))
                return res_doc

            elif self._main_op in BINARY_OPERATORS:
                f, g = self._sons[0], self._sons[1]   # type: Function
                if self._main_op == R_DIVISION:
                    return f'({g} {DIVISION} {f})'
                elif self._main_op == R_SUBTRACTION:
                    return f'({g} {SUBTRACTION} {f})'
                else:
                    return f'({f} {self._main_op} {g})'
            else:
                raise WTFError()

    def __repr__(self):
        return str(self)


class ElementaryFunction(Function):

    def __init__(self, func_name, variable: Var, n=2):
        if func_name not in ELEMENTARY_FUNCTIONS:
            raise NotImplementedError("Unknown function type")

        self._el_type = func_name
        self._n = n
        func, left, right = ELEMENTARY_FUNCTIONS[func_name]
        variable.left = max(variable.left, left)
        variable.right = min(variable.right, right)

        if self._el_type == 'pow':
            if n == 1:
                super(ElementaryFunction, self).__init__(FROM_VAR, {variable},
                                                         variable)
            elif n == 0:
                super(ElementaryFunction, self).__init__(FROM_CONST, set(), 0)
            else:
                func = get_power(n)
                super(ElementaryFunction, self).__init__(FROM_FUNC, {variable},
                                                         func, name=f'pow{n}',
                                                         check_signature=False)
        else:
            super(ElementaryFunction, self).__init__(FROM_FUNC, {variable}, func,
                                                     name=func_name, check_signature=False)

    def _simple_derivative(self, var: Var):
        assert self._main_op == FROM_FUNC
        assert var in self._vars

        if self._el_type == 'sin':
            return ElementaryFunction('cos', var)
        elif self._el_type == 'cos':
            return ElementaryFunction('sin', var) * (-1)
        elif self._el_type == 'sqrt':
            nom = from_const_factory(1)
            denom = ElementaryFunction('sqrt', var) * 2
            return nom / denom
        elif self._el_type == 'tan':
            nom = from_const_factory(1)
            denom = ElementaryFunction('cos', var) ** 2
            return nom / denom
        elif self._el_type == 'log':
            nom = from_const_factory(1)
            denom = from_var_factory(var)
            return nom / denom
        elif self._el_type == 'sinh':
            return ElementaryFunction('cosh', var)
        elif self._el_type == 'cosh':
            return ElementaryFunction('sinh', var)
        elif self._el_type == 'exp':
            return ElementaryFunction('exp', var)
        elif self._el_type == 'pow':
            return ElementaryFunction('pow', var, self._n-1) * self._n
        else:
            raise NotImplementedError()

    def __call__(self, **kwargs):
        if self._main_op == FROM_FUNC:
            var = next(iter(self._vars))
            value = kwargs.get(var, None)
            if value is None:
                value = var()
            return self._sons[0](value)
        else:
            return super(ElementaryFunction, self).__call__(**kwargs)


def from_func_factory(func: callable, variables: typ.Set[Var], name=''):
    return Function(FROM_FUNC,    variables, func, name=name)


def from_var_factory(var: Var):
    return Function(FROM_VAR, {var}, var)


def from_const_factory(const: typ.Union[int, float]):
    return Function(FROM_CONST, set(), const)


def _elementary_factory(name: str):

    def _res_factory(x: typ.Union[Function, int, float, Var]) -> Function:
        if isinstance(x, int) or isinstance(x, float):
            func, _, _ = ELEMENTARY_FUNCTIONS[name]
            return from_const_factory(func(x))

        elif isinstance(x, Var):
            return ElementaryFunction(name, x)

        elif isinstance(x, Function):
            tmp_var = Var('___tmp___')
            tmp_func = ElementaryFunction(name, tmp_var)
            return tmp_func.substitute(___tmp___=x)
        else:
            raise TypeError(f"Unknown variable type: {x}")

    return _res_factory


cos = _elementary_factory('cos')
sin = _elementary_factory('sin')
sqrt = _elementary_factory('sqrt')
tan = _elementary_factory('tan')
log = _elementary_factory('log')
sinh = _elementary_factory('sinh')
cosh = _elementary_factory('cosh')
exp = _elementary_factory('exp')
