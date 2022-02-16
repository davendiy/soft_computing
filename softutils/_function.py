#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 2021-11-29
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from __future__ import annotations
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
    'tanh': (np.tanh, -inf, inf),
    'sigmoid': (lambda x: 1 / (1 + np.exp(x)), -inf, inf),
    'relu': (lambda x: np.max(x, 0), -inf, inf),
    'relu_der': (lambda x: x > 0, -inf, inf),
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


class NamewiseSingleton(type):

    _lock = RLock()

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls.__instances = {}

    def __call__(cls, name, *args, **kwargs):
        if name not in cls.__instances:
            with cls._lock:
                cls.__instances[name] = super().__call__(name, *args, **kwargs)
        return cls.__instances[name]


class _Var(metaclass=NamewiseSingleton):

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
    def left(self, value: typ.Union[float, int]):
        with LOCK:
            self._left = value

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, right: typ.Union[float, int]):
        with LOCK:
            self._right = right

    def random_value(self, size=1):
        if self._left > -inf and self._right < inf:
            if size > 1:
                return np.random.uniform(self._left, self._right, size=size)
            else:
                return np.random.uniform(self._left, self._right, size=size)[0]
        else:
            if size > 1:
                return np.random.normal(0, 1, size=size)
            else:
                return np.random.normal(0, 1, size=size)[0]

    def __hash__(self):
        return hash(self.name)

    def __call__(self):
        if self._value is None:
            raise NotInitialisedVarError(self.name)
        return self._value

    def __eq__(self, other):
        if isinstance(other, _Var):
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

class _Function:

    delta_x = 10e-5

    def __init__(self, main_op: str, variables: typ.Set[_Var],
                 *sons: typ.Union[typ.Callable, float, int, _Function],
                 name='', str_repr='', check_signature=True, **superpos_sons):
        assert main_op in OPERATORS, 'bad operation'
        assert all(isinstance(var, _Var) for var in variables), 'bad variable'
        assert not (len(sons) > 1 and main_op in UNARY_OPERATORS), 'bad amount of sons'
        assert not (len(sons) != 2 and main_op in BINARY_OPERATORS), 'bad amount of sons'
        assert not (main_op in BINARY_OPERATORS and
                    any(not isinstance(son, _Function) for son in sons)), 'bad type of sons'

        assert not (len(sons) != 1 and main_op == SUPERPOSITION
                    and len(superpos_sons) == 0), 'bad superposition format'

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
        func = self._sons[0]    # type: _Function
        if given_vars != func._vars:
            raise ValueError(f"Bad set of variables are given: "
                             f"expected {func._vars}, got {self._superpos_sons}")

        sons_vars = set()
        for el in self._superpos_sons.values():
            if not isinstance(el, _Function):
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
        if not any(isinstance(other, _type)
                   for _type in [_Function, _Var, int, float]):
            other = float(other)

        assert operation_type in BINARY_OPERATORS, 'bad operation'

        if isinstance(other, _Var):
            res_other = _Function(FROM_VAR, {other}, other)
        elif isinstance(other, float) or isinstance(other, int):
            res_other = _Function(FROM_CONST, set(), other)
        else:
            res_other = other
        return _Function(operation_type, self._vars | res_other._vars,
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
        # noinspection PyTypeChecker
        if self._vars - given_pars:
            raise ValueError(f"Not enough parameters: expected {self._vars}, "
                             f"got {given_pars}")

        res_kwargs = {}
        res_vars = set()
        for var in self._vars:
            value = others[var.name]
            if isinstance(value, _Function):
                res_kwargs[var.name] = value
            elif isinstance(value, _Var):
                res_kwargs[var.name] = _from_var_factory(value)
            elif isinstance(value, int) or isinstance(value, float):
                res_kwargs[var.name] = _from_const_factory(value)
            else:
                raise TypeError(f"Element with unknown type for superposition: {value}")

            res_vars |= res_kwargs[var.name]._vars

        return _Function(SUPERPOSITION, res_vars, self, name='', **res_kwargs)

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
            # noinspection PyTypeChecker
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
                if isinstance(var, _Var):
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

    def _partial_complex_der(self, var: _Var):
        assert var in self._vars
        assert self._main_op == SUPERPOSITION
        res = _from_const_factory(0)
        F = self._sons[0]         # type: _Function

        str_rep = ''
        # sum of ( dF / d _sub_var ) * (d _sub_var / d var)
        for _sub_var in F._vars:
            dFdf = F.partial_derivative(_sub_var)
            dFdf = dFdf.substitute(**self._superpos_sons)
            f = self._superpos_sons[_sub_var.name]  # type: _Function
            dfdx = f.partial_derivative(var)
            add = dFdf * dfdx
            res += add
            str_rep += str(add)

        res.set_str_repr(str_rep)
        return res

    def _simple_derivative(self, var: _Var):
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
        return _Function(FROM_FUNC, self._vars, res_func, check_signature=False,
                         str_repr=f'd/d{var} ({self})')

    def partial_derivative(self, var: typ.Union[_Var, str]):

        if var not in self._vars:
            return _from_const_factory(0)

        if self._main_op == SUPERPOSITION:
            return self._partial_complex_der(var)

        elif self._main_op == FROM_FUNC:
            return self._simple_derivative(var)

        elif self._main_op == FROM_VAR:
            return _from_const_factory(1)

        elif self._main_op == FROM_CONST:
            return _from_const_factory(0)

        elif self._main_op in BINARY_OPERATORS:
            return self._binary_op_derivative(var)
        else:
            raise WTFError()

    def _binary_op_derivative(self, var: _Var):
        assert var in self._vars
        u = self._sons[0]   # type: _Function
        v = self._sons[1]   # type: _Function
        u_der = u.partial_derivative(var)
        v_der = v.partial_derivative(var)

        res = SIMPLE_DERIVATIVES[self._main_op](u, v, u_der, v_der)  # type: _Function
        return res

    def partial_derivative_n(self, *variables: _Var):
        res = self
        for var in variables:
            res = res.partial_derivative(var)
        return res

    def optimize(self, *variables: str, algo='gradient_descent',
                 max_iterations=100, eps=10e-5, eta=0.01, dest='max', **fixed_vars):
        params = {var: _Var(var).random_value() for var in variables}
        params.update(fixed_vars)
        if algo != 'gradient_descent':
            raise NotImplementedError()

        multiplier = 1 if dest == 'max' else -1
        for var in params:
            if not isinstance(var, str): print(var, type(var))
        assert all(isinstance(var, str) for var in params)
        cur_value = self(**params)
        history = [cur_value]
        for _ in range(max_iterations):
            for var in variables:
                der = self.partial_derivative(var)
                der = der(**params)
                params[var] += multiplier * eta * der
            next_value = self(**params)
            history.append(next_value)
            # if abs(next_value - cur_value) < eps:
            #     break
            cur_value = next_value

        return history, [params[var] for var in variables]

    def minimize(self, *variables: _Var, algo='gradient_descent', max_iterations=1000,
                 eps=10e-5, eta=0.01, **fixed_vars):
        return self.optimize(*variables, algo=algo, max_iterations=max_iterations, eps=eps,
                             dest='min', eta=eta, **fixed_vars)

    def maximize(self, *variables: _Var, algo='gradient_descent', max_iterations=1000,
                 eps=10e-5, eta=0.01, **fixed_vars):
        return self.optimize(*variables, algo=algo, max_iterations=max_iterations, eps=eps,
                             dest='max', eta=eta, **fixed_vars)

    def gradient(self):
        return [self.partial_derivative(var) for var in self._vars]

    def get_vars(self):
        return self._vars

    def __pow__(self, power, modulo=None):
        if modulo:
            raise NotImplementedError()

        if isinstance(power, int):
            tmp_var = _Var("___tmp_var___")
            tmp_func = _ElementaryFunction('pow', tmp_var, n=power)
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
                func = self._sons[0]  # type: _Function
                res_doc = str(func)
                for el in func.get_vars():
                    # noinspection PyTypeChecker
                    res_doc = res_doc.replace(str(el), str(self._superpos_sons[el]))
                return res_doc

            elif self._main_op in BINARY_OPERATORS:
                f, g = self._sons[0], self._sons[1]   # type: _Function
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


# TODO: replace with _from_func_factory for every elementary function
class _ElementaryFunction(_Function):

    def __init__(self, func_name, variable: _Var, n=2):
        if func_name not in ELEMENTARY_FUNCTIONS:
            raise NotImplementedError("Unknown function type")

        self._el_type = func_name
        self._n = n
        func, left, right = ELEMENTARY_FUNCTIONS[func_name]
        variable.left = max(variable.left, left)
        variable.right = min(variable.right, right)

        if self._el_type == 'pow':
            if n == 1:
                super(_ElementaryFunction, self).__init__(FROM_VAR, {variable},
                                                          variable)
            elif n == 0:
                super(_ElementaryFunction, self).__init__(FROM_CONST, set(), 0)
            else:
                func = get_power(n)
                super(_ElementaryFunction, self).__init__(FROM_FUNC, {variable},
                                                          func, name=f'pow{n}',
                                                          check_signature=False)
        else:
            super(_ElementaryFunction, self).__init__(FROM_FUNC, {variable}, func,
                                                      name=func_name, check_signature=False)

    def _simple_derivative(self, var: _Var):
        assert self._main_op == FROM_FUNC
        assert var in self._vars

        if self._el_type == 'sin':
            return _ElementaryFunction('cos', var)
        elif self._el_type == 'cos':
            return _ElementaryFunction('sin', var) * (-1)
        elif self._el_type == 'sqrt':
            nom = _from_const_factory(1)
            denom = _ElementaryFunction('sqrt', var) * 2
            return nom / denom
        elif self._el_type == 'tan':
            nom = _from_const_factory(1)
            denom = _ElementaryFunction('cos', var) ** 2
            return nom / denom
        elif self._el_type == 'log':
            nom = _from_const_factory(1)
            denom = _from_var_factory(var)
            return nom / denom
        elif self._el_type == 'sinh':
            return _ElementaryFunction('cosh', var)
        elif self._el_type == 'cosh':
            return _ElementaryFunction('sinh', var)
        elif self._el_type == 'tanh':
            return _from_const_factory(1) - _ElementaryFunction('tanh', var) ** 2
        elif self._el_type == 'sigmoid':
            return _from_var_factory(var) * (1 - _from_var_factory(var))
        elif self._el_type == 'relu':
            return _ElementaryFunction('relu_der', var)
        elif self._el_type == 'relu_der':
            return _from_const_factory(0)
        elif self._el_type == 'exp':
            return _ElementaryFunction('exp', var)
        elif self._el_type == 'pow':
            return _ElementaryFunction('pow', var, self._n - 1) * self._n
        else:
            raise NotImplementedError()

    def __call__(self, **kwargs):
        if self._main_op == FROM_FUNC:
            var = next(iter(self._vars))
            # noinspection PyTypeChecker
            value = kwargs.get(var, None)
            if value is None:
                value = var()
            return self._sons[0](value)
        else:
            return super(_ElementaryFunction, self).__call__(**kwargs)


def _from_func_factory(func: callable, variables: typ.Set[_Var], name=''):
    return _Function(FROM_FUNC, variables, func, name=name)


def _from_var_factory(var: _Var):
    return _Function(FROM_VAR, {var}, var)


def _from_const_factory(const: typ.Union[int, float]):
    if not isinstance(const, int):
        const = float(const)
    return _Function(FROM_CONST, set(), const)


def _elementary_factory(name: str):

    def _res_factory(x: typ.Union[_Function, int, float, _Var]) -> _Function:
        if isinstance(x, int) or isinstance(x, float):
            func, _, _ = ELEMENTARY_FUNCTIONS[name]
            return _from_const_factory(func(x))

        elif isinstance(x, _Var):
            return _ElementaryFunction(name, x)

        elif isinstance(x, _Function):
            tmp_var = _Var('___tmp___')
            tmp_func = _ElementaryFunction(name, tmp_var)
            return tmp_func.substitute(___tmp___=x)
        else:
            raise TypeError(f"Unknown variable type: {x}")

    return _res_factory


# noinspection PyPep8Naming
def Var(name, left=float('-inf'), right=float('inf')):
    return _from_var_factory(_Var(name, left, right))


# noinspection PyPep8Naming
def Const(const):
    return _from_const_factory(const)


# noinspection PyPep8Naming
def Function(**kwargs):
    if 'func' in kwargs:
        return _from_func_factory(**kwargs)
    elif 'var' in kwargs:
        return _from_var_factory(**kwargs)
    elif 'const' in kwargs:
        return _from_const_factory(**kwargs)
    else:
        raise NotImplementedError()


cos = _elementary_factory('cos')
sin = _elementary_factory('sin')
sqrt = _elementary_factory('sqrt')
tan = _elementary_factory('tan')
log = _elementary_factory('log')
sinh = _elementary_factory('sinh')
cosh = _elementary_factory('cosh')
exp = _elementary_factory('exp')
tanh = _elementary_factory('tanh')
sigmoid = _elementary_factory('sigmoid')
relu = _elementary_factory('relu')
