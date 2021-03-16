"""HIP code generator."""

import inspect
import os
import subprocess

from pathlib import Path as path
from typing import List, Any


#
# Helpers
#

def join(sep, n):
    """Coerce elements in `n` to strings and join them seperator `sep`."""
    if isinstance(n, BaseNode):
        return sep.join(str(x) for x in n.args)
    return sep.join(str(x) for x in n)


def sjoin(n):
    """Join with spaces."""
    return join(' ', n)


def njoin(n):
    """Join with newlines."""
    return join(os.linesep, n)


def cjoin(n):
    """Join with commas."""
    return join(', ', n)


# XXX get rid of this
def declarations(xs):
    return [x.declaration() for x in xs]


def format(code):
    """Format code using clang-format."""
    try:
        p = subprocess.run(['/opt/rocm/llvm/bin/clang-format', '-style=file'],
                           stdout=subprocess.PIPE,
                           input=str(code),
                           encoding='ascii',
                           check=True)
        return p.stdout
    except FileNotFoundError:
        pass
    return str(code)


def format_and_write(fname, code):
    """Format code and write to `fname`.

    If `fname` already exists, only write if the contents changed.
    """
    code = format(code)
    f = path(fname)
    if f.exists():
        existing = f.read_text()
        if existing == code:
            return
    f.write_text(code)


def walk(x):
    """Tree traversal."""
    if isinstance(x, BaseNode):
        yield x
        for a in x.args:
            if isinstance(a, BaseNode):
                yield from a
            else:
                yield a


def depth_first(x, f):
    """Apply `f` to each node of the AST `x`.

    Nodes are traversed in depth-first order.
    """
    if isinstance(x, BaseNode):
        y = type(x)()
        y.args = [depth_first(a, f) for a in x.args]
        return f(y)
    return f(x)


#
# Code generator base classes
#

def make_raw(s):
    """Make a simple AST node whose string representation is `s`."""
    def decorator(target):
        target.__str__ = lambda self: s
        return target
    return decorator

# XXX there is some redundancy between name_args and the constructor
#     for BaseNode
# XXX tighten up provenance


def name_args(names):
    """Make an AST node with named arguments."""

    def name_args_decorator(target):

        # attach getters for each name
        for i, name in enumerate(names):
            setattr(target, name, property(lambda self, idx=i: self.args[idx]))

        # define a new init that takes args and kwargs using names
        def new_init(self, *args, **kwargs):
            self.args = [None for x in names]
            for i, arg in enumerate(args):
                self.args[i] = arg
            for i, name in enumerate(names):
                if name in kwargs:
                    self.args[i] = kwargs[name]

            # provenance
            self.file_name, self.line_number, *_ = inspect.getframeinfo(
                inspect.currentframe().f_back)

        target.__init__ = new_init
        return target

    return name_args_decorator


class BaseNode:
    """Base AST Node."""

    args: List[Any]
    sep: str = None

    def __init__(self, *args, **kwargs):
        self.file_name, self.line_number, *_ = inspect.getframeinfo(
            inspect.currentframe().f_back)
        self.args = list(args)
        if hasattr(self, '__post_init__'):
            getattr(self, '__post_init__')(self)

    def __str__(self):
        if self.sep is not None:
            return join(self.sep, self.args)
        return str(self.args[0])

    def __iter__(self):
        return walk(self)

    def provenance(self):
        return '/* ' + self.file_name + ':' + str(self.line_number) + ' */'


class BaseNodeOps(BaseNode):
    """BaseNode with basic math operations added."""

    def __add__(self, a):
        return Add(self, a)

    def __radd__(self, a):
        return Add(a, self)

    def __sub__(self, a):
        return Sub(self, a)

    def __rsub__(self, a):
        return Sub(a, self)

    def __mul__(self, a):
        return Multiply(self, a)

    def __rmul__(self, a):
        return Multiply(a, self)

    def __mod__(self, a):
        return Mod(self, a)

    def __rmod__(self, a):
        return Mod(a, self)

    def __truediv__(self, a):
        return Divide(self, a)

    def __rtruediv__(self, a):
        return Divide(a, self)

    def __ge__(self, a):
        return GreaterEqual(self, a)

    def __gt__(self, a):
        return Greater(self, a)

    def __le__(self, a):
        return LessEqual(self, a)

    def __lt__(self, a):
        return Less(self, a)


class ArgumentList(BaseNode):

    def __str__(self):
        args = []
        for x in self.args:
            if isinstance(x, Variable):
                args.append(x.argument())
            else:
                args.append(str(x))
        return cjoin(args)


class StatementList(BaseNode):

    def __add__(self, lst):
        if isinstance(lst, list):
            self.args.extend(lst)
        elif isinstance(lst, StatementList):
            self.args.extend(lst.args)
        else:
            self.args.append(lst)
        return self

    def __str__(self):
        return njoin(self.args)


class Declaration(BaseNode):
    pass


class TemplateList(ArgumentList):
    pass


class CommentBlock(BaseNode):
    def __str__(self):
        return njoin(['/*'] + [' * ' + str(a) for a in self.args] + [' */'])


class Comments(BaseNode):
    def __str__(self):
        return njoin(' // ' + str(a) for a in self.args)


class Pragma(BaseNode):
    def __str__(self):
        return '#pragma ' + sjoin(self.args)


class Include(BaseNode):
    def __str__(self):
        return '#include ' + sjoin(self.args)


class ExternC(BaseNode):
    def __str__(self):
        return 'extern "C" { ' + njoin(self.args) + ' }'


@make_raw(os.linesep + os.linesep)
class LineBreak(BaseNode):
    pass


@make_raw('return;')
class ReturnStatement(BaseNode):
    pass


@make_raw('__syncthreads();')
class SyncThreads(BaseNode):
    pass


#
# Operators
#

def make_unary(prefix):
    def decorator(target):
        target.__str__ = lambda self: prefix + self.args[0]
        return target
    return decorator


def make_binary(separator):
    def decorator(target):
        target.sep = separator
        return target
    return decorator


@make_unary('&')
class Address(BaseNode):
    pass


@name_args(['lhs', 'rhs'])
class Assign(BaseNode):
    def __str__(self):
        return str(self.args[0]) + ' = ' + str(self.args[1]) + ';' \
            + self.provenance()


@make_binary('.')
class Component(BaseNodeOps):
    pass


@make_binary(' + ')
class Add(BaseNodeOps):
    pass


@make_binary(' - ')
class Sub(BaseNodeOps):
    pass


@make_binary(' / ')
class Divide(BaseNodeOps):
    pass


@make_binary(' * ')
class Multiply(BaseNodeOps):
    pass


@make_binary(' % ')
class Mod(BaseNodeOps):
    pass


@make_binary(' > ')
class Greater(BaseNodeOps):
    pass


@make_binary(' >= ')
class GreaterEqual(BaseNodeOps):
    pass


@make_binary(' < ')
class Less(BaseNodeOps):
    pass


@make_binary(' <= ')
class LessEqual(BaseNodeOps):
    pass


#
# Variables
#

@name_args(['variable', 'index'])
class ArrayElement(BaseNodeOps):

    @property
    def x(self):
        return Component(str(self), 'x')

    @property
    def y(self):
        return Component(str(self), 'y')

    def address(self):
        return Address(str(self))

    def __str__(self) -> str:
        return str(self.variable) + '[' + str(self.index) + ']'


@name_args(['name', 'type', 'size', 'array'])
class Variable(BaseNodeOps):

    @property
    def x(self):
        return Component(self.name, 'x')

    @property
    def y(self):
        return Component(self.name, 'y')

    def address(self):
        return Address(self.name)

    def declaration(self):
        if self.size is not None:
            return Declaration(f'{self.type} {self.name}[{self.size}];')
        return Declaration(f'{self.type} {self.name};')

    def argument(self):
        if self.array:
            return f'{self.type} *{self.name}'
        return f'{self.type} {self.name}'

    def __str__(self):
        return str(self.name)

    def __getitem__(self, idx):
        return ArrayElement(self.name, idx)


class ComplexLiteral(BaseNodeOps):
    def __str__(self):
        return '{' + str(self.args[0]) + ', ' + str(self.args[1]) + '}'


class Group(BaseNodeOps):
    def __str__(self):
        return '(' + str(self.args[0]) + ')'


B = Group

#
# Control flow
#


class Block(BaseNode):
    def __str__(self):
        return '{' + njoin(self.args) + '}'


@name_args(['condition', 'body'])
class If(BaseNode):
    def __str__(self):
        return 'if(' + str(self.condition) + ') {' + njoin(self.body) + '}'


@name_args(['condition', 'body'])
class While(BaseNode):
    def __str__(self):
        return 'while(' + str(self.condition) + ') {' + njoin(self.body) + '}'


#
# Functions
#


@name_args(['name', 'arguments', 'templates', 'qualifier'])
class Prototype(BaseNode):
    def __str__(self) -> str:
        f = ''
        if self.templates:
            f += 'template<' + str(self.templates) + '>'
        if self.qualifier is not None:
            f += self.qualifier + ' '
        f += ' void ' + self.name
        f += '(' + str(self.arguments) + ');'
        return f


@name_args(['name', 'value', 'arguments', 'templates', 'qualifier',
            'launch_bounds', 'body', 'meta'])
class Function(BaseNode):

    def __str__(self) -> str:
        f = self.provenance() + os.linesep
        if self.templates:
            f += 'template<' + str(self.templates) + '>'
        if self.qualifier is not None:
            f += self.qualifier + ' '
        if self.launch_bounds is not None:
            f += '__launch_bounds__(' + str(self.launch_bounds) + ') '
        if self.value is None:
            f += ' void '
        elif self.value:
            f += ' ' + str(self.value) + ' '
        f += self.name
        f += '(' + str(self.arguments) + ')'
        f += '{' + njoin(self.body) + '}'
        return f

    def prototype(self):
        return Prototype(self.name, self.arguments, self.templates,
                         self.qualifier)

    def address(self):
        return Address(self.name)


@name_args(['name', 'arguments', 'templates', 'launch_params'])
class Call(BaseNode):

    def __str__(self) -> str:
        f = self.name
        if self.templates:
            f += '<' + cjoin(self.templates) + '>'
        if self.launch_params:
            f += '<<<' + cjoin(self.launch_params) + '>>>'
        f += '(' + cjoin(self.arguments) + ');'
        f += self.provenance() + os.linesep
        return f

    def inline(self):
        return InlineCall(*self.args)


class InlineCall(Call):
    def __str__(self) -> str:
        f = self.name
        if self.templates:
            f += '<' + cjoin(self.templates) + '>'
        f += '(' + cjoin(self.arguments) + ')'
        return f
