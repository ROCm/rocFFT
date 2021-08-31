
from generator import *
from functools import reduce


def product(factors):
    return reduce(lambda x, y: x * y, factors)


def stockham_launch(kernel, **kwargs):
    """Launch helper.  Not used by rocFFT proper."""

    length, params = kernel.meta.length, kernel.meta.params

    # arguments
    scalar_type   = Variable('scalar_type', 'typename')
    embedded_type = Variable('EmbeddedType::NONE', 'EmbeddedType')
    callback_type = Variable('CallbackType::NONE', 'CallbackType')
    stride_type   = Variable('SB_UNIT', 'StrideBin')
    inout         = Variable('inout', 'scalar_type', array=True)
    twiddles      = Variable('twiddles', 'const scalar_type', array=True)
    stride_in     = Variable('stride_in', 'size_t')
    stride_out    = Variable('stride_out', 'size_t')
    nbatch        = Variable('nbatch', 'size_t')
    lds_padding   = Variable('lds_padding', 'unsigned int')
    kargs         = Variable('kargs', 'size_t*')

    # locals
    nblocks = Variable('nblocks', 'int')
    null    = Variable('nullptr', 'void*')

    # XXX precision
    lds_bytes = 16 * length * params.transforms_per_block
    if params.half_lds:
        lds_bytes //= 2

    body = StatementList()
    body += Declarations(nblocks)
    body += Assign(nblocks, B(nbatch + (params.transforms_per_block - 1)) / params.transforms_per_block)
    body += Call(f'forward_length{length}_SBRR',
                 arguments = ArgumentList(twiddles, 1, kargs, kargs + 1, nbatch, lds_padding, null, null, 0, null, null, inout),
                 templates = TemplateList(scalar_type, stride_type, embedded_type, callback_type),
                 launch_params = ArgumentList(nblocks, params.threads_per_block, lds_bytes))

    return Function(name = f'forward_length{length}_launch',
                    templates = TemplateList(scalar_type),
                    arguments = ArgumentList(inout, nbatch, lds_padding, twiddles, kargs, stride_in, stride_out),
                    body = body)



def stockham_launch2d(length, params, **kwargs):
    """Launch helper.  Not used by rocFFT proper."""

    # arguments
    scalar_type = Variable('scalar_type', 'typename')
    sb          = Variable('SB_NONUNIT', 'StrideBin')
    cbtype      = Variable('CallbackType::NONE', 'CallbackType')
    inout       = Variable('inout', 'scalar_type', array=True)
    twiddles    = Variable('twiddles', 'const scalar_type', array=True)
    stride_in   = Variable('stride_in', 'size_t')
    stride_out  = Variable('stride_out', 'size_t')
    nbatch      = Variable('nbatch', 'size_t')
    lds_padding = Variable('lds_padding', 'unsigned int')
    kargs       = Variable('kargs', 'size_t*')

    # locals
    nblocks = Variable('nblocks', 'int')
    null    = Variable('nullptr', 'void*')

    body = StatementList()
    body += Declarations(nblocks)
    body += Assign(nblocks, B(nbatch + (params.transforms_per_block - 1)) / params.transforms_per_block)
    length = 'x'.join(map(str, length))
    body += Call(f'forward_length{length}',
                 arguments = ArgumentList(twiddles, 2, kargs, kargs+2, nbatch, null, null, 0, null, null, inout),
                 templates = TemplateList(scalar_type, sb, cbtype),
                 launch_params = ArgumentList(nblocks, params.threads_per_block))

    return Function(name = f'forward_length{length}_launch',
                    templates = TemplateList(scalar_type),
                    arguments = ArgumentList(inout, nbatch, lds_padding, twiddles, kargs, stride_in, stride_out),
                    body = body)
