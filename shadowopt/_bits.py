"""Low-level helpers shared by every kernel (S*, bitmaps, shared plans) so that all paths use the same primitives."""
import numpy as np
from numba import njit, types
from numba.extending import intrinsic
from llvmlite import ir


@intrinsic
def popcnt64(typingctx, x):
    """Hardware popcount of a uint64 (LLVM ctpop; compiled for the host CPU)."""
    sig = types.uint64(types.uint64)

    def codegen(context, builder, signature, args):
        fn = builder.module.declare_intrinsic("llvm.ctpop", [ir.IntType(64)])
        return builder.call(fn, args)

    return sig, codegen


@njit(cache=True)
def popcount_array(a):
    s = 0
    for i in range(a.shape[0]):
        s += popcnt64(a[i])
    return s


def n_words(nbits):
    return (nbits + 63) // 64
