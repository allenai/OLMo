import argparse
import math
import subprocess
import time

import intrin
import numpy as np
import pandas as pd
import template
from gekko import GEKKO


def mem_model(N, M, T, mu, tu, bits, l1, p, gs, verbose=False):
    m = GEKKO()  # create GEKKO model
    # cinfergen if bits==3:
    # tu = tu*3
    B = m.Const(value=bits)
    TP = m.Const(value=T // p)
    k = m.Var(1, integer=True, lb=1)
    z = m.Var(1, integer=True, lb=1)
    w = m.Var(1, integer=True, lb=1)
    y = m.Var(1, integer=True, lb=1)
    v = m.Var(1, integer=True, lb=1)
    mb = m.Var(mu, integer=True, lb=1)
    if gs != -1:
        gg = m.Var(1, integer=True, lb=1)
    tb = m.Var(tu, integer=True, lb=1, ub=int(T / p))
    L = m.Var(integer=True, lb=0, ub=l1)
    m.Equation(L == 32 * mb * N + B * mb * tb + 32 * tb * N)
    m.Equation(mb * k == M)
    if gs != -1:
        m.Equation(gs * gg == mb)
    # m.Equation(tb * z == T)
    m.Equation(tb * z == TP)
    m.Equation(mu * w == mb)
    m.Equation(tu * y == tb)
    # m.Equation(tb * v == tt)
    m.Maximize(L)
    m.options.SOLVER = 1
    m.solver_options = [
        "minlp_maximum_iterations 1000",  # minlp iterations with integer solution
        "minlp_max_iter_with_int_sol 10",  # treat minlp as nlp
        "minlp_as_nlp 0",  # nlp sub-problem max iterations
        "nlp_maximum_iterations 100",  # 1 = depth first, 2 = breadth first
        "minlp_branch_method 2",  # maximum deviation from whole number
        "minlp_integer_tol 0.00",  # covergence tolerance
        "minlp_gap_tol 0.01",
    ]
    try:
        m.solve(disp=False)
    except:
        try:
            m.solver_options = [
                "minlp_maximum_iterations 1000",  # minlp iterations with integer solution
                "minlp_max_iter_with_int_sol 10",  # treat minlp as nlp
                "minlp_as_nlp 0",  # nlp sub-problem max iterations
                "nlp_maximum_iterations 100",  # 1 = depth first, 2 = breadth first
                "minlp_branch_method 1",  # maximum deviation from whole number
                "minlp_integer_tol 0.00",  # covergence tolerance
                "minlp_gap_tol 0.01",
            ]
            m.solve(disp=False)
        except:
            # mytb = T//p
            mytb = tu
            if gs != -1:
                mymb = gs
                while 32 * (mymb + gs) * N + bits * (mymb + gs) * mytb + 32 * mytb * N < l1:
                    mymb += gs
                while M % mymb != 0:
                    mymb -= gs
                if verbose:
                    print("Failed to solve, using heuristic. mb = ", mymb, "tb = ", mytb)
                return (int(mymb), int(mytb))
            else:
                mymb = mu
                while 32 * (mymb + mu) * N + bits * (mymb + mu) * mytb + 32 * mytb * N < l1:
                    mymb += mu
                while M % mymb != 0:
                    mymb -= mu
                if verbose:
                    print("Failed to solve, using heuristic. mb = ", mymb, "tb = ", mytb)
                return (int(mymb), int(mytb))

    if verbose:
        print("mb = ", int(mb.value[0]), "tb = ", int(tb.value[0]))
    return (int(mb.value[0]), int(tb.value[0]))


def macros():
    return "#include<omp.h>\n#include<immintrin.h>\n#include<fstream>\n\n#define mymin(a,b) ((a)<(b)?(a):(b))\n#define mymax(a,b) ((a)>(b)?(a):(b))\n"


def print_parameters(bits, n, m, t, nb, mb, tb, mu, nu, tu, unroll, p, gs=-1):
    res = ""
    res += "void print_parameters(){\n"
    res += f'  std::cout << {bits} << "bits," << {n} << "," << {m} << "," << {t} << "," << {nb} << "," << {mb} << "," << {tb} << "," << {nu} << "," << {mu} << "," << {tu} << "," << {unroll} << "," << {p}  << "," << {gs} << ",";\n'
    res += "}\n"
    return res


def print_parameters_module(bits, mu, nu, tu, unroll, p, gs=-1):
    res = ""
    res += "void print_parameters(){\n"
    res += "std::ofstream outfile;\n"
    res += 'outfile.open("./autogptq_extension/qigen/tmp.csv", std::ios_base::app);\n'
    res += f'outfile << {bits} << "," << {nu} << "," << {mu} << "," << {tu} << "," << {unroll} << "," << {p}  << "," << {gs} << ",";\n'
    res += "}\n"
    return res


def pack_in(n, m, nb, mb):
    res = ""
    res += "inline void pack_input(float* A, float* B){\n"
    res += "  // copy the full matrix A in blocked format into B\n"
    res += "  uint64_t idx = 0;\n"
    res += f"  const int N = {n};\n"
    res += f"  const int M = {m};\n"
    res += f"  const int nb = {nb};\n"
    res += f"  const int mb = {mb};\n"
    res += "  for(int i = 0; i < N; i+=nb){ \n \
            for(int j = 0; j < M; j+=mb){\n \
                for(int jj = j; jj < mymin(j+mb, M); jj++){\n \
                    for(int ii = i; ii < mymin(i+nb, N); ii++){\n \
                        B[idx] = A[ii*M+jj];\n \
                        idx++;\n \
                    }\n \
                }\n \
            }\n \
        }\n \
    }\n"
    return res


def pack_out(n, t, nb, tb):
    res = ""
    res += "inline void pack_output(float* A, float* B){\n"
    res += "  // copy the full matrix A in blocked format into B\n"
    res += "  uint64_t idx = 0;\n"
    res += f"  const int N = {n};\n"
    res += f"  const int M = {t};\n"
    res += f"  const int nb = {nb};\n"
    res += f"  const int mb = {tb};\n"
    res += "  for(int i = 0; i < N; i+=nb){ \n \
            for(int j = 0; j < M; j+=mb){\n \
                for(int ii = i; ii < mymin(i+nb, N); ii++){\n \
                    for(int jj = j; jj < mymin(j+mb, M); jj++){\n \
                        B[idx] = A[ii*M+jj];\n \
                        idx++;\n \
                    }\n \
                }\n \
            }\n \
        }\n \
    }\n"
    return res


def pack_qw(m, t, mb, tb, tb1, bits=4, cutoff=-1):
    packed = 32 // bits
    res = ""
    if cutoff == -1:
        cutoff = 65
    if bits == 3:
        res += "inline void pack_qw_inner(int* A, int* B, int cutoff){\n"
        res += "  // copy the full matrix A in blocked format into B\n"
        res += "  uint64_t idx = 0;\n"
        res += f"  const int N = {m // 32 * 3};\n"
        res += f"  const int M = {t};\n"
        res += f"  const int nb = {mb // 32 * 3};\n"
        res += f"int mb = {int(tb)};\n"
        res += "    for(int j = 0, tid = 0; j < M; j+=mb, tid++){\n"
        # res += "if(tid==cutoff){\n "
        # res += f"  mb = {tb1};\n"
        # res += "}\n"
        res += "        for(int i = 0; i < N; i+=nb){\n \
                    for(int ii = i; ii < mymin(i+nb, N); ii+=3){\n \
                        for(int jj = j; jj < mymin(j+mb, M); jj+=8){\n \
                            for(int iii = ii; iii < ii + 3; iii++){\n \
                                for(int jjj = jj; jjj < jj + 8; jjj++){\n \
                                    B[idx] = A[iii*M+jjj];\n \
                                    idx++;\n \
                                }\n \
                            }\n \
                        }\n \
                    }\n \
                }\n \
            }\n \
        }\n"
        res += "inline void pack_qw(int* A, int* B){\n"
        res += f"  pack_qw_inner(A, B, {cutoff});\n"
        res += "}\n"
        return res
    else:
        # in case i do this for python i can just add the n,m,nb,mb as function parameters
        res += "inline void pack_qw_inner(int* A, int* B, int cutoff){\n"
        res += "  // copy the full matrix A in blocked format into B\n"
        res += "  uint64_t idx = 0;\n"
        res += f"  const int N = {m // packed};\n"
        res += f"  const int M = {t};\n"
        res += f"  const int nb = {mb // packed};\n"
        res += f"int mb = {int(tb)};\n"
        res += "    for(int j = 0, tid = 0; j < M; j+=mb, tid++){\n"
        # res += "if(tid==cutoff){\n "
        # res += f"  mb = {tb1};\n"
        # res += "}\n"
        res += " for(int i = 0; i < N; i+=nb){\n \
                    for(int ii = i; ii < mymin(i+nb, N); ii++){\n \
                        for(int jj = j; jj < mymin(j+mb, M); jj++){\n \
                            B[idx] = A[ii*M+jj];\n \
                            idx++;\n \
                        }\n \
                    }\n \
                }\n"
        res += "}\n"
        res += "}\n"
        res += "inline void pack_qw(int* A, int* B){\n"
        res += f"  pack_qw_inner(A, B, {cutoff});\n"
        res += "}\n"
        return res


def block_gs(nu_iter, mu, tu, rho, packed, unroll, bits):
    res = ""
    i = 0
    # unroll = 4 # number of bcasts and unpacks
    if bits == 3:
        for j in range(0, tu, 8):
            res += f"__m256i w0_{j} = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/{packed}*3 + k*mb*tb/{packed}*3 + k3*tb/{packed}*3 + jw+{j*3}]);\n"
            res += f"__m256i w1_{j} = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/{packed}*3 + k*mb*tb/{packed}*3 + k3*tb/{packed}*3 + jw+{j*3}+8]);\n"
            res += f"__m256i w2_{j} = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/{packed}*3 + k*mb*tb/{packed}*3 + k3*tb/{packed}*3 + jw+{j*3}+16]);\n"

        u = 0
        first_off = 3
        second_off = 2
        wid = 0
        shift = 0
        while u < 32:
            if u == 10:
                res += f"__m256 v{i}_{u} = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k3+{u})*nb + i1+{i}]);\n"

                for j in range(0, tu, 8):
                    res += f"__m256i ws{j}_10 = _mm256_srli_epi32(w0_{j}, {bits*10});\n"
                    res += f"__m256i temp0_{j} = _mm256_slli_epi32(w1_{j}, 2);\n"
                    res += f"temp0_{j} = _mm256_and_si256(temp0_{j}, mask);\n"
                    res += f"ws{j}_10 = _mm256_or_si256(ws{j}_10, temp0_{j});\n"

                    res += f"__m256i wsa{j}_{u} = _mm256_and_si256(ws{j}_{u}, mask);\n"

                    res += f"__m256 l{j}_{u} = _mm256_cvtepi32_ps(wsa{j}_{u});\n"

                    res += f"acc{i}_{j} = _mm256_fmadd_ps(v{i}_{u}, l{j}_{u}, acc{i}_{j});\n"

                wid = wid + 1
                u = u + 1

            elif u == 21:
                res += f"__m256 v{i}_{u} = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k3+{u})*nb + i1+{i}]);\n"

                for j in range(0, tu, 8):
                    res += f"__m256i ws{j}_{u} = _mm256_srli_epi32(w1_{j}, 31);\n"
                    res += f"__m256i temp1_{j} = _mm256_slli_epi32(w2_{j}, 1);\n"
                    res += f"temp1_{j} = _mm256_and_si256(temp1_{j}, mask);\n"
                    res += f"ws{j}_{u} = _mm256_or_si256(ws{j}_{u}, temp1_{j});\n"

                    res += f"__m256i wsa{j}_{u} = _mm256_and_si256(ws{j}_{u}, mask);\n"

                    res += f"__m256 l{j}_{u} = _mm256_cvtepi32_ps(wsa{j}_{u});\n"

                    res += f"acc{i}_{j} = _mm256_fmadd_ps(v{i}_{u}, l{j}_{u}, acc{i}_{j});\n"

                wid = wid + 1
                u = u + 1

            for k in range(u, u + second_off):
                res += f"__m256 v{i}_{k} = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k3+{k})*nb + i1+{i}]);\n"

            for k in range(u, u + second_off):
                for j in range(0, tu, 8):
                    res += f"__m256i ws{j}_{k} = _mm256_srli_epi32(w{wid}_{j}, {bits*k-wid*32-shift});\n"

                for j in range(0, tu, 8):
                    res += f"__m256i wsa{j}_{k} = _mm256_and_si256(ws{j}_{k}, mask);\n"

                for j in range(0, tu, 8):
                    res += f"__m256 l{j}_{k} = _mm256_cvtepi32_ps(wsa{j}_{k});\n"

                for j in range(0, tu, 8):
                    res += f"acc{i}_{j} = _mm256_fmadd_ps(v{i}_{k}, l{j}_{k}, acc{i}_{j});\n"

            u = u + 2

        return res

    else:
        for j in range(0, tu, 8):
            res += f"__m256i w{j} = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/{packed} + k*mb*tb/{packed} + k3*tb/{packed} + j1+{j}]);\n"

        for u in range(packed - unroll, -1, -unroll):
            for k in range(u + unroll - 1, u - 1, -1):
                res += f"__m256 v{i}_{k} = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k3+{k})*nb + i1+{i}]);\n"

            for k in range(u, u + unroll):
                for j in range(0, tu, 8):
                    res += f"__m256i ws{j}_{k} = _mm256_srli_epi32(w{j}, {bits*k});\n"

                for j in range(0, tu, 8):
                    res += f"__m256i wsa{j}_{k}= _mm256_and_si256(ws{j}_{k}, mask);\n"

                for j in range(0, tu, 8):
                    res += f"__m256 l{j}_{k} = _mm256_cvtepi32_ps(wsa{j}_{k});\n"

                for j in range(0, tu, 8):
                    res += f"acc{i}_{j} = _mm256_fmadd_ps(v{i}_{k}, l{j}_{k}, acc{i}_{j});\n"

        return res


def block(nu_iter, mu, tu, rho, packed, unroll, bits):
    res = ""
    i = 0
    # unroll = 4 # number of bcasts and unpacks
    if bits == 3:
        for j in range(0, tu, 8):
            res += f"__m256i w0_{j} = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/{packed}*3 + k*mb*tb/{packed}*3 + k2*tb/{packed}*3 + jw+{j*3}]);\n"
            res += f"__m256i w1_{j} = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/{packed}*3 + k*mb*tb/{packed}*3 + k2*tb/{packed}*3 + jw+{j*3}+8]);\n"
            res += f"__m256i w2_{j} = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/{packed}*3 + k*mb*tb/{packed}*3 + k2*tb/{packed}*3 + jw+{j*3}+16]);\n"

        u = 0
        first_off = 3
        second_off = 2
        wid = 0
        shift = 0
        while u < 32:
            if u == 10:
                res += f"__m256 v{i}_{u} = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+{u})*nb + i1+{i}]);\n"

                for j in range(0, tu, 8):
                    res += f"__m256i ws{j}_10 = _mm256_srli_epi32(w0_{j}, {bits*10});\n"
                    res += f"__m256i temp0_{j} = _mm256_slli_epi32(w1_{j}, 2);\n"
                    res += f"temp0_{j} = _mm256_and_si256(temp0_{j}, mask);\n"
                    res += f"ws{j}_10 = _mm256_or_si256(ws{j}_10, temp0_{j});\n"

                    res += f"__m256i wsa{j}_{u} = _mm256_and_si256(ws{j}_{u}, mask);\n"

                    res += f"__m256 l{j}_{u} = _mm256_cvtepi32_ps(wsa{j}_{u});\n"

                    res += f"acc{i}_{j} = _mm256_fmadd_ps(v{i}_{u}, l{j}_{u}, acc{i}_{j});\n"

                wid = wid + 1
                u = u + 1

            elif u == 21:
                res += f"__m256 v{i}_{u} = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+{u})*nb + i1+{i}]);\n"

                for j in range(0, tu, 8):
                    res += f"__m256i ws{j}_{u} = _mm256_srli_epi32(w1_{j}, 31);\n"
                    res += f"__m256i temp1_{j} = _mm256_slli_epi32(w2_{j}, 1);\n"
                    res += f"temp1_{j} = _mm256_and_si256(temp1_{j}, mask);\n"
                    res += f"ws{j}_{u} = _mm256_or_si256(ws{j}_{u}, temp1_{j});\n"

                    res += f"__m256i wsa{j}_{u} = _mm256_and_si256(ws{j}_{u}, mask);\n"

                    res += f"__m256 l{j}_{u} = _mm256_cvtepi32_ps(wsa{j}_{u});\n"

                    res += f"acc{i}_{j} = _mm256_fmadd_ps(v{i}_{u}, l{j}_{u}, acc{i}_{j});\n"

                wid = wid + 1
                u = u + 1

            for k in range(u, u + second_off):
                res += f"__m256 v{i}_{k} = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+{k})*nb + i1+{i}]);\n"

            for k in range(u, u + second_off):
                for j in range(0, tu, 8):
                    res += f"__m256i ws{j}_{k} = _mm256_srli_epi32(w{wid}_{j}, {bits*k-wid*32-shift});\n"

                for j in range(0, tu, 8):
                    res += f"__m256i wsa{j}_{k} = _mm256_and_si256(ws{j}_{k}, mask);\n"

                for j in range(0, tu, 8):
                    res += f"__m256 l{j}_{k} = _mm256_cvtepi32_ps(wsa{j}_{k});\n"

                for j in range(0, tu, 8):
                    res += f"acc{i}_{j} = _mm256_fmadd_ps(v{i}_{k}, l{j}_{k}, acc{i}_{j});\n"

            u = u + 2

        return res

    else:
        for j in range(0, tu, 8):
            res += f"__m256i w{j} = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/{packed} + k*mb*tb/{packed} + k2*tb/{packed} + j1+{j}]);\n"

        for u in range(packed - unroll, -1, -unroll):
            for k in range(u + unroll - 1, u - 1, -1):
                res += f"__m256 v{i}_{k} = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+{k})*nb + i1+{i}]);\n"

            for k in range(u, u + unroll):
                for j in range(0, tu, 8):
                    res += f"__m256i ws{j}_{k} = _mm256_srli_epi32(w{j}, {bits*k});\n"

                for j in range(0, tu, 8):
                    res += f"__m256i wsa{j}_{k}= _mm256_and_si256(ws{j}_{k}, mask);\n"

                for j in range(0, tu, 8):
                    res += f"__m256 l{j}_{k} = _mm256_cvtepi32_ps(wsa{j}_{k});\n"

                for j in range(0, tu, 8):
                    res += f"acc{i}_{j} = _mm256_fmadd_ps(v{i}_{k}, l{j}_{k}, acc{i}_{j});\n"

        return res


def accumulators_f(nu, tu, gs=False):
    accumulators = ""
    for i in range(nu):
        for j in range(0, tu, 8):
            if gs:
                accumulators += f"__m256 acc{i}_{j} = _mm256_setzero_ps();\n"
            else:
                accumulators += (
                    f"__m256 acc{i}_{j} = _mm256_loadu_ps(&output[base_output + j + (i1+{i})*t + j1+{j}]);\n"
                )
    return accumulators


def stores_f(nu, tu, gs=False):
    store = ""
    if gs:
        for i in range(nu):
            for j in range(0, tu, 8):
                store += f"__m256 o{i}_{j} = _mm256_loadu_ps(&output[base_output + j + (i1+{i})*t + j1+{j}]);\n"

        for i in range(nu):
            for j in range(0, tu, 8):
                store += (
                    f"__m256 s{i}_{j} = _mm256_loadu_ps(&scales[(k*mb+k1)/gs * t + base_output + j + j1+{j}]);\n"
                )

        for i in range(nu):
            for j in range(0, tu, 8):
                store += f"__m256 f{i}_{j} = _mm256_fmadd_ps(acc{i}_{j}, s{i}_{j}, o{i}_{j});\n"

        for i in range(nu):
            for j in range(0, tu, 8):
                store += f"_mm256_storeu_ps(&output[base_output + j + (i1+{i})*t + j1+{j}], f{i}_{j});\n"
    else:
        for i in range(nu):
            for j in range(0, tu, 8):
                store += f"_mm256_storeu_ps(&output[base_output + j + (i1+{i})*t + j1+{j}], acc{i}_{j});\n"
    return store


def qforward(
    nu, mu, tu, p, unroll, bits, n=0, m=0, t=0, nb=0, mb=0, tb=0, tt=0, cutoff=-1, gs=False, gs_val=-1, module=True
):
    assert module or (gs and gs_val != -1) or (not gs and gs_val == -1)
    if cutoff == -1:
        cutoff = p + 1
    # packed = 32 // bits
    if bits == 3:
        packed = 32
        loopguard = packed
    else:
        packed = 32 // bits
        loopguard = packed
    # compute the parameters from the model

    accumulators = accumulators_f(nu, tu, gs)
    store = stores_f(nu, tu, gs)

    ugemm = ""
    if gs:
        ugemm += "int j1 = 0;\n"
        if bits == 3:
            ugemm += "int jw = 0;\n"
            ugemm += f"for(; j1 < tb-tu+1; j1+=tu, jw+={tu*3})"
            ugemm += "{\n"
        else:
            ugemm += "for(; j1 < tb-tu+1; j1+=tu) {\n"
        ugemm += "for(int k1 = 0; k1 < mb; k1+=gs) {\n"
        ugemm += accumulators
        ugemm += f"for(int k2 = k1; k2 < k1+gs; k2+={loopguard})\n"
        ugemm += "{\n"
        ugemm += block(nu, mu, tu, 16, packed, unroll, bits)
        ugemm += "}\n"
        ugemm += store
        ugemm += "}\n"
        ugemm += "}\n"
    else:
        ugemm += "int j1 = 0;\n"
        if bits == 3:
            ugemm += "int jw = 0;\n"
            ugemm += f"for(; j1 < tb-tu+1; j1+=tu, jw+={tu*3})"
            ugemm += "{\n"
        else:
            ugemm += "for(; j1 < tb-tu+1; j1+=tu) {\n"
        ugemm += accumulators
        ugemm += "for(int k1 = 0; k1 < mb; k1+=mu) {\n"
        ugemm += f"for(int k2 = k1; k2 < k1+mu; k2+={loopguard})"
        ugemm += "{\n"
        ugemm += block(nu, mu, tu, 16, packed, unroll, bits)
        ugemm += "}\n"
        ugemm += "}\n"
        ugemm += store
        ugemm += "}\n"

    res = ""
    res += "inline\n"
    if gs:
        res += f"void q{bits}gemm_gs(const float* __restrict__ input, \n"
    else:
        res += f"void q{bits}gemm(const float* __restrict__ input, \n"
    res += "const int* __restrict__ W, \n"
    res += "const float* __restrict__ scales, \n"
    res += "const float* __restrict__ zeros, \n"
    res += "const float* __restrict__ bias, \n "
    res += "const float* __restrict__ sums, \n "
    res += "float* __restrict__ output,\n\
const int n,\n\
const int m,\n\
const int t,\n\
const int nb,\n\
const int mb,\n\
const int tb,\n\
int ogtt,\n"
    if gs:
        res += "const int gs,\n"
    res += "const int cutoff){\n"

    res += f"#pragma omp parallel num_threads({p})\n"
    res += "{\n"
    res += "int tid;\n"
    res += f"const int mu = {mu};\n"
    res += f"const int nu = {nu};\n"
    res += f"const int tu = {tu};\n"
    res += f"const int on = n / nb;\n"
    res += f"const int om = m / mb;\n"

    mask = (2**bits) - 1
    res += f"const __m256i mask = _mm256_set1_epi32({mask});\n"
    if bits == 3:
        res += f"const __m256i mask4 = _mm256_set1_epi32(4);\n"
        res += f"const __m256i mask6 = _mm256_set1_epi32(6);\n"
    res += "tid = omp_get_thread_num();\n"

    res += "int tt = ogtt;\n"
    res += "if(tid >= cutoff){\n"
    res += f"tt -= tb;\n"
    res += "}\n"
    res += f"const int base_output = tid >= cutoff ?\n \
(tid-cutoff)*tt + (tt+tb)*cutoff: \n \
tid*tt;\n"  # is this >= cutoff or > cutoff?
    if bits != 3:
        res += f"const int base_W = tid >= cutoff ?\n \
((tid-cutoff)*tt + (tt+tb)*cutoff)*m/{packed}: \n \
tid*tt*m/{packed};\n"
    else:
        res += f"const int base_W = tid >= cutoff ?\n \
((tid-cutoff)*tt + (tt+tb)*cutoff)*m/{packed}*3: \n \
tid*tt*m/{packed}*3;\n"

    res += "for(int j = 0; j < tt; j+=tb){\n"
    res += "for(int i = 0; i < on; i++) {\n"
    res += "for(int k = 0; k < om; k++) {\n"
    res += "for(int i1 = 0; i1 < nb; i1+=nu) {\n"
    res += ugemm
    res += "}\n"
    res += "}\n"
    res += "}\n"
    res += "}\n"
    res += "#pragma omp barrier\n"
    # res += "#pragma omp for\n"
    if gs:
        res += "const int ngs = m/gs;\n"
        res += "for (int i = 0; i < n; i++) {\n"
        res += f"for (int j = 0; j < tt; j+={tu})"
        res += "{\n"
        for i in range(0, tu, 8):
            res += f"__m256 acc{i} = _mm256_setzero_ps();\n"
        res += "for (int i1 = 0; i1 < ngs; i1++){\n"
        res += "__m256 r = _mm256_set1_ps(sums[i*ngs + i1]);\n"
        for i in range(0, tu, 8):
            res += f"__m256 z{i} = _mm256_loadu_ps(&zeros[base_output + i1* t + j + {i}]);\n"
        # if not module:
        if bits != 3 or not module:
            for i in range(0, tu, 8):
                res += f"__m256 s{i} = _mm256_loadu_ps(&scales[base_output + i1 * t + j + {i}]);\n"
            for i in range(0, tu, 8):
                res += f"__m256 zs{i} = _mm256_mul_ps(z{i}, s{i});\n"
        for i in range(0, tu, 8):
            # if module:
            if bits == 3 and module:
                res += f"acc{i} = _mm256_fmadd_ps(z{i}, r, acc{i});\n"
            else:
                res += f"acc{i} = _mm256_fmadd_ps(zs{i}, r, acc{i});\n"
        res += "}\n"
        for i in range(0, tu, 8):
            res += f"__m256 o{i} = _mm256_loadu_ps(&output[i*t + base_output + j + {i}]);\n"
        for i in range(0, tu, 8):
            res += f"__m256 b{i} = _mm256_loadu_ps(&bias[base_output + j + {i}]);\n"
        for i in range(0, tu, 8):
            if module:
                res += f"__m256 o1{i} = _mm256_sub_ps(o{i}, acc{i});\n"
            else:
                res += f"__m256 o1{i} = _mm256_add_ps(o{i}, acc{i});\n"
        for i in range(0, tu, 8):
            res += f"__m256 o2{i} = _mm256_add_ps(o1{i}, b{i});\n"
        for i in range(0, tu, 8):
            res += f"_mm256_storeu_ps(&output[i*t + base_output + j + {i}], o2{i});\n"
        res += "}\n"
        res += "}\n"
        res += "}\n"
        res += "}\n"
    else:
        res += "for (int i = 0; i < n; i++) {\n"
        res += "__m256 r = _mm256_set1_ps(sums[i]);\n"
        res += f"for (int j = 0; j < tt; j+={tu})"
        res += "{\n"
        for i in range(0, tu, 8):
            res += f"__m256 o{i} = _mm256_loadu_ps(&output[i*t + base_output + j + {i}]);\n"
        for i in range(0, tu, 8):
            res += f"__m256 z{i} = _mm256_loadu_ps(&zeros[base_output + j + {i}]);\n"
        for i in range(0, tu, 8):
            res += f"__m256 b{i} = _mm256_loadu_ps(&bias[base_output + j + {i}]);\n"
        for i in range(0, tu, 8):
            res += f"__m256 s{i} = _mm256_loadu_ps(&scales[base_output + j + {i}]);\n"
        if bits == 3 and module:
            for i in range(0, tu, 8):
                res += f"__m256 os{i} = _mm256_mul_ps(o{i}, s{i});\n"
        for i in range(0, tu, 8):
            if module:
                if bits == 3:
                    res += f"__m256 zr{i} = _mm256_fnmadd_ps(z{i}, r, os{i});\n"
                else:
                    res += f"__m256 zr{i} = _mm256_fnmadd_ps(z{i}, r, o{i});\n"
            else:
                res += f"__m256 zr{i} = _mm256_fmadd_ps(z{i}, r, o{i});\n"
        for i in range(0, tu, 8):
            # j           res += f"__m256 o2{i} = _mm256_mul_ps(zr{i}, s{i});\n"
            if bits == 3 and module:
                res += f"__m256 o2{i} = _mm256_add_ps(zr{i}, b{i});\n"
            else:
                res += f"__m256 o2{i} = _mm256_fmadd_ps(zr{i}, s{i}, b{i});\n"
        for i in range(0, tu, 8):
            res += f"_mm256_storeu_ps(&output[i*t + base_output + j + {i}], o2{i});\n"
        res += "}\n"
        res += "}\n"
        res += "}\n"
        res += "}\n"

    # wrapper for qgemm if we call from cpp
    if module:
        if gs:
            res += f"inline void forward{bits}_gs_cpu(\n"
        else:
            res += f"inline void forward{bits}_cpu(\n"
        res += "torch::Tensor in, torch::Tensor weight, torch::Tensor out,\n"
        res += "torch::Tensor bias, torch::Tensor scales, torch::Tensor zeros, torch::Tensor sums,\n"
        if gs:
            res += "int N, int M, int T, int nb, int mb, int tb, int tt, int groupsize, int cutoff){\n"
        else:
            res += "int N, int M, int T, int nb, int mb, int tb, int tt, int cutoff){\n"
        res += "int*   W = weight.data_ptr<int>();\n"
        res += "float* input = in.data_ptr<float>();\n"
        res += "float* b   = bias.data_ptr<float>();\n"
        res += "float* s   = scales.data_ptr<float>();\n"
        res += "float* z   = zeros.data_ptr<float>();\n"
        res += "float* r   = sums.data_ptr<float>();\n"
        res += "float* O   = out.data_ptr<float>();\n"
        res += "\n"
        if gs:
            res += f"q{bits}gemm_gs(input, W, s, z, b, r, O, N, M, T, nb, mb, tb, tt, groupsize, cutoff);\n"
        else:
            res += f"q{bits}gemm(input, W, s, z, b, r, O, N, M, T, nb, mb, tb, tt, cutoff);\n"
        res += "}\n"
    else:
        res += "inline void qforward(const float* __restrict__ input, \n \
const int* __restrict__ W, \n\
const float* __restrict__ scales, \n\
const float* __restrict__ zeros, \n\
const float* __restrict__ bias, \n\
const float* __restrict__ sums, \n\
float* __restrict__ output, \n\
int n, \n \
int m, \n \
int t) {\n"
        if gs:
            res += f"q{bits}gemm_gs(input, W, scales, zeros, bias, sums, output, n, m, t, {nb}, {mb}, {tb}, {tt}, {gs_val}, {cutoff});\n"
        else:
            res += f"q{bits}gemm(input, W, scales, zeros, bias, sums, output, n, m, t, {nb}, {mb}, {tb}, {tt}, {cutoff});\n"
        res += "}\n"
    return res


def gen_model(n, m, t, bits, p, gs):
    # get parameters
    if bits == 3:
        packed = 32
        unroll = 3
        nu = 1  # args.n
        mu = 32
        tu = 32
    else:
        packed = 32 // bits
        unroll = 2
        nu = 1  # args.n
        mu = 16
        tu = 32

    # compute the parameters from the model

    nb = n  # it's always small for transformers

    mb, tb = mem_model(n, m, t, mu, tu, bits, l1, p, gs)

    split = np.ones(p)
    split = split * tb
    while np.sum(split) < t:
        split = split + tb

    idx = p - 1
    while np.sum(split) > t:
        split[idx] = split[idx] - tb
        idx = idx - 1

    assert np.sum(split) == t

    split = split.astype(int)
    tt = int(split[0])

    if split[0] == split[-1]:
        cutoff = int(p + 1)
    else:
        cutoff = int(idx + 1)

    if gs == -1:
        code = qforward(
            nu,
            mu,
            tu,
            p,
            unroll,
            n=n,
            m=m,
            t=t,
            nb=nb,
            mb=mb,
            tb=tb,
            tt=tt,
            bits=bits,
            cutoff=cutoff,
            module=False,
        )
    else:
        code = qforward(
            nu,
            mu,
            tu,
            p,
            unroll,
            n=n,
            m=m,
            t=t,
            nb=nb,
            mb=mb,
            tb=tb,
            tt=tt,
            bits=bits,
            cutoff=cutoff,
            gs=True,
            gs_val=gs,
            module=False,
        )
    code += pack_in(n, m, nb, mb)
    # code += pack_qw(m, t, mb, tb, tb, bits=bits)#, cutoff=cutoff)
    code += pack_qw(m, t, mb, tb, tu, bits=bits)
    code += pack_out(n, t, nb, tb)
    code += print_parameters(bits, n, m, t, nb, mb, tb, mu, nu, tu, unroll, p)

    with open("./autogptq_extension/qigen/forward.h", "w") as f:
        f.write(macros())
        f.write(code)


def gen_and_compile(n, m, t, nb, mb, tb, nu, mu, tu, p, unroll, bits=4, gs=-1, module=False):
    split = np.ones(p)
    split = split * tb
    while np.sum(split) < t:
        split = split + tb

    idx = p - 1
    while np.sum(split) > t:
        split[idx] = split[idx] - tb
        idx = idx - 1

    assert np.sum(split) == t

    split = split.astype(int)
    tt = int(split[0])

    if split[0] == split[-1]:
        cutoff = int(p + 1)
    else:
        cutoff = int(idx + 1)

    if gs == -1:
        code = qforward(
            nu,
            mu,
            tu,
            p,
            unroll,
            n=n,
            m=m,
            t=t,
            nb=nb,
            mb=mb,
            tb=tb,
            tt=tt,
            bits=bits,
            cutoff=cutoff,
            module=False,
        )
    else:
        code = qforward(
            nu,
            mu,
            tu,
            p,
            unroll,
            n=n,
            m=m,
            t=t,
            nb=nb,
            mb=mb,
            tb=tb,
            tt=tt,
            bits=bits,
            cutoff=cutoff,
            gs=True,
            gs_val=gs,
            module=False,
        )
    code += pack_in(n, m, nb, mb)
    code += pack_qw(m, t, mb, tb, tu, bits=bits)
    code += pack_out(n, t, nb, tb)
    if module:
        code += print_parameters_module(bits, mu, nu, tu, unroll, p, gs=gs)
    else:
        code += print_parameters(bits, n, m, t, nb, mb, tb, mu, nu, tu, unroll, p, gs=gs)

    # write the code to a file called forward.h
    with open("./autogptq_extension/qigen/forward.h", "w") as f:
        f.write(macros())
        f.write(code)

    # g++ mmm_test.cpp -O3 -ftree-vectorize -mfma -mavx -mavx2 -fno-signaling-nans -fno-trapping-math -fopenmp -o mmm_test
    start = time.time()
    if not module:
        subprocess.call(
            [
                "g++",
                "-O3",
                "-o",
                "./autogptq_extension/qigen/mmm_test",
                "./autogptq_extension/qigen/mmm_test.cpp",
                "-mavx",
                "-mfma",
                "-mavx2",
                "-ftree-vectorize",
                "-fno-signaling-nans",
                "-fno-trapping-math",
                "-march=native",
                "-fopenmp",
            ]
        )
        subprocess.call(["./autogptq_extension/qigen/mmm_test", f"{n}", f"{m}", f"{t}", f"{bits}", f"{gs}"])
    else:
        subprocess.call(
            [
                "g++",
                "-O3",
                "-o",
                "./autogptq_extension/qigen/mmm",
                "./autogptq_extension/qigen/mmm.cpp",
                "-mavx",
                "-mfma",
                "-mavx2",
                "-ftree-vectorize",
                "-fno-signaling-nans",
                "-fno-trapping-math",
                "-march=native",
                "-fopenmp",
            ]
        )
        subprocess.call(["./autogptq_extension/qigen/mmm", f"{n}", f"{m}", f"{t}", f"{bits}", f"{gs}"])
        # subprocess.call(["./autogptq_extension/qigen/mmm", f"{n}", f"{m}", f"{t}", f"{bits}", f"{gs}", ">>", "./autogptq_extension/qigen/tmp.csv"])
    end = time.time() - start
    return end


def grid():
    tt = 64
    for p in [32]:
        # for n in [1, 10]:
        for n in [1]:
            for m in [4096]:
                for t in [4096]:
                    # for mb in range(1,m):
                    # for mb in range(32,512,32):
                    # for mb in [64, 128, 256, 512, 1024, 2048]:
                    for mb in [512, 1024, 2048]:
                        if m % mb == 0:
                            # for tb in range(8,t,8):
                            # for tb in range(32,512,32):
                            # for tb in [16, 32, 64]:#, 128, 192, 256]:
                            # for tb in [32]:#, 128, 192, 256]:
                            for tb in [128, 256]:
                                if t % tb == 0:
                                    # for mu in range(32,mb,32):
                                    for mu in [16, 32]:
                                        if mb % mu == 0:
                                            # for tu in range(8,tb,8):
                                            # for tu in [16, 32]:
                                            for tu in [16, 32, 64, 128]:
                                                if tb % tu == 0:
                                                    for gs in [-1, 128, 64, 32, 16]:
                                                        # for bits in [2, 3, 4]:
                                                        for bits in [4, 3, 2]:
                                                            if bits == 3:
                                                                for u in [5]:
                                                                    gen_and_compile(
                                                                        n,
                                                                        m,
                                                                        t,
                                                                        n,
                                                                        mb,
                                                                        tb,
                                                                        1,
                                                                        mu,
                                                                        tu,
                                                                        p,
                                                                        u,
                                                                        bits=bits,
                                                                        gs=gs,
                                                                    )
                                                            else:
                                                                for u in [1, 2, 4, 8]:
                                                                    gen_and_compile(
                                                                        n,
                                                                        m,
                                                                        t,
                                                                        n,
                                                                        mb,
                                                                        tb,
                                                                        1,
                                                                        mu,
                                                                        tu,
                                                                        p,
                                                                        u,
                                                                        bits=bits,
                                                                        gs=gs,
                                                                    )


def forward_module_gs(nu, mu, tu, p, unroll, bits):
    # packed = 32 // bits
    if bits == 3:
        packed = 32
        loopguard = packed
    else:
        packed = 32 // bits
        loopguard = packed
    # compute the parameters from the model

    accumulators = ""
    for i in range(nu):
        for j in range(0, tu, 8):
            accumulators += f"__m256 acc{i}_{j} = _mm256_setzero_ps();\n"

    store = ""
    for i in range(nu):
        for j in range(0, tu, 8):
            store += f"__m256 o{i}_{j} = _mm256_loadu_ps(&output[base_output + j + (i1+{i})*t + j1+{j}]);\n"

    for i in range(nu):
        for j in range(0, tu, 8):
            store += f"__m256 s{i}_{j} = _mm256_loadu_ps(&scales[(k*mb+k1)/gs * t + base_output + j + j1+{j}]);\n"

    for i in range(nu):
        for j in range(0, tu, 8):
            store += f"__m256 f{i}_{j} = _mm256_fmadd_ps(acc{i}_{j}, s{i}_{j}, o{i}_{j});\n"

    for i in range(nu):
        for j in range(0, tu, 8):
            store += f"_mm256_storeu_ps(&output[base_output + j + (i1+{i})*t + j1+{j}], f{i}_{j});\n"

    ugemm = ""
    if bits == 3:
        ugemm += "int j1 = 0;\n"
        ugemm += "int jw = 0;\n"
        ugemm += f"for(; j1 < tb-tu+1; j1+=tu, jw+={tu*3})"
        ugemm += "{\n"
    else:
        ugemm += "int j1 = 0;\n"
        ugemm += "for(; j1 < tb-tu+1; j1+=tu) {\n"
    ugemm += "for(int k1 = 0; k1 < mb; k1+=gs) {\n"
    ugemm += accumulators
    ugemm += f"for(int k2 = k1; k2 < k1+gs; k2+={loopguard})\n"
    ugemm += "{\n"
    ugemm += block(nu, mu, tu, 16, packed, unroll, bits)
    ugemm += "}\n"
    ugemm += store
    ugemm += "}\n"
    ugemm += "}\n"

    res = ""
    res += "inline\n"
    res += f"void q{bits}gemm_gs(const float* __restrict__ input, \n"
    res += "        const int* __restrict__ W, \n \
            const float* __restrict__ scales, \n"
    res += "const float* __restrict__ zeros, \n"
    res += "        const float* __restrict__ bias, \n "
    res += "        const float* __restrict__ sums,\n"
    res += "        float* __restrict__ output,\n \
            const int n,\n \
            const int m,\n \
            const int t,\n \
            const int nb,\n \
            const int mb,\n \
            const int tb,\n \
            int ogtt,\n \
            const int gs,\n\
            const int cutoff){\n"

    res += f"#pragma omp parallel num_threads({p})\n"
    res += "{\n"
    res += "  int tid;\n"
    res += f"  const int mu = {mu};\n"
    res += f"  const int nu = {nu};\n"
    res += f"  const int tu = {tu};\n"
    res += f"  const int on = n / nb;\n"
    res += f"  const int om = m / mb;\n"

    mask = (2**bits) - 1
    res += f"const __m256i mask = _mm256_set1_epi32({mask});\n"
    if bits == 3:
        res += f"const __m256i mask4 = _mm256_set1_epi32(4);\n"
        res += f"const __m256i mask6 = _mm256_set1_epi32(6);\n"
    res += "tid = omp_get_thread_num();\n"

    res += "int tt = ogtt;\n"
    res += "if(tid >= cutoff){\n"
    res += f"tt -= tb;\n"
    res += "}\n"
    res += f"const int base_output = tid >= cutoff ?\n \
(tid-cutoff)*tt + (tt+tb)*cutoff: \n \
tid*tt;\n"  # is this >= cutoff or > cutoff?
    if bits != 3:
        res += f"const int base_W = tid >= cutoff ?\n \
((tid-cutoff)*tt + (tt+tb)*cutoff)*m/{packed}: \n \
tid*tt*m/{packed};\n"
    else:
        res += f"const int base_W = tid >= cutoff ?\n \
((tid-cutoff)*tt + (tt+tb)*cutoff)*m/{packed}*3: \n \
tid*tt*m/{packed}*3;\n"

    res += "for(int j = 0; j < tt; j+=tb){\n"
    res += "for(int i = 0; i < on; i++) {\n"
    res += "for(int k = 0; k < om; k++) {\n"
    res += "for(int i1 = 0; i1 < nb; i1+=nu) {\n"
    res += ugemm
    res += "}\n"
    res += "}\n"
    res += "}\n"
    res += "}\n"
    res += "const int ngs = m/gs;\n"
    res += "#pragma omp barrier\n"
    # res += "#pragma omp for collapse(2)\n"
    res += "for (int i = 0; i < n; i++) {\n"
    # res += f"    for (int j = 0; j < t; j+={tu})"
    res += f"for (int j = 0; j < tt; j+={tu})"
    res += "{\n"
    # for i in range(0,tu,8):
    # res += f"__m256 o{i} = _mm256_loadu_ps(&output[i*t + j + {i}]);\n"
    for i in range(0, tu, 8):
        res += f"__m256 acc{i} = _mm256_setzero_ps();\n"
    res += "for (int i1 = 0; i1 < ngs; i1++){\n"
    res += "__m256 r = _mm256_set1_ps(sums[i*ngs + i1]);\n"
    for i in range(0, tu, 8):
        # res += f"__m256 z{i} = _mm256_loadu_ps(&zeros[i1 * t + j + {i}]);\n"
        res += f"__m256 z{i} = _mm256_loadu_ps(&zeros[base_output + i1* t + j + {i}]);\n"
    # for i in range(0,tu,8):
    # res += f"__m256 s{i} = _mm256_loadu_ps(&scales[i1 * t + j + {i}]);\n"
    # for i in range(0,tu,8):
    # res += f"__m256 zr{i} = _mm256_mul_ps(z{i}, r);\n"
    # for i in range(0,tu,8):
    # res += f"acc{i} = _mm256_fmadd_ps(zr{i}, s{i}, acc{i});\n"
    for i in range(0, tu, 8):
        res += f"acc{i} = _mm256_fmadd_ps(z{i}, r, acc{i});\n"
    # for i in range(0,tu,8):
    # res += f"__m256 zr{i} = _mm256_mul_ps(z{i}, r);\n"
    # for i in range(0,tu,8):
    # res += f"o{i} = _mm256_fnmadd_ps(zr{i}, s{i}, o{i});\n"
    res += "}\n"
    for i in range(0, tu, 8):
        # res += f"__m256 o{i} = _mm256_loadu_ps(&output[i*t + j + {i}]);\n"
        res += f"__m256 o{i} = _mm256_loadu_ps(&output[i*t + base_output + j + {i}]);\n"
    for i in range(0, tu, 8):
        res += f"__m256 o1{i} = _mm256_sub_ps(o{i}, acc{i});\n"
    for i in range(0, tu, 8):
        # res += f"_mm256_storeu_ps(&output[i*t + j + {i}], o1{i});\n"
        res += f"_mm256_storeu_ps(&output[i*t + base_output + j + {i}], o1{i});\n"
    res += "  }\n"
    res += "}\n"
    res += "}\n"
    res += "}\n"

    # wrapper for qgemm if we call from cpp
    res += f"inline void forward{bits}_gs_cpu(\n"
    res += "torch::Tensor in, torch::Tensor weight, torch::Tensor out,\n"
    res += "torch::Tensor bias, torch::Tensor scales, torch::Tensor zeros, torch::Tensor sums,\n"
    res += "int N, int M, int T, int nb, int mb, int tb, int tt, int groupsize, int cutoff){\n"
    res += "int*   W = weight.data_ptr<int>();\n"
    res += "float* input = in.data_ptr<float>();\n"
    res += "float* b   = bias.data_ptr<float>();\n"
    res += "float* s   = scales.data_ptr<float>();\n"
    # res +=  "int* z   = zeros.data_ptr<int>();\n"
    res += "float* z   = zeros.data_ptr<float>();\n"
    res += "float* r   = sums.data_ptr<float>();\n"
    res += "float* O   = out.data_ptr<float>();\n"
    res += "\n"
    res += f"q{bits}gemm_gs(input, W, s, z, b, r, O, N, M, T, nb, mb, tb, tt, groupsize, cutoff);\n"
    res += "}\n"
    return res


def forward_module(nu, mu, tu, p, unroll, bits):
    # packed = 32 // bits
    if bits == 3:
        packed = 32
        loopguard = packed
    else:
        packed = 32 // bits
        loopguard = packed
    # compute the parameters from the model

    accumulators = ""
    for i in range(nu):
        for j in range(0, tu, 8):
            accumulators += (
                f"__m256 acc{i}_{j} = _mm256_loadu_ps(&output[base_output + j + (i1+{i})*t + j1+{j}]);\n"
            )

    store = ""
    for i in range(nu):
        for j in range(0, tu, 8):
            store += f"_mm256_storeu_ps(&output[base_output + j + (i1+{i})*t + j1+{j}], acc{i}_{j});\n"

    ugemm = ""
    if bits == 3:
        ugemm += "int jw = 0;\n"
        ugemm += f"for(; j1 < tb-tu+1; j1+=tu, jw+={tu*3})"
        ugemm += "{\n"
    else:
        ugemm += "for(; j1 < tb-tu+1; j1+=tu) {\n"
    ugemm += accumulators
    ugemm += "for(int k1 = 0; k1 < mb; k1+=mu) {\n"
    ugemm += f"for(int k2 = k1; k2 < k1+mu; k2+={loopguard})"
    ugemm += "{\n"
    ugemm += block(nu, mu, tu, 16, packed, unroll, bits)
    ugemm += "}\n"
    ugemm += "}\n"
    ugemm += store
    ugemm += "}\n"

    res = ""
    res += "inline\n"
    res += f"void q{bits}gemm(const float* __restrict__ input, \n"
    res += "const int* __restrict__ W, \n"
    res += "const float* __restrict__ scales, \n"
    # res += "const int* __restrict__ zeros, \n"
    res += "const float* __restrict__ zeros, \n"
    res += "const float* __restrict__ bias, \n "
    res += "const float* __restrict__ sums,"
    res += "float* __restrict__ output,\n \
const int n,\n \
const int m,\n \
const int t,\n \
const int nb,\n \
const int mb,\n \
const int tb,\n \
int ogtt,\n \
const int cutoff){\n"

    res += f"#pragma omp parallel num_threads({p})\n"
    res += "{\n"
    res += "int tid, nthreads;\n"
    res += f"const int mu = {mu};\n"
    res += f"const int nu = {nu};\n"
    res += f"const int tu = {tu};\n"
    res += f"const int on = n / nb;\n"
    res += f"const int om = m / mb;\n"

    mask = (2**bits) - 1
    res += f"const __m256i mask = _mm256_set1_epi32({mask});\n"
    if bits == 3:
        res += f"const __m256i mask4 = _mm256_set1_epi32(4);\n"
        res += f"const __m256i mask6 = _mm256_set1_epi32(6);\n"
    res += "tid = omp_get_thread_num();\n"
    # res +=  "  std::cout << \"thread \" << tid << \" started\" << std::endl;\n"
    res += "nthreads = omp_get_num_threads();\n"

    res += "int tt = ogtt;\n"
    res += "if(tid >= cutoff){\n"
    res += f"tt -= tb;\n"
    res += "}\n"
    res += f"const int base_output = tid >= cutoff ?\n \
(tid-cutoff)*tt + (tt+tb)*cutoff: \n \
tid*tt;\n"  # is this >= cutoff or > cutoff?
    if bits != 3:
        res += f"const int base_W = tid >= cutoff ?\n \
((tid-cutoff)*tt + (tt+tb)*cutoff)*m/{packed}: \n \
tid*tt*m/{packed};\n"
    else:
        res += f"const int base_W = tid >= cutoff ?\n \
((tid-cutoff)*tt + (tt+tb)*cutoff)*m/{packed}*3: \n \
tid*tt*m/{packed}*3;\n"

    res += "for(int j = 0; j < tt; j+=tb){\n"
    res += "for(int i = 0; i < on; i++) {\n"
    res += "for(int k = 0; k < om; k++) {\n"
    res += "for(int i1 = 0; i1 < nb; i1+=nu) {\n"
    res += "int j1 = 0;\n"
    res += ugemm
    res += "}\n"
    res += "}\n"
    res += "}\n"
    res += "}\n"
    # res += "#pragma omp barrier\n"
    # res += "#pragma omp for\n"
    res += "for (int i = 0; i < n; i++) {\n"
    res += "__m256 r = _mm256_set1_ps(sums[i]);\n"
    # res += f"for (int j = 0; j < t; j+={tu})"
    res += f"for (int j = 0; j < tt; j+={tu})"
    res += "{\n"
    for i in range(0, tu, 8):
        # res += f"__m256 o{i} = _mm256_loadu_ps(&output[i*t + j + {i}]);\n"
        res += f"__m256 o{i} = _mm256_loadu_ps(&output[i*t + base_output + j + {i}]);\n"
    for i in range(0, tu, 8):
        res += f"__m256 z{i} = _mm256_loadu_ps(&zeros[base_output + j + {i}]);\n"
    for i in range(0, tu, 8):
        res += f"__m256 s{i} = _mm256_loadu_ps(&scales[base_output + j + {i}]);\n"
    for i in range(0, tu, 8):
        res += f"__m256 zr{i} = _mm256_fnmadd_ps(z{i}, r, o{i});\n"
    for i in range(0, tu, 8):
        res += f"__m256 o2{i} = _mm256_mul_ps(zr{i}, s{i});\n"
    for i in range(0, tu, 8):
        res += f"_mm256_storeu_ps(&output[i*t + base_output + j + {i}], o2{i});\n"
    res += "}\n"
    res += "}\n"
    res += "}\n"
    res += "}\n"

    # wrapper for qgemm if we call from cpp
    res += f"inline void forward{bits}_cpu(\n"
    res += "torch::Tensor in, torch::Tensor weight, torch::Tensor out,\n"
    res += "torch::Tensor bias, torch::Tensor scales, torch::Tensor zeros, torch::Tensor sums,\n"
    res += "int N, int M, int T, int nb, int mb, int tb, int tt, int cutoff){\n"
    res += "int*   W = weight.data_ptr<int>();\n"
    res += "float* input = in.data_ptr<float>();\n"
    res += "float* b   = bias.data_ptr<float>();\n"
    res += "float* s   = scales.data_ptr<float>();\n"
    # res +=  "int* z   = zeros.data_ptr<int>();\n"
    res += "float* z   = zeros.data_ptr<float>();\n"
    res += "float* r   = sums.data_ptr<float>();\n"
    res += "float* O   = out.data_ptr<float>();\n"
    res += "\n"
    res += f"q{bits}gemm(input, W, s, z, b, r, O, N, M, T, nb, mb, tb, tt, cutoff);\n"
    res += "}\n"
    return res


def unpack_zeros(bits):
    res = ""
    res += f"void unpack_zeros{bits}_cpu(const int* zv, float* ov, int n, int m)"
    packed = 32 // bits
    mask = (2**bits) - 1
    res += "{\nconst __m256i ones = _mm256_set1_epi32(1);\n"
    res += f"const __m256i mask = _mm256_set1_epi32({mask});\n"
    if bits == 4:
        res += "const __m256i shift = _mm256_set_epi32(28,24,20,16,12,8,4,0);\n"
    elif bits == 3:
        pass
    elif bits == 2:
        res += "const __m256i shift0 = _mm256_set_epi32(30,28,26,24,22,20,18,16);\n"
        res += "const __m256i shift1 = _mm256_set_epi32(14,12,10,8,6,4,2,0);\n"
    else:
        print("ERROR")
    res += "for(int i = 0; i < n; i++){\n"
    if bits == 4:
        res += "for(int j = 0; j < m; j+=8){\n"
        res += "__m256i z = _mm256_set1_epi32(zv[i*m/8 + j/8]);\n"
        res += "__m256i z0 = _mm256_srlv_epi32(z, shift);\n"
        res += "__m256i z1 = _mm256_and_si256(z0, mask);\n"
        res += "__m256i z2 = _mm256_add_epi32(z1, ones);\n"
        res += "__m256 z3 = _mm256_cvtepi32_ps(z2);\n"
        res += "_mm256_storeu_ps(&ov[i*m +j], z3);\n"
    elif bits == 2:
        res += f"for (int j = 0; j < m; j+={packed})"
        res += "{\n"
        res += f"for (int k = 0; k < {packed}; k++)"
        res += "{\n"
        res += f"ov[i*m + j+k] = (((zv[j/{packed}] >> ({bits}*k)) & {mask})+1);\n"
        res += "}\n"
        # res += "for(int j = 0; j < m; j+=16){\n"
        # res += "__m256i z = _mm256_set1_epi32(zv[i*m/16 + j/16]);\n"
        # res += "__m256i z00 = _mm256_srlv_epi32(z, shift0);\n"
        # res += "__m256i z01 = _mm256_srlv_epi32(z, shift1);\n"
        # res += "__m256i z10 = _mm256_and_si256(z00, mask);\n"
        # res += "__m256i z11 = _mm256_and_si256(z01, mask);\n"
        # res += "__m256i z20 = _mm256_add_epi32(z10, ones);\n"
        # res += "__m256i z21 = _mm256_add_epi32(z11, ones);\n"
        # res += "__m256 z30 = _mm256_cvtepi32_ps(z20);\n"
        # res += "__m256 z31 = _mm256_cvtepi32_ps(z21);\n"
        # res += "_mm256_storeu_ps(&ov[i*m +j], z30);\n"
        # res += "_mm256_storeu_ps(&ov[i*m +j+8], z31);\n"
    elif bits == 3:
        # pass
        res += "for(int j = 0; j < m; j+=32){\n"
        res += 'std::cout<<"not yet implemented"<<std::endl;\n'
        # res += "unsigned int z0 = zv[i*m+j/32*3];\n"
        # res += "unsigned int z1 = zv[i*m+j/32*3+1];\n"
        # res += "unsigned int z2 = zv[i*m+j/32*3+2];\n"
        # for i in range(10):
        # res += f"unsigned int z0{i} = ((z0 >> {29 - i*3}) & 7) + 1;\n"
        # for i in range(10):
        # res += f"ov[i*m + j + {i}] = z0{i} * sv[i*m + j + {i}];\n"
        # res += "unsigned int t0 = ((z0<<1 & 6) | (z1>>31)) + 1;\n"
        # res += "ov[i*m + j + 10] = t0 * sv[i*m + j + 10];\n"
        # for i in range(10):
        # res += f"unsigned int z1{i} = ((z1 >> {28 - i*3}) & 7) + 1;\n"
        # for i in range(10):
        # res += f"ov[i*m + j + {11 + i}] = z1{i} * sv[i*m + j + {11 + i}];\n"
        # res += "unsigned int t1 = ((z1<<2 & 6) | (z2>>30)) + 1;\n"
        # res += "ov[i*m + j + 21] = t1 * sv[i*m + j + 21];\n"
        # for i in range(10):
        # res += f"unsigned int z2{i} = ((z2 >> {27 - i*3}) & 7) + 1;\n"
        # for i in range(10):
        # res += f"ov[i*m + j + {22 + i}] = z2{i} * sv[i*m + j + {22 + i}];\n"

    res += "}\n"
    res += "}\n"
    res += "}\n"

    # write the pybind interface
    res += f"void unpack_zeros{bits}(torch::Tensor zeros, torch::Tensor out, int N, int M)"
    res += "{\nint* Z = zeros.data_ptr<int>();\n"
    res += "float* O = out.data_ptr<float>();\n"
    res += f"unpack_zeros{bits}_cpu(Z, O, N, M);\n"
    res += "}\n"

    return res


def gen_module(r, p, bits_list=[2, 3, 4]):
    code = ""
    for bits in bits_list:
        if bits == 3:
            unroll = 3
            nu = 1  # args.n
            mu = 32
            tu = 32
        else:
            unroll = 2
            nu = 1  # args.n
            mu = 16
            # mu = 32
            tu = 32

        code += qforward(nu, mu, tu, p, unroll, bits=bits, module=True, gs=False)
        code += qforward(nu, mu, tu, p, unroll, bits=bits, module=True, gs=True)
        code += pack_qw_module(bits)
        code += unpack_zeros(bits)

    with open("./autogptq_extension/qigen/backend.cpp", "w") as f:
        f.write(template.includes())
        f.write(template.quant_scalar())
        f.write(compute_reduction(p))
        f.write(unquantize_sim(p))
        f.write(code)
        f.write(template.module(bits_list))


def compute_reduction(p):
    res = ""
    res += "void compute_reduction_cpu(const float* in, float* out, int n, int m, int gs){\n"
    res += f"#pragma omp parallel num_threads({p})\n"
    res += "{\n"
    res += "#pragma omp for collapse(2)\n"
    res += "for(int i = 0; i < n; i++){\n"
    res += "for(int j0 = 0; j0 < m; j0+=gs){\n"
    res += "__m256 acc = _mm256_setzero_ps();\n"
    res += "for(int j1 = j0; j1 < j0+gs; j1+=8){\n"
    res += "__m256 x = _mm256_loadu_ps(&in[i*m  + j1]);\n"
    res += "acc = _mm256_add_ps(acc, x);\n"
    res += "}\n"
    # compute simd add reduction
    res += "const __m128 hiQuad = _mm256_extractf128_ps(acc, 1);\n"
    res += "const __m128 loQuad = _mm256_castps256_ps128(acc);\n"
    res += "const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);\n"
    res += "const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);\n"
    res += "const __m128 sumDual = _mm_add_ps(sumQuad, hiDual);\n"
    res += "const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);\n"
    res += "const __m128 sum = _mm_add_ss(hi, sumDual);\n"
    res += "out[(i*m + j0)/gs] = _mm_cvtss_f32(sum);\n"
    res += "}\n"
    res += "}\n"
    res += "}\n"
    res += "}\n"

    # write the pybind interface
    res += f"void compute_reduction(torch::Tensor in, torch::Tensor out, int N, int M, int gs)"
    res += "{\nfloat* I = in.data_ptr<float>();\n"
    res += "float* O = out.data_ptr<float>();\n"
    res += f"compute_reduction_cpu(I, O, N, M, gs);\n"
    res += "}\n"

    return res


def unquantize_sim(p):
    res = ""
    res += (
        "void unquantize_sim_cpu(const int* in, float* out, float* s, float* z, int n, int m, int bits, int gs){\n"
    )
    res += f"#pragma omp parallel num_threads({p})\n"
    res += "{\n"
    res += "int packed = 32/bits;\n"
    res += "int mask = (1<<bits) - 1;\n"
    res += "#pragma omp for\n"
    res += "for(int i0 = 0; i0 < n; i0+=gs){\n"
    res += "int row = i0 / gs;\n"
    res += "for(int i1 = i0; i1 < i0+gs; i1+=packed){\n"
    res += "for(int j0 = 0; j0 < m; j0++){\n"
    res += "for(int k = 0; k < packed; k++){\n"
    res += "out[(i1+k)*m + j0] = ((float)((in[i1*m/packed + j0] >> (bits*k)) & mask) - z[(row)*m + j0]) * s[(row)*m + j0];\n"
    res += "}\n"
    res += "}\n"
    res += "}\n"
    res += "}\n"
    res += "}\n"
    res += "}\n"

    # write the pybind interface
    res += f"void unquantize_sim(torch::Tensor in, torch::Tensor out, torch::Tensor s, torch::Tensor z, int N, int M, int bits, int gs)"
    res += "{\nint* I = in.data_ptr<int>();\n"
    res += "float* O = out.data_ptr<float>();\n"
    res += "float* S = s.data_ptr<float>();\n"
    res += "float* Z = z.data_ptr<float>();\n"
    res += f"unquantize_sim_cpu(I, O, S, Z, N, M, bits, gs);\n"
    res += "}\n"

    return res


def pack_qw_module(bits):
    packed = 32 // bits
    res = ""
    if bits == 3:
        res += f"inline void pack{bits}_qw_inner(int* A, int* B, const int N, const int M, const int nb, const int mb, int cutoff)"
        res += "{\n"
        res += "// copy the full matrix A in blocked format into B\n"
        res += "uint64_t idx = 0;\n"
        # res += f"  const {int(tb)};\n"
        res += "for(int j = 0, tid = 0; j < M; j+=mb, tid++){\n"
        res += "for(int i = 0; i < N; i+=nb){\n \
for(int ii = i; ii < mymin(i+nb, N); ii+=3){\n \
for(int jj = j; jj < mymin(j+mb, M); jj+=8){\n \
for(int iii = ii; iii < ii + 3; iii++){\n \
for(int jjj = jj; jjj < jj + 8; jjj++){\n \
B[idx] = A[iii*M+jjj];\n \
idx++;\n \
}\n \
}\n \
}\n \
}\n \
}\n \
}\n \
}\n"
        res += f"inline void pack{bits}_w_cpu(\n"
        res += "torch::Tensor in, torch::Tensor out,\n"
        res += "int N, int M, int nb, int mb, int cutoff){\n"
        res += "int* input = in.data_ptr<int>();\n"
        res += "int* O = out.data_ptr<int>();\n"
        res += f"pack{bits}_qw_inner(input, O, N, M, nb, mb, cutoff);\n"
        res += "}\n"
        return res
    else:
        # in case i do this for python i can just add the n,m,nb,mb as function parameters
        res += f"inline void pack{bits}_qw_inner(int* A, int* B, const int N, const int M, const int nb, int mb, int cutoff)"
        res += "{\n"
        res += "// copy the full matrix A in blocked format into B\n"
        res += "uint64_t idx = 0;\n"
        res += "for(int j = 0, tid = 0; j < M; j+=mb, tid++){\n"
        res += "for(int i = 0; i < N; i+=nb){\n \
for(int ii = i; ii < mymin(i+nb, N); ii++){\n \
for(int jj = j; jj < mymin(j+mb, M); jj++){\n \
B[idx] = A[ii*M+jj];\n \
idx++;\n \
}\n \
}\n \
}\n"
        res += "}\n"
        res += "}\n"
        res += f"inline void pack{bits}_w_cpu(\n"
        res += "torch::Tensor in, torch::Tensor out,\n"
        res += "int N, int M, int nb, int mb, int cutoff){\n"
        res += "int* input = in.data_ptr<int>();\n"
        res += "int* O = out.data_ptr<int>();\n"
        res += f"  pack{bits}_qw_inner(input, O, N, M, nb, mb, cutoff);\n"
        res += "}\n"
        return res


def gen_module_search(r, p, bits_list=[2, 3, 4]):
    # print measurements to a tmp file and read back best micro parameters
    code = ""

    subprocess.call(["rm", "./autogptq_extension/qigen/tmp.csv"])
    subprocess.call(["touch", "./autogptq_extension/qigen/tmp.csv"])
    with open("./autogptq_extension/qigen/tmp.csv", "w") as f:
        f.write("bits,nu,mu,tu,unroll,p,gs,time\n")

    n, m, t, nb, mb, tb = 1, 4096, 4096, 1, 1024, 32

    for mu in [16]:
        for tu in [16, 32, 64]:
            if tb % tu == 0:
                for gs in [-1, 64]:
                    for bits in [4, 3, 2]:
                        if bits == 3:
                            for u in [5]:
                                print(n, m, t, n, mb, tb, 1, mu, tu, p, u, bits, gs, end="\r", flush=True)
                                gen_and_compile(n, m, t, n, mb, tb, 1, mu, tu, p, u, bits=bits, gs=gs, module=True)
                        else:
                            for u in [1, 2, 4, 8]:
                                print(n, m, t, n, mb, tb, 1, mu, tu, p, u, bits, gs, end="\r", flush=True)
                                gen_and_compile(n, m, t, n, mb, tb, 1, mu, tu, p, u, bits=bits, gs=gs, module=True)

    df = pd.read_csv("./autogptq_extension/qigen/tmp.csv")

    for bits in bits_list:
        bits_df = df[df["bits"] == bits]

        bits_nogs = bits_df[bits_df["gs"] == -1]
        best = bits_nogs[bits_nogs["time"] == bits_nogs["time"].min()]
        nu = int(best["nu"].values[0])
        mu = int(best["mu"].values[0])
        tu = int(best["tu"].values[0])
        unroll = int(best["unroll"].values[0])

        code += qforward(nu, mu, tu, p, unroll, bits=bits, module=True, gs=False)

        bits_gs = bits_df[bits_df["gs"] != -1]
        best = bits_gs[bits_gs["time"] == bits_gs["time"].min()]
        nu_gs = int(best["nu"].values[0])
        mu_gs = int(best["mu"].values[0])
        tu_gs = int(best["tu"].values[0])
        unroll_gs = int(best["unroll"].values[0])
        code += qforward(nu_gs, mu_gs, tu_gs, p, unroll_gs, bits=bits, module=True, gs=True)

        code += pack_qw_module(bits)
        code += unpack_zeros(bits)

    with open("./autogptq_extension/qigen/backend.cpp", "w") as f:
        f.write(template.includes())
        f.write(template.quant_scalar())
        f.write(compute_reduction(p))
        f.write(unquantize_sim(p))
        f.write(code)
        f.write(template.module(bits_list))

    # subprocess.call(["rm", "./autogptq_extension/qigen/tmp.csv"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--t", type=int, default=1024)
    parser.add_argument("--nb", type=int, default=128)
    parser.add_argument("--mb", type=int, default=128)
    parser.add_argument("--tb", type=int, default=128)
    parser.add_argument("--mu", type=int, default=4)
    parser.add_argument("--nu", type=int, default=4)
    parser.add_argument("--tu", type=int, default=8)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--module", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--p", type=int, default=8)
    parser.add_argument("--gs", type=int, default=-1)
    args = parser.parse_args()
    if args.module and args.search:
        gen_module_search(args.r, args.p, [2, 3, 4])
    if args.module and not args.search:
        gen_module(args.r, args.p, [2, 3, 4])
    if args.search and not args.module:
        grid()
    if args.model:
        gen_model(args.n, args.m, args.t, args.bits, args.p, args.gs)
