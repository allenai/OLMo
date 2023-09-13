
def load_int(to, address, const=True):
    if const:
        return f"const __m256i {to} = _mm256_loadu_si256({address});"
    else:
        return f"__m256i {to} = _mm256_loadu_si256({address});"

def load_fp(to, address, const=True):
    if const:
        return f"const __m256 {to} = _mm256_loadu_ps({address});"
    else:
        return f"__m256 {to} = _mm256_loadu_ps({address});"

# to = a * b + c
def vfma(to, a, b, c):
    return f"__m256 {to} = _mm256_fmadd_ps({a}, {b}, {c});"

def vsrli(to, a, b):
    return f"const __m256i {to} = _mm256_srli_epi32({a}, {b});"

def vand(to, a, b):
    return f"const __m256i {to} = _mm256_and_si256({a}, {b});"

def vbroadcast_fp(to, a):
    return f"const __m256 {to} = _mm256_set1_ps({a});"

def vbroadcast_int32(to, a):
    return f"__m256i {to} = _mm256_set1_epi32({a});"

def vsetzero(to):
    return f"__m256 {to} = _mm256_setzero_ps();"

def vcvtepi32_ps(to, a):
    return f"const __m256 {to} = _mm256_cvtepi32_ps({a});"

def _256extractf128_ps(to, a, imm):
    return f"const __m128 {to} = _mm256_extractf128_ps({a}, {imm});"

def _256castps256_ps128(to, a):
    return f"const __m128 {to} = _mm256_castps256_ps128({a});"

def _add_ps(to, a, b):
    return f"const __m128 {to} = _mm_add_ps({a}, {b});"

def _movehl_ps(to, a, b):
    return f"const __m128 {to} = _mm_movehl_ps({a}, {b});"

def _shuffle_ps(to, a, b, imm):
    return f"const __m128 {to} = _mm_shuffle_ps({a}, {b}, {imm});"

def _cvtss_f32(to, a):
    return f"const float {to} = _mm_cvtss_f32({a});"

def _reduce8_acc(a, b, c, d, e, f, g, h):
    res = ""
    res += _256extractf128_ps("hi_quad0", a, 1)
    res += _256extractf128_ps("hi_quad1", b, 1)
    res += _256extractf128_ps("hi_quad2", c, 1)
    res += _256extractf128_ps("hi_quad3", d, 1)
    res += _256extractf128_ps("hi_quad4", e, 1)
    res += _256extractf128_ps("hi_quad5", f, 1)
    res += _256extractf128_ps("hi_quad6", g, 1)
    res += _256extractf128_ps("hi_quad7", h, 1)

    res += _256castps256_ps128("lo_quad0", a)
    res += _256castps256_ps128("lo_quad1", b)
    res += _256castps256_ps128("lo_quad2", c)
    res += _256castps256_ps128("lo_quad3", d)
    res += _256castps256_ps128("lo_quad4", e)
    res += _256castps256_ps128("lo_quad5", f)
    res += _256castps256_ps128("lo_quad6", g)
    res += _256castps256_ps128("lo_quad7", h)

    res += _add_ps("sum_quad0", "lo_quad0", "hi_quad0")
    res += _add_ps("sum_quad1", "lo_quad1", "hi_quad1")
    res += _add_ps("sum_quad2", "lo_quad2", "hi_quad2")
    res += _add_ps("sum_quad3", "lo_quad3", "hi_quad3")
    res += _add_ps("sum_quad4", "lo_quad4", "hi_quad4")
    res += _add_ps("sum_quad5", "lo_quad5", "hi_quad5")
    res += _add_ps("sum_quad6", "lo_quad6", "hi_quad6")
    res += _add_ps("sum_quad7", "lo_quad7", "hi_quad7")

    res += _movehl_ps("hi_dual0", "sum_quad0", "sum_quad0")
    res += _movehl_ps("hi_dual1", "sum_quad1", "sum_quad1")
    res += _movehl_ps("hi_dual2", "sum_quad2", "sum_quad2")
    res += _movehl_ps("hi_dual3", "sum_quad3", "sum_quad3")
    res += _movehl_ps("hi_dual4", "sum_quad4", "sum_quad4")
    res += _movehl_ps("hi_dual5", "sum_quad5", "sum_quad5")
    res += _movehl_ps("hi_dual6", "sum_quad6", "sum_quad6")
    res += _movehl_ps("hi_dual7", "sum_quad7", "sum_quad7")

    res += _add_ps("sum_dual0", "sum_quad0", "hi_dual0")
    res += _add_ps("sum_dual1", "sum_quad1", "hi_dual1")
    res += _add_ps("sum_dual2", "sum_quad2", "hi_dual2")
    res += _add_ps("sum_dual3", "sum_quad3", "hi_dual3")
    res += _add_ps("sum_dual4", "sum_quad4", "hi_dual4")
    res += _add_ps("sum_dual5", "sum_quad5", "hi_dual5")
    res += _add_ps("sum_dual6", "sum_quad6", "hi_dual6")
    res += _add_ps("sum_dual7", "sum_quad7", "hi_dual7")

    res += _shuffle_ps("hi0", "sum_dual0", "sum_dual0", 0x1)
    res += _shuffle_ps("hi1", "sum_dual1", "sum_dual1", 0x1)
    res += _shuffle_ps("hi2", "sum_dual2", "sum_dual2", 0x1)
    res += _shuffle_ps("hi3", "sum_dual3", "sum_dual3", 0x1)
    res += _shuffle_ps("hi4", "sum_dual4", "sum_dual4", 0x1)
    res += _shuffle_ps("hi5", "sum_dual5", "sum_dual5", 0x1)
    res += _shuffle_ps("hi6", "sum_dual6", "sum_dual6", 0x1)
    res += _shuffle_ps("hi7", "sum_dual7", "sum_dual7", 0x1)
    
    res += _add_ps("sum0", "sum_dual0", "hi0")
    res += _add_ps("sum1", "sum_dual1", "hi1")
    res += _add_ps("sum2", "sum_dual2", "hi2")
    res += _add_ps("sum3", "sum_dual3", "hi3")
    res += _add_ps("sum4", "sum_dual4", "hi4")
    res += _add_ps("sum5", "sum_dual5", "hi5")
    res += _add_ps("sum6", "sum_dual6", "hi6")
    res += _add_ps("sum7", "sum_dual7", "hi7")

    res += _cvtss_f32(f"f{a}", "sum0")
    res += _cvtss_f32(f"f{b}", "sum1")
    res += _cvtss_f32(f"f{c}", "sum2")
    res += _cvtss_f32(f"f{d}", "sum3")
    res += _cvtss_f32(f"f{e}", "sum4")
    res += _cvtss_f32(f"f{f}", "sum5")
    res += _cvtss_f32(f"f{g}", "sum6")
    res += _cvtss_f32(f"f{h}", "sum7")

    return res

acc_idx = 0
def _reduce_add(a):
    global acc_idx
    res = ""
    res += _256extractf128_ps(f"hi_quad{acc_idx}", a, 1)
    res += _256castps256_ps128(f"lo_quad{acc_idx}", a)
    res += _add_ps(f"sum_quad{acc_idx}", f"lo_quad{acc_idx}", f"hi_quad{acc_idx}")
    res += _movehl_ps(f"hi_dual{acc_idx}", f"sum_quad{acc_idx}", f"sum_quad{acc_idx}")
    res += _add_ps(f"sum_dual{acc_idx}", f"sum_quad{acc_idx}", f"hi_dual{acc_idx}")
    res += _shuffle_ps(f"hi{acc_idx}", f"sum_dual{acc_idx}", f"sum_dual{acc_idx}", 0x1)
    res += _add_ps(f"sum{acc_idx}", f"sum_dual{acc_idx}", f"hi{acc_idx}")
    res += _cvtss_f32(f"f{a}", f"sum{acc_idx}")
    acc_idx += 1
    return res






