#!/usr/bin/env python3
"""Fixed-parameter certificate for the BMRaccoon tables.

This verifier takes the parameter set, rederives every width, 
rounding parameter and auxiliary modulus from them,
checks the conditions of the security proofs, and runs the bundled
lattice-estimator on the associated MSIS/MLWE instances.  It also reports the
sizes, the communication cost and the ZK-LaBRADOR constraint counts.  It is
not a parameter search.
"""

import functools
import math
import sys
import unicodedata
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from sage.all import binomial, is_prime, next_prime

SCRIPT_DIR = Path(__file__).resolve().parent
ESTIMATOR_DIR = SCRIPT_DIR / "lattice-estimator"
LN2 = math.log(2.0)

LAM = 128
N = 256
LOG2_QH = 64

BLIND_BITS = 40
STAT_BITS = LAM
EPS_SMOOTH = 2.0**-400

# Bits per SampleInBall index of the challenge sampler.
KAPPA_IDX = 160
LABRADOR_PROOF_KB = 125.0
KECCAK_CONSTRAINTS = 24 * 1600
SHAKE256_RATE_BITS = 1088
SHA3_384_RATE_BITS = 832
# FIPS 202 domain suffix (1111 for SHAKE, 01 for SHA3) plus the two pad10*1 bits.
SHAKE_PADDING_BITS = 6
SHA3_PADDING_BITS = 4
# SHA3-384
PREHASH_BITS = 384
PREHASH_COUNT = 4
# The message m is one ring element committed next to r' in C.
ELL_MSG = 1


@dataclass(frozen=True)
class ParameterSet:
    name: str
    log2_qs: int
    n: int
    k: int
    m: int
    beta_c: int
    beta_lwe: int
    sigma_sk: float
    sigma_w: float
    sigma_wp: float
    nu_t: int
    nu_w: int
    q: int
    k_he: int
    p_he: int
    Q_he: int
    sigma_fl: float
    k_ext: int
    n_r: int
    p_ext: int
    Q_ext: int

    @property
    def qs(self):
        return 2.0**self.log2_qs

    @property
    def ell(self):
        return ell()


@dataclass(frozen=True)
class SearchResult:
    log2_qs: int
    n: int
    k: int
    m: int
    q: int
    k_he: int
    k_ext: int
    n_r: int




SEARCH_RESULTS = (
    SearchResult(log2_qs=32, n=32, k=21, m=41, q=1523 * 2**103 + 49,
                 k_he=28, k_ext=24, n_r=89),
    SearchResult(log2_qs=32, n=64, k=21, m=41, q=3045 * 2**103 + 47,
                 k_he=28, k_ext=24, n_r=89),
    SearchResult(log2_qs=64, n=32, k=24, m=47, q=2775 * 2**119 + 187,
                 k_he=37, k_ext=27, n_r=101),
    SearchResult(log2_qs=64, n=64, k=24, m=47, q=1387 * 2**121 + 29,
                 k_he=37, k_ext=27, n_r=101),
)


def log2_sum(*exps):
    finite = [e for e in exps if e != -math.inf]
    if not finite:
        return -math.inf
    top = max(finite)
    return top + math.log2(sum(2.0 ** (e - top) for e in finite))


def log2_binom(a, b):
    return math.log2(int(binomial(a, b)))


# ℓ2 tail e^(1/4)·σ·√dim of Lemma 5; it fails with probability below 2^(−dim/10).
def gaussian_l2_bound(sigma, dim):
    return math.exp(0.25) * sigma * math.sqrt(dim)


def smoothing_bound(dim, eps=EPS_SMOOTH):
    return math.sqrt((math.log(2 * dim) + math.log1p(1 / eps)) / math.pi)


# Coefficient tail with a union bound at 2^−λ over the Nm coefficients.
def gaussian_linf_bound(sigma, m):
    return sigma * math.sqrt((math.log(2 * N * m) + LAM * LN2) / math.pi)


# β_lwe ≥ m log N (AOM-MISIS Sect. 6.2), used directly as the ℓ∞ box of the MLWE secret.
def mlwe_box_target(m):
    return math.ceil(m * math.log(N))


# Middle term of σ_w, with exponent 2/(N(m−k)); it never binds.
def nonce_width_entropy_floor(log2_q, k, m):
    return math.sqrt(2 * math.pi) * 2 * N * 2.0 ** (log2_q * (1.0 / m + 2.0 / (N * (m - k))))


def ell(lam=LAM, n_ring=N):
    return math.ceil(2 * lam / math.log2(n_ring))


def log2_challenge_space(beta_c):
    return log2_binom(N, beta_c) + beta_c


# Pr[BadHash] ≤ 4Q^2/|S_b|^ℓ + 4Q^2/|S_c| with Q = q_h + q_s + 1 oracle queries.
def log2_badhash(log2_qs, beta_c):
    two_q_sq = 2 * (1 + math.log2(2.0**LOG2_QH + 2.0**log2_qs + 1))
    return log2_sum(two_q_sq - ell() * math.log2(2 * N), two_q_sq - log2_challenge_space(beta_c))


# Lemma 2: q = 2^ν·q_ν + q_⊥ with q_⊥ ≤ 2^(ν−1) − 1.
def is_rounding_admissible(q, nu):
    if nu < 1 or nu >= q.bit_length() - 1:
        return False
    return q % (1 << nu) <= (1 << (nu - 1)) - 1


# q_ν = ⌊q/2^ν⌉, which equals ⌊q/2^ν⌋ under Lemma 2.
def rounded_modulus(q, nu):
    return (q + (1 << (nu - 1))) >> nu



def log2_sampleinball_abort(beta_c):
    return math.log2(sum(range(N - beta_c + 1, N + 1))) - KAPPA_IDX


def signature_bounds(p):
    """Norm bounds of the signature scheme.

    The Lemma 5 ℓ2 tails of s_i, the nonces and r' give β_z', the bound the
    verifier checks.  From β_z' follow the MSIS bound of the Thm 3 reduction and
    the AOM-MISIS bounds β_s, β_b, β_u, β_sis of ZT25 Thm 1.  Also returns β_r,
    the shift that r' must hide, and the blindness distance ε_stat.
    """
    n, k, m, l, bc = p.n, p.k, p.m, p.ell, p.beta_c
    Nm, Nk = N * m, N * k
    out = {}
    out["beta_sk"] = gaussian_l2_bound(p.sigma_sk, Nm)
    out["beta_w"] = gaussian_l2_bound(p.sigma_w, Nm)
    out["beta_wp"] = gaussian_l2_bound(p.sigma_wp, Nm)
    bsk, bw, bwp = out["beta_sk"], out["beta_w"], out["beta_wp"]
    # β_z' of MRaccoon* (Thm 4); a BMRaccoon signature satisfies the same bound.
    rounding = ((3 * n - 1) * bc * 2.0 ** (p.nu_t - 1) + 4 * 2.0 ** (p.nu_w - 1)) * math.sqrt(Nk)
    out["beta_zp"] = math.sqrt(2) * (n * bc * bsk + n * (1 + l) * bw + bwp) + rounding
    out["beta_bms"] = out["beta_zp"]
    bz = out["beta_zp"]
    # MSIS bound of the Thm 3 reduction.
    out["beta_msis"] = 4 * bz + 2.0 ** (p.nu_w + 2) * math.sqrt(Nk)
    out["log2_signing_queries"] = math.log2(1 + p.qs * (1 + l))
    # AOM-MISIS bounds of ZT25 Thm 1 with β_z replaced by β'_z (Remark 1).
    out["beta_s"] = (4 * bz + (2.0 ** (p.nu_w + 1) + (3 * n - 1) * bc * 2.0**p.nu_t) * math.sqrt(Nk)
                     + 2 * math.sqrt(2) * (n - 1) * bc * bsk)
    out["beta_b"] = 2 * p.sigma_sk * bc
    out["beta_u"] = 2 / p.sigma_sk + bc * math.sqrt(8 * N * p.qs) / p.sigma_w
    out["beta_sis"] = out["beta_s"] + math.sqrt(Nm) * out["beta_b"]
    out["beta_lwe"] = p.beta_lwe
    out["beta_r"] = n * bw * (1 + l)
    # Per-message blindness distance of the user's shift r' (Thm 8).
    out["eps_stat"] = 2 * math.sqrt(math.pi) * out["beta_r"] / p.sigma_wp + math.sqrt(6 * EPS_SMOOTH)
    return out


def encryption_bounds(p, sb):
    """ℓ∞ bounds of the BGV encryption of the signer's response z_i.

    The plaintext is bounded by B_plain = β_c·B_sk + (1+ℓ)·B_w.  After HE.Eval
    (honest multipliers, or any witness that L_i,2 accepts) and the flooding
    noise, the ciphertext noise must stay below the no-wrap bound mod Q_he.
    Also returns the Rényi bound R_2 for circuit privacy over the q_s(m−k)
    evaluations, and the failure probability of the flooding-coin ℓ2 bound.
    """
    l, d = p.ell, p.k_he
    E = p.qs * (p.m - p.k)
    out = {"coin_dim": (2 * d + 1) * N, "E": E}
    out["B_sk_inf"] = gaussian_linf_bound(p.sigma_sk, p.m)
    out["B_w_inf"] = gaussian_linf_bound(p.sigma_w, p.m)
    out["B_plain"] = p.beta_c * out["B_sk_inf"] + (1 + l) * out["B_w_inf"]
    # Fresh noise with binary key and coins (BGV note, Lemma 1).
    out["B_enc"] = 2 * N * d + 1
    # ℓ1 bound on the multipliers s_i, r_i^(j) of one coordinate for an honest signer.
    out["B_alpha1"] = math.exp(0.25) * N * (p.sigma_sk + l * p.sigma_w)
    out["B_eval_honest"] = out["B_alpha1"] * out["B_enc"]
    # Decryption must also succeed for every witness L_i,2 accepts, which only has the
    # coefficient range checks (BGV note Sect. 6).
    out["B_eval_mal"] = N * (out["B_sk_inf"] + l * out["B_w_inf"]) * out["B_enc"]
    out["B_eval"] = max(out["B_eval_honest"], out["B_eval_mal"])
    # ℓ2 shift that evaluation adds to the flooding coins.
    out["B_shift"] = out["B_alpha1"] * math.sqrt(N * (2 * d + 1))
    eps_fac = 2 * math.log2((1 + EPS_SMOOTH) / (1 - EPS_SMOOTH))
    out["cp_log2_R2"] = eps_fac + 2 * math.pi * E * out["B_shift"] ** 2 / p.sigma_fl**2 / LN2
    out["eps_cp"] = math.sqrt(math.pi) * out["B_shift"] / p.sigma_fl + math.sqrt(6 * EPS_SMOOTH)
    # Flooding noise: Gaussian tail for honest coins, Cauchy–Schwarz on the ℓ2 coin bound otherwise.
    out["B_fl_honest"] = (p.sigma_fl * math.sqrt(2 * N * d + 1)
                          * math.sqrt((math.log(2 * N * (p.m - p.k) * E) + LAM * LN2) / math.pi))
    out["B_theta_fl_l2"] = gaussian_l2_bound(p.sigma_fl, N * (2 * d + 1))
    out["B_fl_mal"] = math.sqrt(2 * N * d + 1) * out["B_theta_fl_l2"]
    out["B_fl"] = max(out["B_fl_honest"], out["B_fl_mal"])
    out["B_final"] = out["B_eval"] + out["B_fl"]
    # Lemma 5 failure of the flooding-coin ℓ2 bound over the m−k encryptions.
    out["log2_eps_he"] = math.log2(p.m - p.k) - N * (2 * d + 1) / 10.0
    return out


def encryption_modulus_min(he, p_he):
    return 2 * (he["B_plain"] + p_he * he["B_final"]) + 1


def commitment_bounds(p, sb):
    """Bounds of the BDLOP commitment C to (r', m).

    B_msg bounds the committed coefficients and B_Egamma the noise E·γ of a
    binary opening; together they give the no-wrap condition for extraction
    mod Q_ext.  B_bind is the ℓ2 norm of the difference of two openings, used
    for statistical binding.
    """
    out = {"ell_ext": p.m + ELL_MSG}
    out["B_gamma"] = 1.0
    # Binary E and γ: each coefficient of E·γ is at most N·n_r (BDLOP note, Lemma 1).
    out["B_Egamma"] = N * p.n_r
    # Coefficient bound of the committed (r', m); m is binary, so r' dominates.
    out["B_msg"] = max(1.0, gaussian_linf_bound(p.sigma_wp, p.m))
    out["B_m0"] = out["B_msg"]
    # ℓ2 norm of the difference of two binary openings.
    out["B_bind"] = math.sqrt(N * p.n_r)
    return out


def commitment_modulus_min(cs, p_ext):
    return 2 * (cs["B_msg"] + p_ext * cs["B_Egamma"]) + 1


# Widths are rounded up to the next 2^(j/100).
def round_width_up(x):
    return 2.0 ** (math.ceil(100 * math.log2(x)) / 100.0)


# Q_he and Q_ext must be coprime to the plaintext moduli p_he and p_ext.
def next_prime_above(x, avoid=None):
    candidate = int(next_prime(math.ceil(x)))
    while candidate == avoid:
        candidate = int(next_prime(candidate))
    return candidate


# Smallest β_c with |S_c| ≥ 2^(2λ) and Pr[BadHash] ≤ 2^−128.
def choose_challenge_weight(log2_qs):
    for beta_c in range(1, N + 1):
        if log2_challenge_space(beta_c) >= 2 * LAM and log2_badhash(log2_qs, beta_c) <= -STAT_BITS:
            return beta_c
    raise ValueError("no admissible challenge weight")



def key_width(beta_lwe, m):
    return round_width_up(max(4 * beta_lwe * math.sqrt(m * N), math.sqrt(math.log(6 * m * N) / math.pi)))


def nonce_width(log2_qs, m, beta_c, sigma_sk, log2_q, k):
    return round_width_up(max(sigma_sk * beta_c * math.sqrt(8 * N * 2.0**log2_qs) / 2,
                              nonce_width_entropy_floor(log2_q, k, m),
                              math.sqrt((math.log(2 * N * m) + LAM) / math.pi)))

def blinding_width(n, m, sigma_w):
    beta_r = n * gaussian_l2_bound(sigma_w, N * m) * (1 + ell())
    return round_width_up(max(4 * math.sqrt(math.pi) * beta_r / 2.0**-BLIND_BITS,
                              math.sqrt((math.log(2 * N * m) + LAM) / math.pi)))


# Keeps the rounding terms of β_z' within a constant factor of σ_w'·√(Nk).
def rounding_bits(sigma_wp, n, beta_c):
    return int(math.floor(math.log2(sigma_wp / (n * beta_c)))), int(math.floor(math.log2(sigma_wp)))


# Smallest σ_fl with R_2 ≤ e over all q_s(m−k) evaluations.
def flooding_width(p):
    shift = math.exp(0.25) * N * (p.sigma_sk + p.ell * p.sigma_w) * math.sqrt(N * (2 * p.k_he + 1))
    return round_width_up(shift * math.sqrt(2 * math.pi * p.qs * (p.m - p.k)))


def derive_parameter_set(r):
    """Rebuild a full parameter set from one search result.

    The search fixes (q_s, n, k, m, q, k_he, k_ext, n_r).  The rest follows in
    order: β_c, β_lwe, σ_sk, σ_w, σ_w', the rounding bits ν_t and ν_w, the
    flooding width σ_fl, and last the odd plaintext moduli p_he, p_ext and the
    primes Q_he, Q_ext just above their no-wrap bounds.
    """
    beta_c = choose_challenge_weight(r.log2_qs)
    beta_lwe = mlwe_box_target(r.m)
    sigma_sk = key_width(beta_lwe, r.m)
    sigma_w = nonce_width(r.log2_qs, r.m, beta_c, sigma_sk, math.log2(r.q), r.k)
    sigma_wp = blinding_width(r.n, r.m, sigma_w)
    nu_t, nu_w = rounding_bits(sigma_wp, r.n, beta_c)
    p = ParameterSet("q_s=2^{}, n={}".format(r.log2_qs, r.n), log2_qs=r.log2_qs, n=r.n, k=r.k, m=r.m,
                     beta_c=beta_c, beta_lwe=beta_lwe, sigma_sk=sigma_sk, sigma_w=sigma_w, sigma_wp=sigma_wp,
                     nu_t=nu_t, nu_w=nu_w, q=r.q, k_he=r.k_he, p_he=0, Q_he=0, sigma_fl=1.0,
                     k_ext=r.k_ext, n_r=r.n_r, p_ext=0, Q_ext=0)
    # σ_fl depends on k_he and the widths, so it is filled in after the first pass.
    p = replace(p, sigma_fl=flooding_width(p))
    sb = signature_bounds(p)
    he = encryption_bounds(p, sb)
    # Odd plaintext moduli; the ciphertext moduli are the next primes above the no-wrap bounds.
    p_he = 2 * math.ceil(he["B_plain"]) + 1
    Q_he = next_prime_above(encryption_modulus_min(he, p_he), avoid=p_he)
    cs = commitment_bounds(p, sb)
    p_ext = 2 * math.ceil(cs["B_msg"]) + 1
    Q_ext = next_prime_above(commitment_modulus_min(cs, p_ext), avoid=p_ext)
    return replace(p, p_he=p_he, Q_he=Q_he, p_ext=p_ext, Q_ext=Q_ext)


# Gaussian heuristic for λ_1 of the BDLOP kernel lattice (dimension N·n_r, determinant Q^(N·k_ext)).
def log2_commitment_lambda1(k_ext, n_r, log2_Q):
    return 0.5 * math.log2(n_r * N / (2 * math.pi * math.e)) + log2_Q * k_ext / n_r


def statistical_terms(p, sb, he):
    """log2 of the statistical terms of the security proofs and of correctness.

    BadHash, nonce-commitment guessing, nonce collisions and prehash
    collisions must each be at most 2^−128.  Blindness (ε_stat), circuit
    privacy and correctness have their own targets and are checked in evaluate.
    """
    t = {}
    t["badhash"] = log2_badhash(p.log2_qs, p.beta_c)
    # Min-entropy of a nonce commitment against q_s(q_h + q_s) guesses.
    gamma = p.k * N * math.log2(p.sigma_w)
    t["commitment_guess"] = p.log2_qs + math.log2(2.0**LOG2_QH + p.qs) - gamma
    # Q·2^(−2κ+2) term of ZT25 Thm 1.
    t["nonce_collision"] = sb["log2_signing_queries"] - 2 * LAM + 2
    t["eps_stat"] = math.log2(sb["eps_stat"])
    t["cp_log2_R2"] = he["cp_log2_R2"]
    t["cp_eps"] = math.log2(he["eps_cp"])
    t["prehash_collision"] = (math.log2(PREHASH_COUNT) + 2 * math.log2(2.0**LOG2_QH + p.qs + 1)
                              - PREHASH_BITS)
    # Correctness: Lemma 5 failures of the honest vectors, HE decryption, SampleInBall aborts.
    t["correctness"] = log2_sum(math.log2(p.n * (p.ell + 2) + 1) - N * p.m / 10.0,
                                     math.log2(p.n) + he["log2_eps_he"],
                                     log2_sampleinball_abort(p.beta_c))
    return t


# Expected Golomb–Rice code length of a rounded Gaussian, minimized over the parameter b.
@functools.lru_cache(maxsize=None)
def _golomb_rice_bits_log(log2_std):
    std = 2.0**log2_std
    best = math.inf
    b0 = max(0, int(math.floor(math.log2(max(std, 1.0)))))
    for b in range(max(0, b0 - 2), b0 + 3):
        step, tail, j = 2.0**b, 0.0, 1
        while True:
            pj = math.erfc(j * step / (std * math.sqrt(2)))
            tail += pj
            if pj < 1e-12 or j > 10**5:
                break
            j += 1
        best = min(best, 1 + b + tail + 1)
    return best


def golomb_rice_bits(std):
    return _golomb_rice_bits_log(round(math.log2(std), 3))


def sizes(p, cs):
    """Sizes of keys, signatures, protocol messages and communication.

    vk_i is stored rounded mod q_{ν_t}.  In the signature, c is an index into
    the challenge set, and z and h are Golomb–Rice coded with the standard
    deviation of their coefficients.  Nonce commitments, ciphertexts, ek and C
    are sent uncompressed, and each ZK-LaBRADOR proof counts as 125 KB.
    Every *_bits entry is also returned as *_kb.
    """
    n, k, m, l = p.n, p.k, p.m, p.ell
    out = {}
    log_q = math.ceil(math.log2(p.q))
    out["vk_bits"] = k * N * math.ceil(math.log2(rounded_modulus(p.q, p.nu_t)))
    # z and h are Golomb–Rice coded; z sums r', the n(1+ℓ) nonces and n·β_c key terms.
    s_z = math.sqrt(p.sigma_wp**2 + n * (1 + l) * p.sigma_w**2 + n * p.beta_c * p.sigma_sk**2)
    std_z = s_z / math.sqrt(2 * math.pi)
    std_h = math.sqrt((std_z**2 + n * p.beta_c * 4.0**p.nu_t / 12.0) / 4.0**p.nu_w + 1.0 / 6.0)
    out["c_bits"] = math.ceil(log2_challenge_space(p.beta_c))
    out["z_bits"] = (m - k) * N * golomb_rice_bits(std_z)
    out["h_bits"] = k * N * golomb_rice_bits(std_h)
    out["sig_bits"] = out["c_bits"] + out["z_bits"] + out["h_bits"]
    # Nonce commitments w_i^(j) are sent unrounded.
    out["com_bits"] = (l + 1) * k * N * log_q
    log_Qhe = math.ceil(math.log2(p.Q_he))
    out["ctx_bits"] = (p.k_he + 1) * N * log_Qhe
    out["ek_bits"] = p.k_he * N * log_Qhe + 256
    out["C_bits"] = (p.k_ext + cs["ell_ext"]) * N * math.ceil(math.log2(p.Q_ext))
    proof = LABRADOR_PROOF_KB * 8000
    # U sends C, ek, the list W and ℓ+1 ciphertexts to each signer; S_i returns com_i and m−k ciphertexts.
    out["u_to_s_bits"] = out["C_bits"] + out["ek_bits"] + n * out["com_bits"] + (l + 1) * out["ctx_bits"] + proof
    out["s_to_u_bits"] = out["com_bits"] + (m - k) * out["ctx_bits"] + proof
    out["per_signer_bits"] = out["u_to_s_bits"] + out["s_to_u_bits"]
    out["total_bits"] = n * out["per_signer_bits"]
    out["pi1_kb"] = out["pi2_kb"] = LABRADOR_PROOF_KB
    return {**out, **{key[:-5] + "_kb": val / 8000.0 for key, val in out.items() if key.endswith("_bits")}}


def r1cs_lifting(p, sb, he, cs):
    """Size of each modular equation of the proofs after lifting to the integers.

    An equation F = 0 mod Q is proved as clift(F) = Q·κ over Z.  Returns log2
    of the largest |clift(F)| + Q|κ|, to be compared with M/2 for the proof
    modulus M = 2^256 + 1, so that the lifted equation does not wrap mod M.
    """
    l, k, m = p.ell, p.k, p.m
    bsk, bw, bwp = he["B_sk_inf"], he["B_w_inf"], cs["B_msg"]
    lhs = {
        "CS (mod Q_ext)": p.Q_ext / 2 * (N * p.n_r + 1) + p.Q_ext,
        "round w~ (mod q)": p.q / 2 + p.q * ((m - k) * N * bwp + l + 2),
        "HE.Enc (mod Q_he)": p.Q_he / 2 * (N * p.k_he + 1) + p.Q_he,
        "round vk (mod q)": p.q / 2 + p.q * ((m - k) * N * bsk + 1),
        "w_i^(j) (mod q)": p.q / 2 + p.q * ((m - k) * N * bw + 1),
        "HE.Eval (mod Q_he)": p.Q_he / 2 * N * (bsk + l * bw + p.k_he * he["B_theta_fl_l2"]) + p.Q_he,
    }
    return {name: math.log2(v) for name, v in lhs.items()}


@functools.lru_cache(maxsize=1)
def _load_estimator():
    sys.path.insert(0, str(ESTIMATOR_DIR))
    warnings.filterwarnings("ignore")
    from estimator import LWE, SIS, ND
    return LWE, SIS, ND


def _min_cost_bits(raw):
    bits = {}
    for name, item in raw.items():
        try:
            rop = float(item.get("rop"))
        except Exception:
            continue
        if rop > 0 and math.isfinite(rop):
            bits[name] = math.log2(rop)
    return (min(bits.values()) if bits else math.inf), {k: round(v, 2) for k, v in bits.items()}


def lwe_estimate(n, m, q, beta):
    LWE, _, ND = _load_estimator()
    # β = 0 marks the binary keys and coins of HE and CS.
    dist = ND.Binary if beta == 0 else ND.Uniform(-beta, beta)
    params = LWE.Parameters(n=n, q=q, Xs=dist, Xe=dist, m=m)
    raw = LWE.estimate.rough(params, quiet=True)
    return _min_cost_bits(raw)


def sis_estimate(n, m, q, length):
    _, SIS, _ = _load_estimator()
    params = SIS.Parameters(n=n, q=q, m=m, length_bound=math.floor(length), norm=2)
    raw = SIS.estimate.rough(params, quiet=True)
    return _min_cost_bits(raw)


def instances(p, sb):
    """MSIS and MLWE instances passed to the lattice-estimator.

    Each entry is (name, kind, n, m, modulus, extra, β).  For SIS, n × m is the
    matrix size and extra the ℓ2 length bound.  For LWE, n is the secret
    dimension, m the number of samples, β the ℓ∞ box of secret and error
    (0 for binary), and extra the bits subtracted from the estimate.
    """
    l_ext = p.m + ELL_MSG
    return [
        ("MSIS (beta_sis)", "sis", N * p.k, N * p.m, p.q, sb["beta_sis"], 0),
        ("MSIS (beta_msis)", "sis", N * p.k, N * p.m, p.q, sb["beta_msis"], 0),
        ("MLWE (beta_lwe)", "lwe", N * (p.m - p.k - 1), N * p.k, p.q, 0.0, p.beta_lwe),
        ("HE IND-CPA enc MLWE", "lwe", N * p.k_he, N * (p.k_he + 1), p.Q_he, 0.0, 0),
        ("HE IND-CPA key MLWE", "lwe", N * p.k_he, N * p.k_he, p.Q_he, 0.0, 0),
        ("CS hiding MLWE", "lwe", N * (p.n_r - p.k_ext - l_ext), N * (p.k_ext + l_ext), p.Q_ext, 0.0, 0),
        ("CS setup MLWE (- log2 ell_ext)", "lwe", N * p.k_ext, N * p.n_r, p.Q_ext, math.log2(l_ext), 0),
    ]


def hardness(p, sb):
    out = {}
    for name, kind, n_, m_, mod, extra, beta in instances(p, sb):
        if kind == "sis":
            bits, raw = sis_estimate(n_, m_, mod, extra)
        else:
            bits, raw = lwe_estimate(n_, m_, mod, beta)
            bits -= extra
        out[name] = {"bits": bits, "attacks": raw}
    return out


# FIPS 202 sponge: the padded input fills ⌈(L + pad)/rate⌉ blocks, one permutation each.
def sha3_384_permutations(input_bits):
    return int(math.ceil((input_bits + SHA3_PADDING_BITS) / SHA3_384_RATE_BITS))


# The first 1088 output bits are free; each further block costs one permutation.
def shake256_permutations(input_bits, output_bits):
    absorb = math.ceil((input_bits + SHAKE_PADDING_BITS) / SHAKE256_RATE_BITS)
    squeeze = max(0, math.ceil(output_bits / SHAKE256_RATE_BITS) - 1)
    return int(absorb + squeeze)


# Bits of a bounded integer, counted twice for the two Range decompositions.
def range_bits(bound):
    return 2 * math.ceil(math.log2(2 * bound + 1))


def proof_costs(p, sb, he, cs):
    """Constraint and witness counts of the two ZK-LaBRADOR proofs.

    L_1 (user): Keccak permutations for μ, H_non and H_sig, the samplers, the
    l2 norm of r', the BDLOP opening, the rounding of w̃ and the l+1
    encryptions.  L_i,2 (signer): l2 norms of s_i, the nonces and the flooding
    coins, the rounding of vk_i, the nonce commitments and HE.Eval.  W and K̂
    are prehashed with SHA3-384 outside the proof.  Binary constraints count
    only the non-affine Keccak and sampler gates; the witness size
    bit-decomposes every bounded integer.
    """
    k, m, l, d, kap, rho = p.k, p.m, p.ell, p.k_he, p.k_ext, p.n_r
    lc = m + ELL_MSG
    log_q = math.ceil(math.log2(p.q))
    b_w = math.ceil(math.log2(rounded_modulus(p.q, p.nu_w)))
    b_t = math.ceil(math.log2(rounded_modulus(p.q, p.nu_t)))
    # W and K̂ are prehashed outside the proof; only their 384-bit digests enter L_1.
    bits_W = p.n * (l + 1) * k * N * log_q
    bits_K = (p.n + 1) * k * N * b_t
    bits_m = N
    bits_wt = k * N * b_w
    user_binary = {
        "μ = SHA3-384(d_K ‖ m)": KECCAK_CONSTRAINTS * sha3_384_permutations(PREHASH_BITS + bits_m),
        "H_non: SHAKE256(d_W ‖ μ)": KECCAK_CONSTRAINTS * shake256_permutations(
            2 * PREHASH_BITS, l * (1 + int(math.log2(N)))),
        "H_non: SampleMon (ℓ·N)": l * N,
        "H_sig: ρ = SHA3-384(μ ‖ Ser(w̃))": KECCAK_CONSTRAINTS * sha3_384_permutations(PREHASH_BITS + bits_wt),
        "H_sig: SampleInBall SHAKE256(ρ)": KECCAK_CONSTRAINTS * shake256_permutations(
            PREHASH_BITS, p.beta_c * (KAPPA_IDX + 1)),
        "H_sig: SampleInBall Fisher–Yates": p.beta_c * (3 * N + 4),
    }
    # Binary witness of L_1: Keccak states, sampler bits, m, r', the rounding of w̃,
    # the BDLOP opening and the quotients of the lifted equations.
    keccak_bits = sum(v for key, v in user_binary.items() if "SampleMon" not in key and "Fisher" not in key)
    user_witness = (keccak_bits + l * (1 + int(math.log2(N))) + 2 * l * N + p.beta_c * (3 * N + 4) + N
                    + m * N * range_bits(cs["B_msg"]) + rho * N
                    + k * N * (2 * log_q + 2 * b_w + 1 + 2 * p.nu_w)
                    + (l + 1) * (2 * d + 1) * N
                    + (kap + lc) * N * range_bits(N * rho + 1)
                    + k * N * range_bits((m - k) * N * cs["B_msg"] + l + 2)
                    + (l + 1) * (d + 1) * N * range_bits(N * d + 1)
                    + range_bits(sb["beta_wp"] ** 2))
    # Binary witness of L_i,2: s_i, the ℓ+1 nonces, the vk rounding, the flooding coins
    # and the quotients of the lifted equations.
    signer_witness = (m * N * range_bits(he["B_sk_inf"]) + (l + 1) * m * N * range_bits(he["B_w_inf"])
                      + k * N * (2 * log_q + 1 + 2 * p.nu_t)
                      + (m - k) * (2 * d + 1) * N * range_bits(he["B_theta_fl_l2"])
                      + k * N * range_bits((m - k) * N * he["B_sk_inf"] + 1)
                      + (l + 1) * k * N * range_bits((m - k) * N * he["B_w_inf"] + 1)
                      + (m - k) * (d + 1) * N * range_bits(N * (he["B_sk_inf"] + l * he["B_w_inf"])
                                                           + d * N * he["B_theta_fl_l2"] + 2)
                      + range_bits(sb["beta_sk"] ** 2) + (l + 1) * range_bits(sb["beta_w"] ** 2))
    user = {
        "hash_input_bits": {"W| → d_W (outside proof": bits_W, "K̂| → d_K (outside proof": bits_K,
                            "w̃| (inside proof": bits_wt},
        "prehash_permutations": sha3_384_permutations(bits_W) + sha3_384_permutations(bits_K),
        "binary": user_binary,
        "binary_total": sum(user_binary.values()),
        "arithmetic": {"Norm2 ‖r′‖ ≤ β_w′ (mN)": m * N},
        "linear": {"BDLOP opening mod Q_ext": (kap + lc) * N,
                   "SampleMon one-hot": 2 * l,
                   "rounding of w̃": 2 * k * N,
                   "ℓ+1 encryptions mod Q_he": (l + 1) * (d + 1) * N},
        "witness_bits": user_witness,
    }
    signer = {
        "binary": {},
        "binary_total": 0,
        "arithmetic": {"Norm2 s_i, r_i^(j) ((ℓ+2)·mN)": (l + 2) * m * N,
                       "Norm2 θ_fl ((m−k)(2d+1)N)": (m - k) * (2 * d + 1) * N},
        "linear": {"rounding of vk_i": 2 * k * N,
                   "w_i^(j) = A r_i^(j) mod q": (l + 1) * k * N,
                   "HE.Eval + flooding mod Q_he": (m - k) * (d + 1) * N},
        "witness_bits": signer_witness,
    }
    for proof in (user, signer):
        proof["arithmetic_total"] = sum(proof["arithmetic"].values())
        proof["linear_total"] = sum(proof["linear"].values())
        proof["proof_kb"] = LABRADOR_PROOF_KB
    return {"L1": user, "L2": signer}


def evaluate(p):
    """Compute every bound, estimate and check for one parameter set.

    Returns the parameters, derived quantities, all bounds, the log2 of the
    statistical terms, the lattice-estimator results, sizes, lifting sizes and
    proof costs, and the result of each condition of the security proofs.
    """
    sb = signature_bounds(p)
    he = encryption_bounds(p, sb)
    cs = commitment_bounds(p, sb)
    st = statistical_terms(p, sb, he)
    hard = hardness(p, sb)
    Nm = N * p.m
    checks = {
        "rounding admissible (nu_t)": is_rounding_admissible(p.q, p.nu_t),
        "rounding admissible (nu_w)": is_rounding_admissible(p.q, p.nu_w),
        "2 beta_z' < q/2": 2 * sb["beta_zp"] < p.q / 2,
        "beta_sis < q (MSIS non-trivial)": sb["beta_sis"] < p.q,
        "sigma_w >= sqrt(2 pi) 2N q^(1/m+2/(N(m-k)))": p.sigma_w >= nonce_width_entropy_floor(math.log2(p.q), p.k, p.m),
        "sigma_w >= max{sigma_sk beta_c sqrt(8Nq_s)/2, sqrt(2 pi) 2N q^(1/m+2/(N(m-k))), sqrt((ln(2Nm)+kappa)/pi)}":
            p.sigma_w >= max(p.sigma_sk * p.beta_c * math.sqrt(8 * N * p.qs) / 2,
                             nonce_width_entropy_floor(math.log2(p.q), p.k, p.m),
                             math.sqrt((math.log(2 * Nm) + LAM) / math.pi)),
        "(nu_t, nu_w) = (log2(sigma_w'/(n beta_c)), log2 sigma_w')":
            (p.nu_t, p.nu_w) == rounding_bits(p.sigma_wp, p.n, p.beta_c),
        "beta_lwe = ceil(m ln N)": p.beta_lwe == mlwe_box_target(p.m),
        "sigma_sk >= 4 beta_lwe sqrt(mN)": p.sigma_sk >= 4 * p.beta_lwe * math.sqrt(Nm),
        "mN >= 2 kappa": Nm >= 2 * LAM,
        "sigma_i >= sqrt(ln(6mN)/pi)": min(p.sigma_sk, p.sigma_w) >= math.sqrt(math.log(6 * Nm) / math.pi),
        "sigma >= eta_eps": min(p.sigma_sk, p.sigma_w, p.sigma_wp) >= smoothing_bound(Nm),
        "2 pi (2 beta_r)^2 <= sigma_w'^2": 2 * math.pi * (2 * sb["beta_r"]) ** 2 <= p.sigma_wp**2,
        "2 pi B_shift^2 <= sigma_fl^2": 2 * math.pi * he["B_shift"] ** 2 <= p.sigma_fl**2,
        "exp(-pi q^2/sigma_w^2) <= 1/2": p.q > p.sigma_w,
        "HE: p_he > 2 B_plain": p.p_he > 2 * he["B_plain"],
        "HE: decryption no-wrap": encryption_modulus_min(he, p.p_he) <= p.Q_he and math.gcd(p.p_he, p.Q_he) == 1,
        "CS: p_ext > 2 ||(m,r')||_inf": p.p_ext > 2 * cs["B_m0"],
        "CS: extraction no-wrap": commitment_modulus_min(cs, p.p_ext) <= p.Q_ext and math.gcd(p.p_ext, p.Q_ext) == 1,
        "CS: n_r > k_ext + ell_ext": p.n_r > p.k_ext + cs["ell_ext"],
        "q, Q_he, Q_ext prime": all(is_prime(x) for x in (p.q, p.Q_he, p.Q_ext)),
        "stat terms <= 2^-128": all(v <= -STAT_BITS for kk, v in st.items()
                                    if kk not in ("eps_stat", "correctness", "cp_eps", "cp_log2_R2")),
        "circuit privacy (Renyi R_2 <= e over q_s(m-k) evaluations)": st["cp_log2_R2"] <= 1 / LN2 + 1e-9,
        "CS: binding (statistical, sqrt(N rho) < lambda_1)":
            math.log2(cs["B_bind"]) < log2_commitment_lambda1(p.k_ext, p.n_r, math.log2(p.Q_ext)),
        "correctness <= 2^-128": st["correctness"] <= -LAM,
        "blindness eps_stat <= 2^-40": st["eps_stat"] <= -BLIND_BITS,
        "sigma_w' >= max{4 sqrt(pi) beta_r / eps_stat, sqrt((ln(2Nm)+kappa)/pi)}":
            p.sigma_wp >= max(4 * math.sqrt(math.pi) * sb["beta_r"] / 2.0**-BLIND_BITS,
                              math.sqrt((math.log(2 * Nm) + LAM) / math.pi)),
        "lattice instances >= 128": all(v["bits"] >= LAM for v in hard.values()),
    }
    params = asdict(p)
    for key in ("q", "p_he", "Q_he", "p_ext", "Q_ext"):
        params[key] = str(params[key])
    return {
        "params": params,
        "derived": {"ell": p.ell, "log2_q": math.log2(p.q), "log2_q_nu_t": math.log2(rounded_modulus(p.q, p.nu_t)),
                    "log2_q_nu_w": math.log2(rounded_modulus(p.q, p.nu_w)), "log2_Sc": log2_challenge_space(p.beta_c)},
        "sig_bounds": {k: (math.log2(v) if k.startswith("beta") and v > 0 else v) for k, v in sb.items()},
        "he_bounds": he,
        "cs_bounds": cs,
        "stat_log2": st,
        "hardness": hard,
        "sizes": sizes(p, cs),
        "r1cs_lifting_log2": r1cs_lifting(p, sb, he, cs),
        "proofs": proof_costs(p, sb, he, cs),
        "checks": checks,
        "passes": all(checks.values()),
    }


INSTANCE_LABELS = [
    ("MSIS (beta_sis)", "MSIS β_sis"),
    ("MSIS (beta_msis)", "MSIS β_msis"),
    ("MLWE (beta_lwe)", "MLWE β_lwe = ⌈m ln N⌉"),
    ("HE IND-CPA enc MLWE", "HE IND-CPA, ciphertext"),
    ("HE IND-CPA key MLWE", "HE IND-CPA, key"),
    ("CS hiding MLWE", "CS hiding"),
    ("CS setup MLWE (- log2 ell_ext)", "CS setup ind. − log₂ ℓ_ext"),
]

STAT_LABELS = [
    ("badhash", "Pr[BadHash]", -STAT_BITS),
    ("commitment_guess", "q_s(q_h+q_s)/2^γ", -STAT_BITS),
    ("nonce_collision", "Q·2^(−2κ+2)", -STAT_BITS),
    ("prehash_collision", "prehash collisions 4·q²/2^384 (d_W, d_K, μ, ρ)", -STAT_BITS),
    ("correctness", "correctness error", -LAM),
    ("eps_stat", "ε_stat (blindness)", -BLIND_BITS),
]


FAIL_MARK = " ✗"


def bits_cell(value, target, fmt="{:.1f}"):
    return fmt.format(value) + ("" if value >= target else FAIL_MARK)


def stat_cell(value, target):
    txt = "< −1000" if value < -1000 else "{:.1f}".format(value).replace("-", "−")
    return txt + ("" if value <= target else FAIL_MARK)


def fmt_kb(x):
    return "{:,.2f}".format(x) if x < 10000 else "{:,.0f}".format(x)


def fmt_bits(x):
    return str(int(x).bit_length())


def fmt_power(x):
    return "2^{:.2f}".format(math.log2(float(x)))


def fmt_count(x):
    return "{:,} (2^{:.2f})".format(x, math.log2(x)) if x > 0 else "0"


@dataclass
class TextTable:
    title: str
    headers: list
    aligns: list
    rows: list = field(default_factory=list)
    section_ends: set = field(default_factory=set)
    strong_rows: set = field(default_factory=set)
    caption: str = ""

    def add_row(self, *cells, strong=False):
        if strong:
            self.strong_rows.add(len(self.rows))
        self.rows.append([str(cell) for cell in cells])

    def add_section(self):
        if self.rows:
            self.section_ends.add(len(self.rows) - 1)


def display_width(text):
    return sum(0 if unicodedata.combining(ch) else 1 for ch in text)


def pad_cell(text, width, align):
    gap = width - display_width(text)
    if align == "r":
        return " " * gap + text
    if align == "c":
        return " " * (gap // 2) + text + " " * (gap - gap // 2)
    return text + " " * gap


def render_table(table):
    header_lines = [header.split("\n") for header in table.headers]
    depth = max(len(lines) for lines in header_lines)
    header_lines = [[""] * (depth - len(lines)) + lines for lines in header_lines]
    widths = [max([display_width(line) for line in header_lines[i]] + [display_width(row[i]) for row in table.rows])
              for i in range(len(table.headers))]
    total = sum(widths) + 3 * (len(widths) - 1)

    def line(cells, aligns):
        return " | ".join(pad_cell(cell, width, align) for cell, width, align in zip(cells, widths, aligns)).rstrip()

    out = [table.title, "=" * total]
    for level in range(depth):
        out.append(line([lines[level] for lines in header_lines], table.aligns))
    out.append("-" * total)
    for index, row in enumerate(table.rows):
        out.append(line(row, table.aligns))
        if index in table.section_ends and index < len(table.rows) - 1:
            out.append("-" * total)
    out.append("=" * total)
    if table.caption:
        out += table.caption.split("\n")
    return "\n".join(out)


def case_table(title, columns, label="Parameter"):
    return TextTable(title=title, headers=["", label] + list(columns), aligns=["l", "l"] + ["r"] * len(columns))


def parameter_table(sets):
    t = case_table("BMRaccoon parameters  (λ = {}, q_h = 2^{})".format(LAM, LOG2_QH),
                   ["q_s = 2^{}\nn = {}".format(s.log2_qs, s.n) for s in sets])

    def row(group, label, fn):
        t.add_row(group, label, *[fn(s) for s in sets])

    row("Ring", "ring degree N", lambda s: str(N))
    t.add_section()
    row("Moduli", "signature modulus bits ⌈log₂ q⌉", lambda s: fmt_bits(s.q))
    row("", "BGV modulus bits ⌈log₂ Q_he⌉", lambda s: fmt_bits(s.Q_he))
    row("", "commitment modulus bits ⌈log₂ Q_ext⌉", lambda s: fmt_bits(s.Q_ext))
    t.add_section()
    row("Signature", "public dimension k", lambda s: str(s.k))
    row("", "module width m", lambda s: str(s.m))
    row("", "nonces ℓ", lambda s: str(s.ell))
    row("", "challenge weight β_c", lambda s: str(s.beta_c))
    row("", "σ_sk", lambda s: fmt_power(s.sigma_sk))
    row("", "σ_w", lambda s: fmt_power(s.sigma_w))
    row("", "σ_w'", lambda s: fmt_power(s.sigma_wp))
    row("", "ν_t", lambda s: str(s.nu_t))
    row("", "ν_w", lambda s: str(s.nu_w))
    t.add_section()
    row("HE (BGV)", "rank k_he", lambda s: str(s.k_he))
    row("", "plaintext modulus bits ⌈log₂ p_he⌉", lambda s: fmt_bits(s.p_he))
    row("", "σ_fl", lambda s: fmt_power(s.sigma_fl))
    t.add_section()
    row("CS (BDLOP)", "rank κ = k_ext", lambda s: str(s.k_ext))
    row("", "randomness length ρ = n_r", lambda s: str(s.n_r))
    row("", "extraction modulus bits ⌈log₂ p_ext⌉", lambda s: fmt_bits(s.p_ext))
    return t


def security_table(sets, results, log2_qs):
    t = case_table("Security  —  q_s = 2^{}   (bits; lattice-estimator, core-SVP)".format(log2_qs),
                   ["n = {}".format(s.n) for s in sets], "Instance / statistical term")
    R = [results[s.name] for s in sets]
    for i, (key, label) in enumerate(INSTANCE_LABELS):
        group = "MRaccoon[β_z'] / MRaccoon*" if i == 0 else ("BMRaccoon parts" if i == 3 else "")
        t.add_row(group, label, *[bits_cell(r["hardness"][key]["bits"], LAM) for r in R])
    t.add_section()
    for i, (key, label, target) in enumerate(STAT_LABELS):
        t.add_row("log₂ of statistical terms" if i == 0 else "",
                  "{}  ≤ {}".format(label, "{:.2f}".format(target) if target > 0 else target).replace("-", "−"),
                  *[stat_cell(r["stat_log2"][key], target) for r in R])
    t.add_section()
    t.add_row("Summary", "computational security",
              *["{:.1f}".format(min(v["bits"] for v in r["hardness"].values())) for r in R], strong=True)
    t.add_row("", "correctness", *["{:.1f}".format(-r["stat_log2"]["correctness"]) for r in R], strong=True)
    t.add_row("", "statistical blindness", *["{:.2f}".format(-r["stat_log2"]["eps_stat"]) for r in R], strong=True)
    return t


def size_table(sets, results, log2_qs):
    t = case_table("Sizes  —  q_s = 2^{}   (KB, 1 KB = 1000 bytes)".format(log2_qs),
                   ["n = {}".format(s.n) for s in sets], "Object")
    Z = [results[s.name]["sizes"] for s in sets]

    def row(group, label, key, strong=False):
        t.add_row(group, label, *[fmt_kb(z[key]) for z in Z], strong=strong)

    row("Keys & sig.", "verification key vk_i", "vk_kb")
    row("", "signature σ = (c, z, h)", "sig_kb", True)
    row("", "   z", "z_kb")
    row("", "   h", "h_kb")
    row("", "   c", "c_kb")
    t.add_section()
    row("Messages", "nonce commitment com_i", "com_kb")
    row("", "one HE ciphertext", "ctx_kb")
    row("", "HE key ek", "ek_kb")
    row("", "commitment C", "C_kb")
    row("", "proof π_1 / each π_i,2", "pi1_kb")
    t.add_section()
    row("Communication", "U → S_i  (C, ek, W, ctx_b, ctx_c, π_1)", "u_to_s_kb")
    row("", "S_i → U  (com_i, ctx_z_i, π_i,2)", "s_to_u_kb")
    row("", "per signer", "per_signer_kb", True)
    row("", "total (n signers)", "total_kb", True)
    return t


def proof_table(sets, results, log2_qs):
    t = case_table("ZK-LaBRADOR proofs  —  q_s = 2^{}   (L_1, L_i,2; prehashed hash inputs)".format(log2_qs),
                   ["n = {}".format(s.n) for s in sets], "Constraint / quantity")
    P = [results[s.name]["proofs"] for s in sets]
    for name, title in (("L1", "π_1 for L_1 (user)"), ("L2", "π_i,2 for L_i,2 (each signer)")):
        first = True
        for key in P[0][name].get("hash_input_bits", {}):
            t.add_row(title if first else "", "hash input |{}, bits)".format(key),
                      *[fmt_count(pr[name]["hash_input_bits"][key]) for pr in P])
            first = False
        for key in P[0][name]["binary"]:
            t.add_row(title if first else "", key, *[fmt_count(pr[name]["binary"][key]) for pr in P])
            first = False
        t.add_row(title if first else "", "binary quadratic constraints",
                  *[fmt_count(pr[name]["binary_total"]) for pr in P], strong=True)
        for key in P[0][name]["arithmetic"]:
            t.add_row("", key, *[fmt_count(pr[name]["arithmetic"][key]) for pr in P])
        t.add_row("", "arithmetic quadratic constraints",
                  *[fmt_count(pr[name]["arithmetic_total"]) for pr in P], strong=True)
        for key in P[0][name]["linear"]:
            t.add_row("", key, *[fmt_count(pr[name]["linear"][key]) for pr in P])
        t.add_row("", "linear rows", *[fmt_count(pr[name]["linear_total"]) for pr in P], strong=True)
        if "prehash_permutations" in P[0][name]:
            t.add_row("", "prehash of W, K̂ outside proof (Keccak perms)",
                      *["{:,}".format(pr[name]["prehash_permutations"]) for pr in P])
        t.add_row("", "binary witness (estimate, bits)", *[fmt_count(pr[name]["witness_bits"]) for pr in P])
        t.add_row("", "proof size (KB, ZK-LaBRADOR)", *["≈ {:.0f}".format(pr[name]["proof_kb"]) for pr in P],
                  strong=True)
        t.add_section()
    t.add_row("Per session", "π_1 + n·π_i,2 (KB)",
              *[fmt_kb(pr["L1"]["proof_kb"] + s.n * pr["L2"]["proof_kb"]) for s, pr in zip(sets, P)], strong=True)
    t.caption = ("Prehashing with SHA3-384: d_W = SHA3-384(W), d_K = SHA3-384(K̂) outside the proof; "
                 "μ = SHA3-384(d_K ‖ m),\nb = H_non(W, m, K̂) = SampleMon(SHAKE256(d_W ‖ μ)), "
                 "c = H_sig(w̃, m, K̂) = SampleInBall(SHA3-384(μ ‖ Ser(w̃))) inside L_1; "
                 "one Keccak-f[1600] = 24·1600 binary constraints.\n"
                 "Proof size: ≈ 115–125 KB per ZK-LaBRADOR proof with zero-knowledge, nearly constant in the input size.\n"
                 "Witness: every bounded integer bit-decomposed (two Range decompositions), quotient widths from "
                 "the B_κ bounds.")
    return t


def summary_table(sets, results):
    headers = ["q_s", "n", "⌈log₂ q⌉", "⌈log₂ Q_he⌉", "⌈log₂ Q_ext⌉", "|σ| KB", "|vk| KB", "comm./signer KB",
               "total comm. KB", "security", "correctness", "blindness", "status"]
    t = TextTable(title="Summary of all parameter sets", headers=headers, aligns=["r"] * 12 + ["c"])
    last = None
    for s in sets:
        r = results[s.name]
        if last is not None and s.log2_qs != last:
            t.add_section()
        last = s.log2_qs
        z = r["sizes"]
        t.add_row("2^{}".format(s.log2_qs), str(s.n), fmt_bits(s.q),
                  fmt_bits(s.Q_he), fmt_bits(s.Q_ext), fmt_kb(z["sig_kb"]), fmt_kb(z["vk_kb"]), fmt_kb(z["per_signer_kb"]),
                  fmt_kb(z["total_kb"]), bits_cell(min(v["bits"] for v in r["hardness"].values()), LAM),
                  bits_cell(-r["stat_log2"]["correctness"], LAM),
                  bits_cell(-r["stat_log2"]["eps_stat"], BLIND_BITS, "{:.2f}"),
                  "PASS" if r["passes"] else "FAIL")
    return t


def checks_status(sets, results):
    failed = [(s.name, name) for s in sets for name, ok in results[s.name]["checks"].items() if not ok]
    if not failed:
        count = len(next(iter(results.values()))["checks"])
        return "All {} checks passed for all {} parameter sets.".format(count, len(sets))
    return "\n".join("FAILED  {}: {}".format(name, check) for name, check in failed)


HEADER_LINES = [
    "BMRaccoon concrete parameters",
    "λ = 128, N = 256, q_h = 2^64, ℓ = 32;  q_s ∈ {2^32, 2^64},  n ∈ {32, 64}",
    "β_lwe = ⌈m·ln N⌉,  σ_sk = max{4·β_lwe·√(mN), √(ln(6mN)/π)},",
    "σ_w = max{σ_sk·β_c·√(8N·q_s)/2, √(2π)·2N·q^(1/m + 2/(N(m−k))), √((ln(2Nm)+κ)/π)};",
    "σ_w' = max{4√π·β_r/ε_stat, √((ln(2Nm)+κ)/π)} with ε_stat = 2^−40;  ν_t = ⌊log₂(σ_w'/(n·β_c))⌋,  ν_w = ⌊log₂ σ_w'⌋",
    "Policy: every lattice instance ≥ 128 bits; each statistical term ≤ 2^−128; correctness ≤ 2^−128;",
    "        statistical blindness 2·ε_stat ≤ 2^−40; HE (BGV) and CS (BDLOP) with binary keys and coins,",
    "        B bounds in ℓ∞;",
    "        ZK-LaBRADOR proofs taken as 125 KB each.",
    "Hardness: lattice-estimator, core-SVP (rough), minimum over attacks",
]


def header():
    return "\n".join(HEADER_LINES)


def render_header(text):
    lines = text.split("\n")
    rule = "=" * max(display_width(line) for line in lines)
    return "\n".join([rule] + lines + [rule])


def main():
    """Derive the parameter sets, evaluate them and print all tables.

    The exit code is 0 only if every check passes for every set.
    """
    sets = [derive_parameter_set(r) for r in SEARCH_RESULTS]
    print(render_header(header()))
    print()
    print(render_table(parameter_table(sets)))
    print()
    print("Evaluating {} parameter sets with the lattice-estimator ...".format(len(sets)), flush=True)

    results = {}
    for s in sets:
        results[s.name] = evaluate(s)

    for log2_qs in sorted({s.log2_qs for s in sets}):
        group = [s for s in sets if s.log2_qs == log2_qs]
        for table in (security_table(group, results, log2_qs), size_table(group, results, log2_qs),
                      proof_table(group, results, log2_qs)):
            print()
            print(render_table(table))
    print()
    print(render_table(summary_table(sets, results)))
    print()
    print(checks_status(sets, results))
    return 0 if all(r["passes"] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

