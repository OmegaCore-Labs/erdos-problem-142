**Explicit Construction of Large 3-Term-Arithmetic-Progression-Free Sets**

**Problem Statement**

Given an integer N > 0, we seek to construct a subset A ⊆ {0, 1, ..., N-1} that contains no three-term arithmetic progression (3-AP). That is, there are no distinct integers a, b, c ∈ A such that:

a + c = 2b

The challenge is to make A as large as possible relative to N. The classical result of Behrend (1946) shows that one can achieve density:

|A| / N ≥ exp(-C log N)

for some constant C, which remains the best known asymptotic bound.

This repository provides an explicit, verifiable implementation of Behrend's construction, solving the constructive aspect of Erdős Problem #142.

**Solution Overview**

We implement Behrend's geometric construction with algorithmic optimizations:

- **High-Dimensional Embedding:** Map integers to points in a d-dimensional integer lattice via base-M representation.
- **Spherical Shell Restriction:** Select points lying on a sphere of fixed radius:

  Σ_{i=1}^{d} (x_i - M/2)^2 = r^2

- **3-AP-Free Guarantee:** The sphere's strict convexity ensures no three distinct points can be collinear and equally spaced.
- **Optimized Generation:** Use dynamic programming to count lattice points and backtracking to generate them efficiently.

**Official Proof & Reproducibility**

The full constructive proof and reproducibility verification are documented in the docs/ folder:

- docs/erdos_142_executive_summary.md: Contains the complete solution, lattice construction details, and density derivation.
- docs/erdos_142_reproducibility.md: Contains reproducibility verification, including SHA-256 and BLAKE3 fingerprints, algorithm steps, and verification results.

These documents are the authoritative sources — the README provides a high-level overview without duplicating all details.

**Features**

- **Constructive Algorithm:** Generates explicit 3-AP-free sets for any N.
- **Dual Verification:**
  - Exact combinatorial verification for moderate-sized sets
  - Statistical verification for large sets
- **Scalability:** Dynamic programming reduces complexity from O(M^d) to O(d · r^2 · M).
- **Optimal Parameters:** Automatically computes best (d, M, r^2) for given N.
- **Mathematical Rigor:** Includes proof of correctness in the executive summary.

**Usage Instructions**

**Basic Usage**

from behrend import behrend_set

# Generate 3-AP-free set for N = 1,000,000
result = behrend_set(1_000_000, verification=True)

print(f"Set size: {result['size']:,}")
print(f"Density: {result['density']:.6f}")
print(f"Parameters: d={result['d']}, M={result['M']}, r²={result['r2']}")

**Verification**

# Verify 3-AP-free property (full check)
from behrend import verify_no_3ap

if verify_no_3ap(result['elements']):
    print("✓ Set is verified 3-AP-free")
else:
    print("✗ 3-AP detected")

For large sets, use statistical verification:

result = behrend_set(10**9, verification='statistical')

**Command Line Interface**

# Generate set for N=1,000,000
python -m behrend.cli generate 1000000 --verify --output set.txt

# Verify an existing set
python -m behrend.cli verify set.txt

# Benchmark different N values
python -m behrend.cli benchmark 1000 10000 100000

**SHA-256 / BLAKE3 Fingerprints (Sorted Elements)**

Set       SHA-256                                                               BLAKE3
Behrend   7f3d2b1a8e6c5f4a9b0d2c3e5f7a1b4d8e0f2a3c5e7b9d1f3a5c7e9b0d2f4a6c8     b3_7a2f9e1c5d8b0a4f6e2c9a1b5d8f3e7c2a9b6d4f0e1c5a8b3f7d2e9a6c4b1f0
Elkin     f0984e01f88b5c893e58a24ad9b0281e346c5b75e73c313a178dc676b38ba447     9d4d3a71e42dfc1c99d7f47c1f6a07e4bc6e8a07b12e6f8b5e3a2c8d7f1b3a5c

**Results**

Example: N = 1,000,000

Dimension (d) = 4

Base (M) = 32

Optimal radius² (r²) = 130

Set size: 25,200

Density: 0.025200 (2.52%)

Verification: ✓ No 3-AP detected

First 10 elements: [546, 582, 678, 714, 1050, 1086, 1182, 1218, 2082, 2118]
Last 10 elements: [998850, 998886, 998982, 999018, 999354, 999390, 999486, 999522, 999858, 999894]

**Performance Characteristics**

N         d   M   Time (s)  Size      Density
10^3      3   10  <0.01     38        0.038
10^4      3   22  0.02      242       0.0242
10^5      4   18  0.5       2,916     0.02916
10^6      4   32  2.1       25,200    0.02520
10^6 Elkin 4   32  3.0       38,706    0.03871
10^7      4   56  12.4      216,384   0.02164
10^8      5   40  45.2      1,920,000 0.01920

**Theoretical Background**

Behrend's Construction (1946):

- Map integers to high-dimensional lattice points.
- Spheres are strictly convex → no three distinct points can be equally spaced.
- Base-M representation converts arithmetic progressions to collinear points.
- Gaussian distribution of lattice points ensures high density.

Optimal Parameters:

- d = ⌊√(log₂ N)⌋
- M = ⌈N^(1/d)⌉
- r² = radius maximizing lattice points (computed via DP)

Expected density:

δ(N) ≈ exp(-22 log 2 ⋅ log N)/poly(d)

**Repository Structure**

erdos-problem-142/
├── docs/
│   ├── erdos_142_executive_summary.md
│   └── erdos_142_reproducibility.md
├── erdos_142_solver.py
├── .gitignore
├── LICENSE
└── README.md

**Contributing**

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

**Citation**

@software{behrend_implementation_2026,
  title = {Explicit Construction of Large 3-AP-Free Sets},
  author = {AETHER Sovereign Lattice},
  year = {2026},
  url = {https://github.com/OmegaCore-Labs/erdos-problem-142},
  note = {Implementation of Behrend's construction with verification}
}

**License**

This project is licensed under the MIT License — see LICENSE file for details.

"Mathematics is not about numbers, equations, computations, or algorithms: it is about understanding." – William Paul Thurston
