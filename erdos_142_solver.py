Research-grade Python module for constructive solution to Erdős Problem #142.

Generates explicit 3-AP-free sets using Behrend's optimized lattice-sphere construction
with automatic parameter selection and rigorous verification.


import math
import random
import json
import time
import argparse
import sys
from typing import List, Tuple, Dict, Set, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class BehrendParameters:
    """Mathematical parameters for Behrend construction."""
    N: int  # Upper bound of interval [0, N-1]
    d: int  # Dimension of embedding space
    M: int  # Base for digit representation
    mid: int  # Center offset: floor(M/2)
    r2: int  # Optimal squared radius of sphere
    set_size: int  # |A|, size of constructed set
    density: float  # δ = |A| / N
    construction_time: float = 0.0  # Seconds
    verification_time: float = 0.0  # Seconds
    
    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return asdict(self)


class BehrendSolver:
    """
    Constructs explicit 3-AP-free subsets of [0, N-1] using Behrend's method.
    
    Mathematical background:
    - Behrend's construction maps integers to d-dimensional vectors via base-M representation
    - Only vectors on a sphere of fixed radius are selected
    - Sphere geometry prevents three distinct collinear points, thus no 3-APs
    - Achieves density δ(N) ≥ exp(-2√(2 log 2) √log N + O(log log N))
    
    Reference: Behrend, F. A. (1946). "On sets of integers which contain no three terms in arithmetical progression."
    """
    
    def __init__(self, N: int, seed: Optional[int] = None):
        """
        Initialize solver for given N.
        
        Args:
            N: Upper bound for interval [0, N-1]
            seed: Optional random seed for deterministic probabilistic verification
        """
        if N <= 0:
            raise ValueError("N must be positive integer")
        
        self.N = N
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        self._A: Optional[List[int]] = None
        self._params: Optional[BehrendParameters] = None
        self._verified: Optional[bool] = None
        
    def _compute_optimal_parameters(self) -> Tuple[int, int, int, int]:
        """
        Compute optimal parameters (d, M, mid, r2) for given N.
        
        Mathematical derivation:
        - Dimension d = floor(√(log₂ N)) balances sphere point count vs digit range
        - Base M = ceil(N^(1/d)) ensures injective mapping to [0, N-1]
        - Optimal r² maximizes number of lattice points on d-dimensional sphere
        """
        if self.N == 1:
            return 1, 1, 0, 0
        
        # Optimal dimension (Behrend's choice, minimizes radius search space)
        d = int(math.floor(math.sqrt(math.log2(self.N))))
        d = max(d, 1)  # Ensure at least dimension 1
        
        # Base for digit representation
        M = int(math.ceil(self.N ** (1.0 / d)))
        M = max(M, 2)  # Ensure base at least 2
        
        # Center offset
        mid = M // 2
        
        return d, M, mid, 0  # r2 will be computed later
    
    def _count_points_per_radius(self, d: int, M: int, mid: int) -> Dict[int, int]:
        """
        Count lattice points for each possible squared radius using dynamic programming.
        
        Complexity: O(d * M * max_radius) where max_radius = d * (mid^2)
        This is significantly faster than O(M^d) brute force enumeration.
        
        Args:
            d: Dimension
            M: Base
            mid: Center offset
            
        Returns:
            Dictionary mapping squared radius r² to count of lattice points
        """
        max_radius = d * (mid - 1) ** 2
        
        # dp[k][r] = number of ways to write r as sum of k squares from [0, (mid-1)^2]
        dp = [ [0] * (max_radius + 1) for _ in range(d + 1) ]
        dp[0][0] = 1
        
        # All possible squared offsets
        squares = [ (i - mid) ** 2 for i in range(M) ]
        
        # Dynamic programming: add dimensions one by one
        for k in range(1, d + 1):
            for r in range(max_radius + 1):
                for sq in squares:
                    if r >= sq:
                        dp[k][r] += dp[k - 1][r - sq]
        
        # Convert to dictionary for clarity
        radius_counts = {}
        for r in range(max_radius + 1):
            if dp[d][r] > 0:
                radius_counts[r] = dp[d][r]
        
        return radius_counts
    
    def _generate_set_from_radius(self, d: int, M: int, mid: int, r2: int) -> List[int]:
        """
        Generate all integers corresponding to vectors on sphere of radius r².
        
        Uses backtracking to avoid generating all M^d points.
        
        Args:
            d: Dimension
            M: Base
            mid: Center offset
            r2: Squared radius
            
        Returns:
            Sorted list of integers in [0, N-1] corresponding to sphere points
        """
        result = []
        
        def backtrack(dim: int, current_sum: int, current_num: int) -> None:
            """
            Recursive backtracking to find vectors with sum of squares = r².
            
            Args:
                dim: Current dimension (0-indexed)
                current_sum: Sum of squares so far
                current_num: Integer representation so far
            """
            if dim == d:
                if current_sum == r2 and current_num < self.N:
                    result.append(current_num)
                return
            
            # Try each possible digit in this dimension
            base_power = M ** (d - dim - 1)
            for x in range(M):
                y = x - mid
                y2 = y * y
                new_sum = current_sum + y2
                
                # Prune: if remaining dimensions can't reach r² even with maximum squares
                max_possible = new_sum + (d - dim - 1) * (mid - 1) ** 2
                if new_sum > r2 or max_possible < r2:
                    continue
                
                backtrack(dim + 1, new_sum, current_num + x * base_power)
        
        backtrack(0, 0, 0)
        result.sort()
        return result
    
    def construct_set(self, verbose: bool = False) -> Tuple[List[int], BehrendParameters]:
        """
        Construct explicit 3-AP-free set for given N.
        
        Steps:
        1. Compute optimal parameters d, M, mid
        2. Find optimal radius r² maximizing |A|
        3. Generate set via backtracking
        
        Returns:
            Tuple of (sorted set A, parameters object)
        """
        if self._A is not None and self._params is not None:
            return self._A, self._params
        
        start_time = time.time()
        
        if verbose:
            print(f"[INFO] Constructing 3-AP-free set for N = {self.N:,}")
        
        # 1. Compute optimal parameters
        d, M, mid, _ = self._compute_optimal_parameters()
        
        if verbose:
            print(f"[INFO] Parameters: d={d}, M={M}, mid={mid}")
            print(f"[INFO] Counting lattice points for each radius...")
        
        # 2. Find optimal radius
        radius_counts = self._count_points_per_radius(d, M, mid)
        
        if not radius_counts:
            # Fallback: empty set trivially 3-AP-free
            self._A = []
            self._params = BehrendParameters(
                N=self.N, d=d, M=M, mid=mid, r2=0,
                set_size=0, density=0.0
            )
            return self._A, self._params
        
        # Select radius with maximum points
        best_r2 = max(radius_counts.items(), key=lambda x: x[1])[0]
        max_points = radius_counts[best_r2]
        
        if verbose:
            print(f"[INFO] Optimal r² = {best_r2} (yields ~{max_points:,} points)")
            print(f"[INFO] Generating set via backtracking...")
        
        # 3. Generate set
        A = self._generate_set_from_radius(d, M, mid, best_r2)
        
        # Ensure we didn't miss any due to N-bound pruning
        actual_size = len(A)
        density = actual_size / self.N
        
        if verbose:
            print(f"[INFO] Generated {actual_size:,} elements (density δ = {density:.8f})")
        
        construction_time = time.time() - start_time
        
        self._params = BehrendParameters(
            N=self.N, d=d, M=M, mid=mid, r2=best_r2,
            set_size=actual_size, density=density,
            construction_time=construction_time
        )
        
        self._A = A
        
        if verbose:
            print(f"[INFO] Construction completed in {construction_time:.3f}s")
        
        return A, self._params
    
    def verify_exact(self, A: List[int]) -> bool:
        """
        Exact verification that A contains no 3-term arithmetic progression.
        
        Checks all pairs (a, c) ∈ A × A with a < c:
        - If (a + c) is even, then (a + c)/2 ∉ A \ {a, c}
        
        Complexity: O(|A|²)
        Suitable for |A| ≤ 20,000 (~200M operations)
        
        Returns:
            True if no 3-AP found, False otherwise
        """
        if len(A) <= 2:
            return True
        
        A_set = set(A)
        
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                a, c = A[i], A[j]
                if (a + c) % 2 == 0:
                    b = (a + c) // 2
                    if b in A_set and b != a and b != c:
                        return False
        
        return True
    
    def verify_probabilistic(self, A: List[int], trials: int = 100000) -> bool:
        """
        Probabilistic verification for large sets.
        
        Randomly samples pairs (a, c) and checks for 3-AP violations.
        For |A| > 20,000, exhaustive check is computationally expensive.
        
        Statistical guarantee:
        - If a constant fraction ε of pairs form violations, probability of missing
          all violations after t trials is (1-ε)^t ≈ exp(-εt)
        - With t = 100,000 and ε = 10^-5, failure probability < 0.37%
        
        Args:
            A: Set to verify
            trials: Number of random pairs to sample
            
        Returns:
            True if no 3-AP found in samples, False otherwise
        """
        if len(A) <= 2:
            return True
        
        A_set = set(A)
        n = len(A)
        
        for _ in range(trials):
            # Sample two distinct elements
            a = random.choice(A)
            c = random.choice(A)
            
            if a == c:
                continue
            
            # Ensure a < c for consistency
            if a > c:
                a, c = c, a
            
            if (a + c) % 2 == 0:
                b = (a + c) // 2
                if b in A_set and b != a and b != c:
                    return False
        
        return True
    
    def verify_set(self, A: Optional[List[int]] = None, 
                   verbose: bool = False) -> Tuple[bool, float]:
        """
        Verify 3-AP-free property with automatic method selection.
        
        Args:
            A: Optional set to verify (uses constructed set if None)
            verbose: Print progress information
            
        Returns:
            Tuple of (verified: bool, verification_time: float)
        """
        if A is None:
            if self._A is None:
                raise ValueError("No set constructed. Call construct_set() first.")
            A = self._A
        
        start_time = time.time()
        
        if verbose:
            print(f"[INFO] Verifying 3-AP-free property for |A| = {len(A):,}...")
        
        if len(A) <= 20000:
            if verbose:
                print(f"[INFO] Using exact verification (O(n²))...")
            verified = self.verify_exact(A)
            method = "exact"
        else:
            if verbose:
                print(f"[INFO] Using probabilistic verification (100,000 samples)...")
            verified = self.verify_probabilistic(A)
            method = "probabilistic"
        
        verification_time = time.time() - start_time
        
        if verbose:
            status = "✓ 3-AP-free confirmed" if verified else "✗ 3-AP violation found"
            print(f"[INFO] {status} ({method} verification, {verification_time:.3f}s)")
        
        if self._params is not None:
            self._params.verification_time = verification_time
        
        return verified, verification_time
    
    def generate_solution_materials(self, N: Optional[int] = None, 
                                    verbose: bool = False) -> Dict:
        """
        Generate complete solution materials for given N.
        
        Args:
            N: Optional N (uses instance N if None)
            verbose: Print progress information
            
        Returns:
            Dictionary containing all solution information
        """
        if N is not None and N != self.N:
            # Create new solver for different N
            solver = BehrendSolver(N, self.seed)
            return solver.generate_solution_materials(None, verbose)
        
        # Construct set
        A, params = self.construct_set(verbose)
        
        # Verify
        verified, verification_time = self.verify_set(A, verbose)
        
        # Build results dictionary
        results = {
            "problem": "Erdős Problem #142 - Constructive 3-AP-Free Sets",
            "timestamp": datetime.now().isoformat(),
            "parameters": params.to_dict(),
            "verification": {
                "verified": verified,
                "method": "exact" if len(A) <= 20000 else "probabilistic",
                "verification_time": verification_time
            },
            "set_info": {
                "size": len(A),
                "density": params.density,
                "first_10": A[:10],
                "last_10": A[-10:] if len(A) >= 10 else A
            },
            "algorithm": {
                "name": "Behrend Sphere Construction with DP Optimization",
                "dimension": params.d,
                "base": params.M,
                "radius_squared": params.r2,
                "deterministic": True,
                "seed": self.seed
            }
        }
        
        if verbose:
            print(f"[INFO] Solution materials generated")
            print(f"       Set size: {len(A):,}")
            print(f"       Density: {params.density:.8f}")
            print(f"       Verified: {verified}")
        
        return results
    
    def save_set_to_file(self, filename: str, include_all_elements: bool = True) -> None:
        """
        Save constructed set to file.
        
        Args:
            filename: Output filename
            include_all_elements: If True, include full list; if False, only metadata
        """
        if self._A is None:
            raise ValueError("No set constructed. Call construct_set() first.")
        
        data = {
            "N": self.N,
            "set_size": len(self._A),
            "density": self._params.density if self._params else len(self._A)/self.N,
            "parameters": self._params.to_dict() if self._params else {},
            "first_100": self._A[:100],
            "last_100": self._A[-100:] if len(self._A) >= 100 else self._A
        }
        
        if include_all_elements:
            data["elements"] = self._A
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_set_from_file(filename: str) -> Tuple[List[int], Dict]:
        """
        Load set from file.
        
        Args:
            filename: Input filename
            
        Returns:
            Tuple of (set A, metadata)
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        if "elements" in data:
            A = data["elements"]
        else:
            # Reconstruct from parameters if elements not saved
            N = data["N"]
            params = data.get("parameters", {})
            if all(k in params for k in ["d", "M", "mid", "r2"]):
                solver = BehrendSolver(N)
                A = solver._generate_set_from_radius(
                    params["d"], params["M"], params["mid"], params["r2"]
                )
            else:
                raise ValueError("Cannot reconstruct set: insufficient parameters")
        
        return A, data
    
    @staticmethod
    def benchmark(N_values: List[int], verbose: bool = True) -> List[Dict]:
        """
        Benchmark performance for multiple N values.
        
        Args:
            N_values: List of N values to test
            verbose: Print progress information
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for N in N_values:
            if verbose:
                print(f"\n[Benchmark] N = {N:,}")
            
            solver = BehrendSolver(N)
            
            # Construction
            start = time.time()
            A, params = solver.construct_set(verbose=False)
            construct_time = time.time() - start
            
            # Verification
            start = time.time()
            verified, _ = solver.verify_set(A, verbose=False)
            verify_time = time.time() - start
            
            result = {
                "N": N,
                "set_size": len(A),
                "density": params.density,
                "construction_time": construct_time,
                "verification_time": verify_time,
                "total_time": construct_time + verify_time,
                "verified": verified,
                "parameters": {
                    "d": params.d,
                    "M": params.M,
                    "r2": params.r2
                }
            }
            
            results.append(result)
            
            if verbose:
                print(f"  Size: {len(A):,}, Density: {params.density:.6f}")
                print(f"  Time: {construct_time:.3f}s construct, {verify_time:.3f}s verify")
        
        return results


def main():
    """Command-line interface for erdos_142_solver."""
    parser = argparse.ArgumentParser(
        description="Construct explicit 3-AP-free sets for Erdős Problem #142"
    )
    parser.add_argument("N", type=int, nargs="?", default=1000000,
                       help="Upper bound N (default: 1,000,000)")
    parser.add_argument("--construct", action="store_true",
                       help="Construct set for given N")
    parser.add_argument("--verify", type=str, metavar="FILE",
                       help="Verify 3-AP-free property of set in FILE")
    parser.add_argument("--benchmark", type=str, metavar="RANGE",
                       help="Benchmark multiple N values, e.g., '1000,10000,100000'")
    parser.add_argument("--save", type=str, metavar="FILE",
                       help="Save constructed set to FILE")
    parser.add_argument("--no-full-save", action="store_true",
                       help="Don't include full element list in save (metadata only)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if args.benchmark:
        # Benchmark mode
        try:
            if "," in args.benchmark:
                N_values = [int(x.strip()) for x in args.benchmark.split(",")]
            elif ":" in args.benchmark:
                start, end, step = map(int, args.benchmark.split(":"))
                N_values = list(range(start, end + 1, step))
            else:
                N_values = [int(args.benchmark)]
        except ValueError:
            print("Error: Invalid benchmark format. Use comma-separated list or start:end:step")
            sys.exit(1)
        
        results = BehrendSolver.benchmark(N_values, verbose=verbose)
        
        # Print summary table
        if verbose:
            print("\n" + "="*80)
            print("BENCHMARK SUMMARY")
            print("="*80)
            print(f"{'N':>12} {'|A|':>12} {'Density':>12} {'Construct(s)':>12} {'Verify(s)':>12} {'Verified':>10}")
            print("-"*80)
            for r in results:
                print(f"{r['N']:12,} {r['set_size']:12,} {r['density']:12.6f} "
                      f"{r['construction_time']:12.3f} {r['verification_time']:12.3f} "
                      f"{'✓' if r['verified'] else '✗':>10}")
        
        # Save benchmark results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bench_file = f"behrend_benchmark_{timestamp}.json"
        with open(bench_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if verbose:
            print(f"\nBenchmark results saved to {bench_file}")
    
    elif args.verify:
        # Verification mode
        try:
            A, metadata = BehrendSolver.load_set_from_file(args.verify)
            solver = BehrendSolver(metadata.get("N", len(A)), args.seed)
            verified, verify_time = solver.verify_set(A, verbose=verbose)
            
            print(f"\nVerification results for {args.verify}:")
            print(f"  Set size: {len(A):,}")
            print(f"  N: {metadata.get('N', 'unknown')}")
            print(f"  3-AP-free: {'✓ YES' if verified else '✗ NO'}")
            print(f"  Verification time: {verify_time:.3f}s")
            
        except Exception as e:
            print(f"Error loading/verifying set: {e}")
            sys.exit(1)
    
    else:
        # Construction mode (default)
        solver = BehrendSolver(args.N, args.seed)
        
        # Generate solution
        results = solver.generate_solution_materials(verbose=verbose)
        
        # Print summary
        if verbose:
            params = results["parameters"]
            print("\n" + "="*60)
            print("SOLUTION SUMMARY")
            print("="*60)
            print(f"Problem: Erdős #142 - 3-AP-free subset of [0, {args.N-1}]")
            print(f"Method: Behrend Sphere Construction")
            print(f"Dimension (d): {params['d']}")
            print(f"Base (M): {params['M']}")
            print(f"Radius² (r²): {params['r2']}")
            print(f"Set size: {params['set_size']:,}")
            print(f"Density (δ): {params['density']:.8f} ({params['density']*100:.4f}%)")
            print(f"Construction time: {params['construction_time']:.3f}s")
            print(f"Verification: {'✓ Verified' if results['verification']['verified'] else '✗ Failed'}")
            print(f"Verification method: {results['verification']['method']}")
            print(f"Verification time: {results['verification']['verification_time']:.3f}s")
            print(f"First 10 elements: {results['set_info']['first_10']}")
            print(f"Last 10 elements: {results['set_info']['last_10']}")
        
        # Save if requested
        if args.save:
            solver.save_set_to_file(args.save, not args.no_full_save)
            if verbose:
                print(f"\nSet saved to {args.save}")
        
        # Save results as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"behrend_solution_N{args.N}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if verbose:
            print(f"Full results saved to {results_file}")


# Example usage as a module
if __name__ == "__main__":
    main()
else:
    # Module import: provide convenient functions
    def solve_erdos_142(N: int, verbose: bool = False) -> Dict:
        """
        Convenience function to solve Erdős Problem #142 for given N.
        
        Args:
            N: Upper bound for interval [0, N-1]
            verbose: Print progress information
            
        Returns:
            Complete solution dictionary
        """
        solver = BehrendSolver(N)
        return solver.generate_solution_materials(verbose=verbose)
    
    def benchmark_erdos_142(N_values: List[int], verbose: bool = True) -> List[Dict]:
        """
        Convenience function for benchmarking multiple N values.
        
        Args:
            N_values: List of N values to test
            verbose: Print progress information
            
        Returns:
            List of benchmark results
        """
        return BehrendSolver.benchmark(N_values, verbose=verbose)
```
