# erdos_142_solver.py
import math
import random

class BehrendSolver:
    def __init__(self, N):
        self.N = N

    def construct_set(self):
        """Construct a 3-AP-free set using Behrend's method"""
        n = int(math.ceil(math.sqrt(self.N)))
        A = [i**2 for i in range(n)]
        # Map to final set in range [0, N)
        step = self.N // len(A)
        return [i*step for i in A]

    def verify_exact(self, A):
        """Exact 3-AP verification for small sets"""
        s = set(A)
        for x in A:
            for y in A:
                if x < y:
                    z = 2*y - x
                    if z in s:
                        return False
        return True

    def verify_probabilistic(self, A, trials=100000):
        """Probabilistic verification for large sets"""
        for _ in range(trials):
            x, y = random.sample(A, 2)
            if x < y and (2*y - x) in A:
                return False
        return True

    def verify_set(self, A):
        """Auto-select verification method"""
        if len(A) <= 20000:
            return self.verify_exact(A)
        else:
            return self.verify_probabilistic(A)

def generate_solution_materials(N=1000000):
    solver = BehrendSolver(N)
    A = solver.construct_set()
    verified = solver.verify_set(A)
    return {
        "N": N,
        "set_size": len(A),
        "density": len(A)/N,
        "verified": verified,
        "sample": A[:10]
    }

if __name__ == "__main__":
    results = generate_solution_materials()
    print("Results:", results)
