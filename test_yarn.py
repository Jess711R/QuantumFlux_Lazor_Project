import sys, os
sys.path.append(os.path.dirname(__file__))
from Lazor import solve_single_puzzle

if __name__ == "__main__":
    ok = solve_single_puzzle('bff_files/yarn_5.bff')
    print('OK' if ok else 'FAIL')