# neuro_driver.py
# Simple driver that computes demo probabilistic percepts and queries Prolog.
# Requires: pyswip (pip install pyswip)

from pyswip import Prolog, Functor, Variable, Query
import random

# Demo "percept classifier" (toy). Replace with a real model later.
def percept_probs_for_cell(cell):
    x,y = cell
    # deterministic toy mapping for demo clarity:
    # If visited at [1,2] -> high breeze; otherwise small noise.
    if cell == (1,2):
        return {'breeze_prob': 0.9, 'stench_prob': 0.05}
    if cell == (1,1):
        return {'breeze_prob': 0.1, 'stench_prob': 0.05}
    # far cells: low probabilities
    return {'breeze_prob': 0.02 + random.random()*0.03,
            'stench_prob': 0.01 + random.random()*0.02}

def main():
    prolog = Prolog()
    prolog.consult("neuro_symbolic.pl")

    # Clear any existing python_output facts (precaution)
    list(prolog.query("retractall(python_output(_,_,_))"))

    # We will compute percept probs only for visited cells (that is the conceptual flow)
    visited_cells = [ (1,1), (1,2) ]  # matches Prolog visited facts
    for c in visited_cells:
        probs = percept_probs_for_cell(c)
        # assert python_output([X,Y], breeze_prob, 0.9).
        prolog.assertz(f"python_output([{c[0]},{c[1]}], breeze_prob, {probs['breeze_prob']})")
        prolog.assertz(f"python_output([{c[0]},{c[1]}], stench_prob, {probs['stench_prob']})")
        print(f"Asserted for {c}: {probs}")

    # Query risk for some example cells
    cells_to_check = [(2,2), (1,3), (4,4)]
    for cell in cells_to_check:
        q1 = list(prolog.query(f"risk_of_pit([{cell[0]},{cell[1]}], R)"))
        q2 = list(prolog.query(f"risk_of_wampa([{cell[0]},{cell[1]}], R2)"))
        rp = q1[0]['R'] if q1 else 0.0
        rw = q2[0]['R2'] if q2 else 0.0
        print(f"Cell {cell}: pit risk={rp:.3f}, wampa risk={rw:.3f}, total={rp+rw:.3f}")
        # Also ask Prolog if it's safe_move:
        safe = list(prolog.query(f"safe_move([{cell[0]},{cell[1]}])"))
        print("  => safe_move:", bool(safe))

    # List all safe moves Prolog currently knows
    safe_list = list(prolog.query("list_safe_moves(L)"))
    if safe_list:
        print("Safe moves list:", safe_list[0]['L'])
    else:
        print("No safe moves found.")

if __name__ == "__main__":
    main()
