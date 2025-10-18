import numpy as np
import cvxpy as cp

def perform_targeted_embedding_surgery_neg(query_vec, doc_vecs, pairs, margin):
    """
    Apply embedding surgery by modifying only relevant document vectors (r),
    to satisfy feedback constraints (r ranked above n).

    Parameters:
    - query_vec: numpy array of shape (d,)
    - doc_vecs: numpy array of shape (k, d)
    - pairs: list of (n, r) index tuples where r should rank above n
    - margin: enforced separation between r and n
    - normalize: whether to normalize updated vectors

    Returns:
    - updated_doc_vecs: numpy array of shape (k, d) with modified vectors
    """
    
    d = query_vec.shape[1]
    
    # select the relevant docs
    indices = sorted(set(n for n, _ in pairs))
    delta_vars = {n: cp.Variable(d) for n in indices}

    # accumulate constraints
    constraints = []
    for n, r in pairs:
        delta_n = delta_vars[n]
        sim_n = query_vec @ (doc_vecs[n] + delta_n)
        sim_r = query_vec @ doc_vecs[r]  # fixed, not updated
        constraints.append(sim_r >= (sim_n + margin))
    
    # solve
    objective = cp.Minimize(cp.sum_squares(cp.vstack([delta_vars[n] for n in indices])))
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP)
    except:
        print('Solver error')
        print(pairs)
        return doc_vecs, indices, {}

    # Apply updates
    updated = doc_vecs.copy()
    for n in indices:
        updated[n] += delta_vars[n].value

    deltas_serializable = {i: delta.value.tolist() for i, delta in delta_vars.items()}
    return updated, indices, deltas_serializable

def perform_targeted_embedding_surgery_pos(query_vec, doc_vecs, pairs, margin):
    """
    Apply embedding surgery by modifying only relevant document vectors (r),
    to satisfy feedback constraints (r ranked above n).

    Parameters:
    - query_vec: numpy array of shape (d,)
    - doc_vecs: numpy array of shape (k, d)
    - pairs: list of (n, r) index tuples where r should rank above n
    - margin: enforced separation between r and n
    - normalize: whether to normalize updated vectors

    Returns:
    - updated_doc_vecs: numpy array of shape (k, d) with modified vectors
    """
    d = query_vec.shape[1]
    
    # select the relevant docs
    indices = sorted(set(r for _, r in pairs))
    delta_vars = {r: cp.Variable(d) for r in indices}

    # accumulate constraints
    constraints = []
    for n, r in pairs:
        delta_r = delta_vars[r]
        sim_r = query_vec @ (doc_vecs[r] + delta_r)
        sim_n = query_vec @ doc_vecs[n]  # fixed, not updated
        constraints.append(sim_r >= (sim_n + margin))

    # solve
    objective = cp.Minimize(cp.sum_squares(cp.vstack([delta_vars[r] for r in indices])))
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP)
    except:
        print('Solver error')
        print(pairs)
        return doc_vecs, indices, {}

    # Apply updates
    updated = doc_vecs.copy()
    for r in indices:
        updated[r] += delta_vars[r].value
        
    deltas_serializable = {i: delta.value.tolist() for i, delta in delta_vars.items()}
    return updated, indices, deltas_serializable

def perform_embedding_surgery(query_vec, doc_vecs, pairs, margin):
    """
    Apply embedding surgery by modifying both relevant (r) and non-relevant (n)
    document vectors to satisfy feedback constraints: r ranked above n.

    Parameters:
    - query_vec: numpy array of shape (d,)
    - doc_vecs: numpy array of shape (k, d)
    - pairs: list of (n, r) index tuples where r should rank above n
    - margin: enforced separation between r and n
    - normalize: whether to normalize updated vectors

    Returns:
    - updated_doc_vecs: numpy array of shape (k, d) with modified vectors
    """
    d = query_vec.shape[1]

    # Get all unique document indices involved in the surgery
    involved_indices = sorted(set(n for n, _ in pairs) | set(r for _, r in pairs))
    delta_vars = {i: cp.Variable(d) for i in involved_indices}

    # Define constraints
    constraints = []
    for n, r in pairs:
        delta_r = delta_vars[r]
        delta_n = delta_vars[n]
        sim_r = query_vec @ (doc_vecs[r] + delta_r)
        sim_n = query_vec @ (doc_vecs[n] + delta_n)
        constraints.append(sim_r >= (sim_n + margin))

    # Objective: minimize the total squared changes for all involved vectors
    objective = cp.Minimize(cp.sum_squares(cp.vstack([delta_vars[i] for i in involved_indices])))
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP)
    except:
        print('Solver error')
        print(pairs)
        return doc_vecs, involved_indices, {}

    # Apply updates
    updated = doc_vecs.copy()
    for i in involved_indices:
        updated[i] += delta_vars[i].value
            
    deltas_serializable = {i: delta.value.tolist() for i, delta in delta_vars.items()}
    return updated, involved_indices, deltas_serializable
