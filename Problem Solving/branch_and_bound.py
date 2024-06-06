'''Branch and Bound Algorithm
1. Using a heuristic, find a solution x_h to the optimization problem. Store its value B = f(x_h) (if no heuristic
available, set B = infinity). B denotes the best solution so far, and will be used as an upper bound on candidate
solutions
2. Initialize a queue to hold a partial solution with none of the variables of the problem assigned
3. Loop until the queue is empty
    1. Take a Node N off the queue
    2. If N represents a single candidate solution x and f(x) < B, then x is the best solution so far. Record it
        and set B = f(x)
    3. Else, branch on N to produce new nodes N_i. For each of these:
        1. If bound(N_i) > B, do nothing
        2. Else, store N_i on the queue'''

# TODO: Implement the following functions and classes
class Node:
    def __init__(self):
        self.candidate_nodes = []
        pass

    def represent_single_candidate(self):
        pass

    def candidate(self):
        pass


def heuristic_solve(problem):
    pass


def populate_candidates(problem):
    pass


def objective_function(candidate):
    pass

def lower_bound_function(candidate):
    pass


# assuming the objective function f is to be minimized
def branch_and_bound_solve(problem, objective_function, lower_bound_function):
    problem_upper_bound = float('inf')
    heuristic_solution = heuristic_solve(problem)  # x.h
    problem_upper_bound = objective_function(heuristic_solution)  # B = f(x_h)
    current_optimum = heuristic_solution

    # Step 2
    # Implement Queue for Candidate Solutions
    candidate_queue = []
    # problem-specific queue in initialization
    candidate_queue = populate_candidates(problem)
    while len(candidate_queue) > 0:
        # Step 3
            # Step 3.1
        node = candidate_queue.pop(0)
        # node represents N above
        if node.represent_single_candidate():
            # Step 3.2
            if objective_function(node.candidate()) < problem_upper_bound:
                current_optimum = node.candidate()
                problem_upper_bound = objective_function(current_optimum)
        else:
            # Step 3.3
            for child_branch in node.candidates_nodes:
                if lower_bound_function(child_branch) <= problem_upper_bound:
                    candidate_queue.append(child_branch)
                # otherwise, bound(N_i) > B, so we prune the branch
    return current_optimum

