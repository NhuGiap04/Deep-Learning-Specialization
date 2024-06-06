'''Backtracking Algorithm
- At each step in Backtracking, we try to extend a given partial solution a = (a1, a2,..., ak) by adding another element
at the end.
- After this extension, we must test:
    . whether it is a complete solution
    . else, check whether the partial solution is still potentially extendable to some complete solution
'''


def backtracking_DFS(a, k):
    # if a = (a1, a2, ..., ak) is a solution then report it
    # else:
    #   k = k + 1
    #   Construct S_k, the set of candidates for position k of a
    #   while S_k is not empty:
    #       a_k = an element in S_k
    #       S_k = S_k - {a_k}
    #       if potentially_extendable(a) == True:
    #           backtracking_DFS(a, k)
    pass

