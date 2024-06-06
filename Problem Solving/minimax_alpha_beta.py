def game_over():
    # TODO: Implement your Stopping function here
    return True


def static_evaluation(position):
    # TODO: Implement your evaluation function here
    return 0


def children(position):
    # TODO: Generate your children state here
    children_list = []
    return children_list


def minimax(position, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or game_over() in position:
        return static_evaluation(position)

    if maximizingPlayer:
        max_eval = - float('inf')
        for child in children(position):
            evaluation = minimax(child, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for child in children(position):
            evaluation = minimax(child, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, evaluation)
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return min_eval


# initial call
current_position = [1]
minimax(current_position, 3, -float("inf"), float("inf"), True)