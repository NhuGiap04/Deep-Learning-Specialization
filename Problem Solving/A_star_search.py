'''A* Algorithm
- A* is an informed search algorithm, or a best-first search, meaning that it is formulated in terms of weighted graphs:
starting from a specific starting node of a graph, it aims to find a path to the given goal node having the smallest
cost (least distance travelled, shortest time, etc.).
- It does this by maintaining a tree of paths originating at the start node and extending those paths one edge at a time
 until its termination criterion is satisfied.
- At each iteration of its main loop, A* needs to determine which of its paths to extend. It does so based on the cost
 of the path and an estimate of the cost required to extend the path all the way to the goal. Specifically, A* selects
 the path that minimizes f(n) = g(n) + h(n) where n is the next node on the path
*) g(n) : the cost to reach the node, h(n): heuristic function that estimates the cost of the cheapest path from n to
the goal
*) h(n) is admissible heuristic(never overestimates the cost to reach the goal)
*) h(n) is consistent if h(n) <= c(n, a, n') + h(n') (every successor n' of n generated by any action a)
'''

from operator import attrgetter


class Node:
    def __init__(self, parent=None, action=None):
        # TODO: Implement the problem abstraction: Define the state space, action lists
        self.state = None
        self.parent = parent
        self.action = action
        self.action_list = []
        self.g = 0
        self.h = 0
        self.f = 0

    def f_value(self):
        self.f = self.g + self.h

    def goal_test(self):
        # TODO: implement the goal test condition
        return True

    def child_node(self, parent, action):
        # TODO: implement the child node function(get the node's neighbor)
        return Node()

    def cost(self):
        # TODO: implement the cost for reaching from the node to its child
        return 1

    def heuristic(self):
        # TODO: implement the heuristic function
        return 2

    def solution(self):
        res_list = [self.action]
        p = self.parent
        while p:
            res_list.append(p.action)
            p = p.parent
        res = res_list[0:-1].copy()
        res.reverse()
        return res


# let openList equal empty list of nodes
open_list = []
# let closedList equal empty list of nodes
closed_list = []
# put startNode on the openList (leave it's f at zero)
start_node = Node()
open_list.append(start_node)
# while openList is not empty
#     let currentNode equal the node with the least f value
#     remove currentNode from the openList
#     add currentNode to the closedList
#     if currentNode is the goal
#         You've found the exit!
#     let children of the currentNode equal the adjacent nodes
#     for each child in the children
#         if child is in the closedList
#             continue to beginning of for loop
#         child.g = currentNode.g + distance b/w child and current
#         child.h = distance from child to end
#         child.f = child.g + child.h
#         if child.position is in the openList's nodes positions
#             if child.g is higher than the openList node's g
#                 continue to beginning of for loop
#         add the child to the openList
while len(open_list) != 0:
    current_node = min(open_list, key=attrgetter('f'))
    open_list.remove(current_node)
    closed_list.append(current_node)
    if current_node.goal_test():
        current_node.solution()
    children_nodes = []
    for i in range(len(current_node.action_list)):
        child = current_node.child_node(current_node, current_node.action_list[i])
    for child in children_nodes:
        if child in closed_list:
            continue
        child.g = current_node.g + current_node.cost()
        child.h = child.heuristic()
        child.f_value()
        if child.state in [node.state for node in open_list]:
            if child.g > max([node.g for node in open_list]):
                continue
        open_list.append(child)
    if len(open_list) == 0:
        print("NO SOLUTION FOUND")


