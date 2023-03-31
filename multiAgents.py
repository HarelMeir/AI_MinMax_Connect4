"""
Introduction to Artificial Intelligence, 89570, Bar Ilan University, ISRAEL

Student name: Harel Meir
Student ID: 205588940

"""

# multiAgents.py
# --------------
# Attribution Information: part of the code were created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# http://ai.berkeley.edu.
# We thank them for that! :)


import random, util, math

import gameUtil
from connect4 import Agent


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 1  # agent is always index 1
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class BestRandom(MultiAgentSearchAgent):

    def getAction(self, gameState):
        return gameState.pick_best_move()


class MinimaxAgent(MultiAgentSearchAgent):
    def get_minimax_val(self, gameState, depth):
        best_action = ""
        # if its a goal state, or the depth is 0 -> returns the heuristic eval.
        if gameState.is_terminal() or depth == 0:
            return self.evaluationFunction(gameState), best_action

        # getting the turn from the gameState
        turn = gameState.turn

        # getting the successors of the current node.
        actions = gameState.getLegalActions(turn)
        succs = [(action, gameState.generateSuccessor(self.index, action)) for action in actions]

        # lowering the depth var.
        depth -= 1

        # if its the turn of the agent, he is the max player.
        if turn == self.index:
            # init cur_max to -inf.
            cur_max = float('-inf')

            for action, succ in succs:
                # switching the turn
                succ.switch_turn(succ.turn)

                # getting the value of the minimax recursively
                v, _ = self.get_minimax_val(succ, depth)

                # getting the max which is the
                if cur_max < v:
                    cur_max, best_action = v, action
                cur_max = max(v, cur_max)
            return cur_max, best_action

        # if its the turn of the player, he is the min player.
        else:
            # setting the cur_min for the max value.
            cur_min = float('inf')

            for action, succ in succs:
                # switching the turn.
                succ.switch_turn(succ.turn)

                # getting the value
                v, _ = self.get_minimax_val(succ, depth)
                if cur_min > v:
                    cur_min, best_action = v, action
            return cur_min, best_action

    def getAction(self, gameState):
        v, best_action = self.get_minimax_val(gameState, self.depth)
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    # a method for max value check.
    def max_val(self,succs,depth,a, b):
        cur_max = float('-inf')

        for action, succ in succs:
            # switching the turn.
            succ.switch_turn(succ.turn)

            # getting the max value.
            cur_max = max(cur_max, self.a_b_puring(succ, depth, a, b))

            # if the max is bigger than b - we can cut!
            if cur_max > b:
                break
            a = max(a, cur_max)
        return cur_max


    def min_val(self,succs,depth, a , b):
        cur_min = float('inf')

        for action, succ in succs:
            # switching the turn.
            succ.switch_turn(succ.turn)

            cur_min = min(cur_min, self.a_b_puring(succ, depth, a, b))

            # if the value is smaller than a -> we can cut!
            if cur_min < a:
                break
            b = min(b, cur_min)
        return cur_min



    def a_b_puring(self, gameState,depth, a, b):
        # if its a goal state, or the depth is 0 -> returns the heuristic eval.
        if gameState.is_terminal() or depth == 0:
            return self.evaluationFunction(gameState)

        # getting the turn
        turn = gameState.turn

        # getting the successors of the current node.
        actions = gameState.getLegalActions(self.index)
        succs = [(action, gameState.generateSuccessor(self.index, action)) for action in actions]

        # decreasing the depth.
        depth -= 1

        # if its the agent turn - he is the max player. otherwise, min player.
        if turn == self.index:
            return self.max_val(succs,depth, a, b)

        else:
            return self.min_val(succs,depth,a, b)

    def getAction(self, gameState):
        """
            Your minimax agent with alpha-beta pruning (question 2)
        """
        a = float('-inf')
        b = float('inf')
        best_value = float('-inf')
        best_action = 0  # the action that leads to max value

        # getting the succs.
        actions = gameState.getLegalActions(self.index)
        succs = [(action, gameState.generateSuccessor(self.index, action)) for action in actions]

        depth = self.depth - 1
        for action, succ in succs:
            # switching the turn.
            succ.switch_turn(succ.turn)

            # getting the val.
            v = self.a_b_puring(succ, depth, a, b)

            if v > best_value:
                best_value,best_action = v , action

        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    # max value method from the pesuedo code.
    def max_val(self,succs,depth,best_action):
        # init cur_max to -inf.
        final_v = float('-inf')

        for action, succ in succs:
            succ.switch_turn(succ.turn)
            v, _ = self.expecti_max(succ, depth)
            if final_v < v:
                final_v, best_action = v, action
        return final_v, best_action

    # exp value method from the pesudo code.
    def exp_val(self,succs,depth, size_succs):
        # init the value to 0.
        final_v = 0
        # uniform according to the instructions.
        p = 1 / size_succs

        # looping over the successors to get the value.
        for action, succ in succs:
            succ.switch_turn(succ.turn)
            v, _ = self.expecti_max(succ, depth)
            final_v += v
        final_v *= p
        return final_v, action

    # expectimax main method.
    def expecti_max(self, gameState, depth):
        best_action = ""
        # if its a goal state, or the depth is 0 -> returns the heuristic eval.
        if gameState.is_terminal() or depth == 0:
            return self.evaluationFunction(gameState), best_action

        # getting the turn.
        turn = gameState.turn

        # getting all the successors of the current node.
        actions = gameState.getLegalActions(turn)
        succs = [(action, gameState.generateSuccessor(self.index, action)) for action in actions]

        # getting the number of successors.
        size_succs = len(succs)

        # decreasing the depth by 1.
        depth -= 1


        if turn == self.index:
            cur_max, best_action = self.max_val(succs,depth, best_action)
            return cur_max, best_action
        else:
            final_v, action = self.exp_val(succs,depth, size_succs)
            return final_v, action


    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        value, best_action = self.expecti_max(gameState, self.depth)
        return best_action
