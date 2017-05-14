# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList =  currentGameState.getFood().asList()
        distToGhost = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        closestDisToGhost = min(distToGhost)
        distToFood = [util.manhattanDistance(newPos, foodPos) for foodPos in foodList]
        closestDisToFood = 0 if not distToFood else min(distToFood)
        return - closestDisToFood - (0 if closestDisToGhost >=2 else 1000000)

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        values = [self.minValue(gameState.generateSuccessor(0, move), 0, 1) for move in legalMoves]
        v = max(values)
        chosenIndex = [index for index in range(len(values)) if values[index] == v]
        return legalMoves[chosenIndex[0]]
        
    def maxValue(self, state, depth):
        legalMoves = state.getLegalActions(0)
        if (depth == self.depth or not legalMoves):
          return self.evaluationFunction(state)
        values = [self.minValue(state.generateSuccessor(0, move), depth, 1) for move in legalMoves]
        return max(values)

    def minValue(self, state, depth, ghostIndex):
        legalMoves = state.getLegalActions(ghostIndex)
        if not legalMoves:
          return self.evaluationFunction(state)
        values = []
        if (ghostIndex == state.getNumAgents() - 1):
          values = [self.maxValue(state.generateSuccessor(ghostIndex, move), depth + 1) for move in legalMoves]
        else:
          values = [self.minValue(state.generateSuccessor(ghostIndex, move), depth, ghostIndex + 1) for move in legalMoves]
        return min(values)
  
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        v = - 1000000
        optimalMove = ""
        alpha = - 1000000
        beta = 1000000
        for move in gameState.getLegalActions(0):
          successorState = gameState.generateSuccessor(0, move)
          newValue = self.minValue(successorState, 0, 1, alpha, beta)
          if (newValue > v):
            v = newValue
            optimalMove = move
          if (v > beta):
            return v
          alpha = max(alpha, v)      
        return optimalMove
        
    def maxValue(self, state, depth, alpha, beta):
        legalMoves = state.getLegalActions(0)
        if (depth == self.depth or not legalMoves):
          return self.evaluationFunction(state)
        v = - 1000000
        for move in legalMoves:
          successorState = state.generateSuccessor(0, move)
          v = max(v, self.minValue(successorState, depth, 1, alpha, beta))
          if (v > beta):
            return v
          alpha = max(alpha, v)
        #print "v: %f, depth: %d" % (v, depth)
        return v

    def minValue(self, state, depth, ghostIndex, alpha, beta):
        legalMoves = state.getLegalActions(ghostIndex)
        if not legalMoves:
          return self.evaluationFunction(state)
        v = 1000000
        for move in legalMoves:
          successorState = state.generateSuccessor(ghostIndex, move)
          if (ghostIndex == state.getNumAgents() - 1):
            v = min(v, self.maxValue(successorState, depth + 1, alpha, beta))
          else:
            v = min(v, self.minValue(successorState, depth, ghostIndex + 1, alpha, beta))
          if (v < alpha): 
            return v
          beta = min(beta, v)
          
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        values = [self.expValue(gameState.generateSuccessor(0, move), 0, 1) for move in legalMoves]
        v = max(values)
        chosenIndex = [index for index in range(len(values)) if values[index] == v]
        return legalMoves[chosenIndex[0]]
        
    def maxValue(self, state, depth):
        legalMoves = state.getLegalActions(0)
        if (depth == self.depth or not legalMoves):
          return self.evaluationFunction(state)
        values = [self.expValue(state.generateSuccessor(0, move), depth, 1) for move in legalMoves]
        return max(values)

    def expValue(self, state, depth, ghostIndex):
        legalMoves = state.getLegalActions(ghostIndex)
        if not legalMoves:
          return self.evaluationFunction(state)
        values = []
        if (ghostIndex == state.getNumAgents() - 1):
          values = [self.maxValue(state.generateSuccessor(ghostIndex, move), depth + 1) for move in legalMoves]
        else:
          values = [self.expValue(state.generateSuccessor(ghostIndex, move), depth, ghostIndex + 1) for move in legalMoves]
        return float(sum(values)) / len(values)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    foodList = currentGameState.getFood().asList()
    currentPos = currentGameState.getPacmanPosition()
    disToFood = [util.manhattanDistance(currentPos, foodPos) for foodPos in foodList]
    closestDisToFood = 0 if not disToFood else min(disToFood)
    #print "score: %d, closet: %d, len: %d" %(currentGameState.getScore(), closestDisToFood, len(disToFood))
    return currentGameState.getScore() * 2 - closestDisToFood - len(foodList)

# Abbreviation
better = betterEvaluationFunction

