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
from turtledemo.penrose import inflatedart

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        #print(newPos)
        #print(newFood)
        #print(newScaredTimes)
        #print(str(newGhostStates))
        newFoodLocations = newFood.asList()
        if len(newFoodLocations) == 0:
            return 1
        distancesToFood = []
        for location in newFoodLocations:
            distancesToFood.append(manhattanDistance(newPos, location))
        distanceToFood = min(distancesToFood)
        ghostPositions = successorGameState.getGhostPositions()
        result = 0
        for position in ghostPositions:
            distance = manhattanDistance(newPos, position)
            if (distance == 0 or distance == 1):
                return -1
        return result + (1/distanceToFood) + (currentGameState.getNumFood() - successorGameState.getNumFood())

        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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
        self.numberofagents = 0

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def minimum(self, gameState: GameState, currentAgent, currentDepth, depthLimitReached):
        v = float('inf')
        actions = gameState.getLegalActions(currentAgent)
        actionvalue = {}
        actionvalue[v] = actions[0]
        newAgent, newDepth = self.getCurrentAgentAndDepth((currentAgent + 1), currentDepth)
        for action in actions:
            successor = gameState.generateSuccessor(currentAgent, action)
            if (depthLimitReached):
                vtemp = self.evaluationFunction(successor)
            else:
                vtemp, bestaction = self.value(successor, newAgent, newDepth)
            v = min(v, vtemp)
            actionvalue[vtemp] = action
        return (v, actionvalue[v])

    def maximum(self, gameState: GameState, currentAgent, currentDepth):
        v = float('-inf')
        actions = gameState.getLegalActions(currentAgent)
        actionvalue = {}
        actionvalue[v] = actions[0]
        newAgent, newDepth = self.getCurrentAgentAndDepth((currentAgent+1), currentDepth)
        for action in actions:
            successor = gameState.generateSuccessor(currentAgent, action)
            vtemp, bestaction = self.value(successor, newAgent, newDepth)
            v = max(v, vtemp)
            actionvalue[vtemp] = action
        return (v, actionvalue[v])

    def value(self, gameState: GameState, currentAgent, currentDepth):
        depthLimitReached = False
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""
        if (currentAgent == self.numberofagents - 1):
            if (currentDepth == self.depth):
                actions = gameState.getLegalActions()
                if len(actions) == 0:
                    return self.evaluationFunction(gameState), ""
                else:
                    depthLimitReached = True
        if currentAgent == 0:
            return self.maximum(gameState, currentAgent, currentDepth)
        else:
            return self.minimum(gameState, currentAgent, currentDepth, depthLimitReached)

    def getCurrentAgentAndDepth(self, agent, depth):
        if (agent > self.numberofagents - 1):
            agent = 0
            depth = depth + 1
        return agent, depth

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self.numberofagents = gameState.getNumAgents()
        (value, action) = self.value(gameState, 0, 1)
        return action


        #util.raiseNotDefined()






class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def minimum(self, gameState: GameState, currentAgent, currentDepth, depthLimitReached, alpha, beta):
        v = float('inf')
        actions = gameState.getLegalActions(currentAgent)
        actionvalue = {}
        actionvalue[v] = actions[0]
        newAgent, newDepth = self.getCurrentAgentAndDepth((currentAgent + 1), currentDepth)
        for action in actions:
            successor = gameState.generateSuccessor(currentAgent, action)
            if (depthLimitReached):
                vtemp = self.evaluationFunction(successor)
            else:
                vtemp, bestaction = self.value(successor, newAgent, newDepth, alpha, beta)
            v = min(v, vtemp)
            actionvalue[vtemp] = action
            if (v < alpha):
                return (v, actionvalue[v])
            beta = min(beta, v)
        return (v, actionvalue[v])

    def maximum(self, gameState: GameState, currentAgent, currentDepth, alpha, beta):
        v = float('-inf')
        actions = gameState.getLegalActions(currentAgent)
        actionvalue = {}
        actionvalue[v] = actions[0]
        newAgent, newDepth = self.getCurrentAgentAndDepth((currentAgent+1), currentDepth)
        for action in actions:
            successor = gameState.generateSuccessor(currentAgent, action)
            vtemp, bestaction = self.value(successor, newAgent, newDepth, alpha, beta)
            v = max(v, vtemp)
            actionvalue[vtemp] = action
            if (v > beta):
                return (v, actionvalue[v])
            alpha = max(alpha, v)
        return (v, actionvalue[v])

    def value(self, gameState: GameState, currentAgent, currentDepth, alpha, beta):
        depthLimitReached = False
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""
        if (currentAgent == self.numberofagents - 1):
            if (currentDepth == self.depth):
                actions = gameState.getLegalActions()
                if len(actions) == 0:
                    v = self.evaluationFunction(gameState)
                    #self.beta = min(self.beta, v)
                    return v, ""
                else:
                    depthLimitReached = True
        if currentAgent == 0:
            return self.maximum(gameState, currentAgent, currentDepth, alpha, beta)
        else:
            return self.minimum(gameState, currentAgent, currentDepth, depthLimitReached, alpha, beta)

    def getCurrentAgentAndDepth(self, agent, depth):
        if (agent > self.numberofagents - 1):
            agent = 0
            depth = depth + 1
        return agent, depth

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.numberofagents = gameState.getNumAgents()
        alpha = float('-inf')
        beta = float('inf')
        (value, action) = self.value(gameState, 0, 1, alpha, beta)
        return action
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectation(self, gameState: GameState, currentAgent, currentDepth, depthLimitReached):
        v = 0
        actions = gameState.getLegalActions(currentAgent)
        #actionvalue = {}
        #actionvalue[v] = actions[0]
        probability = 1.0 / len(actions)
        newAgent, newDepth = self.getCurrentAgentAndDepth((currentAgent + 1), currentDepth)
        for action in actions:
            successor = gameState.generateSuccessor(currentAgent, action)
            if (depthLimitReached):
                vtemp = self.evaluationFunction(successor)
            else:
                vtemp, bestaction = self.value(successor, newAgent, newDepth)
            v = v + (probability * vtemp)
            #actionvalue[vtemp] = action
        #return (v, actionvalue[v])
        return (v, "")

    def maximum(self, gameState: GameState, currentAgent, currentDepth):
        v = float('-inf')
        actions = gameState.getLegalActions(currentAgent)
        actionvalue = {}
        actionvalue[v] = actions[0]
        newAgent, newDepth = self.getCurrentAgentAndDepth((currentAgent+1), currentDepth)
        for action in actions:
            successor = gameState.generateSuccessor(currentAgent, action)
            vtemp, bestaction = self.value(successor, newAgent, newDepth)
            v = max(v, vtemp)
            actionvalue[vtemp] = action
        return (v, actionvalue[v])

    def value(self, gameState: GameState, currentAgent, currentDepth):
        depthLimitReached = False
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""
        if (currentAgent == self.numberofagents - 1):
            if (currentDepth == self.depth):
                actions = gameState.getLegalActions()
                if len(actions) == 0:
                    return self.evaluationFunction(gameState), ""
                else:
                    depthLimitReached = True
        if currentAgent == 0:
            return self.maximum(gameState, currentAgent, currentDepth)
        else:
            return self.expectation(gameState, currentAgent, currentDepth, depthLimitReached)

    def getCurrentAgentAndDepth(self, agent, depth):
        if (agent > self.numberofagents - 1):
            agent = 0
            depth = depth + 1
        return agent, depth

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.numberofagents = gameState.getNumAgents()
        (value, action) = self.value(gameState, 0, 1)
        return action
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    "*** YOUR CODE HERE ***"
    FoodLocations = Food.asList()
    if len(FoodLocations) == 0:
        return float('inf')
    distancesToFood = []
    for location in FoodLocations:
        distancesToFood.append(manhattanDistance(Pos, location))
    distanceToFood = min(distancesToFood)
    ghostPositions = currentGameState.getGhostPositions()
    result = 0
    distanceToGhost = 0
    for position in ghostPositions:
        distanceToGhost = manhattanDistance(Pos, position)
        if (distanceToGhost <= 0.7):
            #return float('-inf')
            if (ScaredTimes[0] > 0):
                print('a')
                return float('inf')
            else:
                print('b')
                return float('-inf')
    capsuleLocations = currentGameState.getCapsules()
    distancesToCapsules = []
    for location in capsuleLocations:
        distancesToCapsules.append(manhattanDistance(Pos, location))
    if (len(distancesToCapsules) <= 1):
        pass

    else:
        distanceToCapsule = min(distancesToCapsules)
    if ((ScaredTimes[0] > 0) and (distanceToGhost < ScaredTimes[0])):
        #result = currentGameState.getScore()
        result = 1 / (distanceToGhost)
    if (len(distancesToCapsules) <= 2 and len(distancesToCapsules) > 0):
        result = (1 / distanceToCapsule)
    else:
        result =  (1 / distanceToFood) - currentGameState.getNumFood()
    #return result + 2 * (1 / distanceToFood) + 8 * (1 / distanceToCapsule) - currentGameState.getNumFood() - len(capsuleLocations)
    #return result + (1 / distanceToFood) - currentGameState.getNumFood()
    print(result)
    return result
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
