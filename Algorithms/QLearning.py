# Q-learning is a form of reinforcment learning. It learns a policy to collect
# The highest reward.
#
# For some set of states and some set of actions, we learn a quality function.
#   Q: State X Action -> Real Numbers.
#   For a given state, the policy is to take the action with the highest
#   quality
#
# The reward is determined by the state
#   reward: State X Action -> Real Number
# Transitions define how actions lead from one state to another
#   transition: State X Action -> State

import pickle
import collections
from util.data import do_nothing


class Observation:
    """A single observation of a reinforcement learning task"""
    def __init__(self, startState, action, nextState, reward):
        self.startState = startState
        self.action = action
        self.nextState = nextState
        self.reward = reward


class ReinforcementTask:
    """Defines a reinforcement learning task. This is an abstract interface to
    be implemented"""
    def transition(state, action):
        raise NotImplementedError("Function not implemented")

    def startState():
        raise NotImplementedError("Function not implemented")

    def isEndState(state):
        raise NotImplementedError("Function not implemented")

    def stateEq(state1, state2):
        raise NotImplementedError("Function not implemented")


class ReinforcementLearner:
    """Defines a learning algorithm that can be trained on reinformcement
    tasks"""
    def chooseAction(self, state):
        raise NotImplementedError("Fucntion not implemented")

    def observeResult(self, state, action, nextState, reward):
        raise NotImplementedError("Function not implemented")


StepResult = collections.namedtuple('StepResult', ['action', 'reward'])


def RunStep(learner, task, state, train=True):
    chosenAction = learner.chooseAction(state)
    startState = state.copy()
    reward = task.transition(state, chosenAction)
    if train:
        learner.observeResult(startState, chosenAction, state, reward)
    return StepResult(action=chosenAction,
                      reward=reward)


EpisodeResult = collections.namedtuple('EpisodeResult',
                                       ['totalReward',
                                        'endState'])


def RunEpisode(learner, task, logState=do_nothing, train=True):
    state = task.startState()
    totalReward = 0
    while not task.isEndState(state):
        copyState = state.copy()
        action = RunStep(learner, task, state, train=train)
        logState(copyState, action)

        # quit if the learner stalls
        if task.stateEq(copyState, state):
            break

    logState(state, None)
    return EpisodeResult(totalReward=totalReward,
                         endState=state)


class Episode:
    def __init__(self, states, actions, rewards, finalState=None):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.finalState = finalState

    def processedRewards(self, discountFactor):
        """Processes the rewards, so that previous (state, action) values get
        reward for future results"""
        assert discountFactor >= 0 and discountFactor <= 1,\
            "discountFactor must be between 0 and 1"
        processedRewards = collections.deque()

        pReward = 0

        for actual_reward in self.rewards[::-1]:
            pReward = actual_reward + discountFactor * pReward
            processedRewards.appendleft(pReward)

        return processedRewards


def SimulateEpisode(learner, task):
    """Runs an episode to completion, returns the states, chosen actions and
    rewards"""
    state = task.startState()
    rewards = []
    states = []
    actions = []
    while not task.isEndState(state):
        curState = state.copy()
        action = learner.chooseAction(state)
        reward = task.transition(state, action)
        actions.append(action)
        states.append(curState)
        rewards.append(reward)

    return Episode(states=states,
                   actions=actions,
                   rewards=rewards,
                   finalState=state)


def RunEpisodeSaveFile(learner, task, filename, train=True):
    episodeRecord = []

    def logState(state, action):
        episodeRecord.append((state, action))

    RunEpisode(learner, task, logState, train=train)

    with open(filename, "wb") as f:
        pickle.dump(episodeRecord, f)  # TODO concider a safter alternative
