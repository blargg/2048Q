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


class ReinforcementTask:
    """Defines a reinforcement learning task. This is an abstract interface to
    be implemented"""
    def transition(state, action):
        raise NotImplementedError("Function not implemented")

    def reward(state, action):
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


def LearnStep(learner, task, state):
    chosenAction = learner.chooseAction(state)
    startState = state.copy()
    task.transition(state, chosenAction)
    reward = task.reward(startState, chosenAction)
    learner.observeResult(startState, chosenAction, state, reward)
    return StepResult(action=chosenAction,
                      reward=reward)


EpisodeResult = collections.namedtuple('EpisodeResult',
                                       ['totalReward',
                                        'endState'])


def LearnEpisode(learner, task, logState=do_nothing):
    state = task.startState()
    totalReward = 0
    while not task.isEndState(state):
        copyState = state.copy()
        action = LearnStep(learner, task, state)
        logState(copyState, action)

        # quit if the learner stalls
        if task.stateEq(copyState, state):
            break

    logState(state, None)
    return EpisodeResult(totalReward=totalReward,
                         endState=state)


def LearnEpisodeSaveFile(learner, task, filename):
    episodeRecord = []

    def logState(state, action):
        episodeRecord.append((state, action))

    LearnEpisode(learner, task, logState)

    with open(filename, "wb") as f:
        pickle.dump(episodeRecord, f)  # TODO concider a safter alternative
