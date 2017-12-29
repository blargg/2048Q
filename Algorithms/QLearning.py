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


class ReinforcementTask:
    """Defines a reinforcement learning task. This is an abstract interface to
    be implemented"""
    def transition(state, action):
        raise NotImplementedError("Function not implemented")

    def reward(state, action):
        raise NotImplementedError("Function not implemented")

    def startState():
        raise NotImplementedError("Function not implemented")


class ReinforcementLearner:
    """Defines a learning algorithm that can be trained on reinformcement
    tasks"""
    def choseAction(state):
        raise NotImplementedError("Fucntion not implemented")

    def observeResult(previousState, action, nextState, reward):
        raise NotImplementedError("Function not implemented")


def qLearnStep(learner, task, state):
    pass
