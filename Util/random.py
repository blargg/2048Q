import random


def bernoulli(p):
    """Generates a random variable based on a bernoulli distribution
    parameterized by p.
    returns: 1 with probability p
             0 with probability (1 - p)"""
    assert 0 <= p, "p must be positive"
    assert p <= 1, "p must be less than 1"
    if random.random() < p:
        return 1
    else:
        return 0
