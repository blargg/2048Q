
def dictStringArray(d):
    return ["({}, {})".format(key, d[key]) for key in d]


def printDict(d):
    for s in dictStringArray(d):
        print(s)


def do_nothing(*kwargs):
    pass
