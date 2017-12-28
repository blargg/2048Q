from Game.Model import emptyBoard, act, Action, isGameOver


def readAction(line):
    if line == "up":
        return Action.UP
    if line == "down":
        return Action.DOWN
    if line == "left":
        return Action.LEFT
    if line == "right":
        return Action.RIGHT
    return None


def runGame():
    """Runs a single game of 2048 on the command line"""
    board = emptyBoard()
    while True:
        print(board)
        if isGameOver(board):
            print("board is full")
            break

        line = input("which direction?")
        print(line)
        if line == "exit":
            break

        action = readAction(line)
        if action is None:
            print("Input not recognized, enter a direction")
        else:
            act(board, action)
    print("Game Over")
