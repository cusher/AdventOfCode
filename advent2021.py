from copy import deepcopy


def day1():
    with open('1.txt') as f:
        nums = [int(val) for val in f.readlines()]
    counter = 0
    for i in range(len(nums) - 1):
        if nums[i] < nums[i + 1]:
            counter += 1
    print(counter)
    counter = 0
    for i in range(len(nums) - 3):
        if nums[i] < nums[i + 3]:
            counter += 1
    print(counter)


def day2():
    with open('2.txt') as f:
        entries = [line.strip().split(' ') for line in f.readlines()]
    x = 0
    d = 0
    for entry in entries:
        k, v = (entry[0], int(entry[1]))
        if k == 'forward':
            x += v
        elif k == 'up':
            d -= v
        elif k == 'down':
            d += v
    print(x * d)
    aim = 0
    x = 0
    d = 0
    for entry in entries:
        k, v = (entry[0], int(entry[1]))
        if k == 'forward':
            x += v
            d += aim * v
        elif k == 'up':
            aim -= v
        elif k == 'down':
            aim += v
    print(x * d)


def day3():
    def common_bit(position, values, most):
        count = sum(v[position] == '1' for v in values)
        return str(int((count >= len(values) / 2) == most))

    with open('3.txt') as f:
        entries = [line.strip() for line in f.readlines()]
    n = len(entries[0])
    ratings = [''.join([common_bit(i, entries, bool(option)) for i in range(n)]) for option in range(2)]
    print(int(ratings[0], 2) * int(ratings[1], 2))
    for option in range(2):
        matches = list(entries)
        for i in range(n):
            cb = common_bit(i, matches, bool(option))
            matches = list(filter(lambda entry: entry[i] == cb, matches))
            if len(matches) == 1:
                ratings[option] = matches[0]
                break
    print(int(ratings[0], 2) * int(ratings[1], 2))


def day4():
    def picker(boards):
        for pick in picks:
            for board in [*boards, *transposed]:
                for row in board:
                    if pick in row:
                        row.remove(pick)
                    if len(row) == 0:
                        total = sum(sum(board, [])) * pick
                        return total

    def loser(boards):
        n = len(boards)
        winners = [i for i in range(n)]
        for pick in picks:
            for i, board in enumerate([*boards, *transposed]):
                for row in board:
                    if pick in row:
                        row.remove(pick)
                        if len(row) == 0:
                            if i % n in winners:
                                winners.remove(i % n)
                                if len(winners) == 0:
                                    return sum(sum(boards[i % n], [])) * pick
    boards = []
    with open('4.txt') as f:
        picks = list(map(int, f.readline().split(',')))
        board = []
        for line in f:
            g = list(map(int, line.strip().split()))
            if len(g) == 0:
                if len(board) > 0:
                    boards.append(board)
                    board = []
            else:
                board.append(g)
        if len(board) > 0:
            boards.append(board)
    transposed = [list(map(list, zip(*board))) for board in boards]
    print(picker(deepcopy(boards)))
    print(loser(deepcopy(boards)))


if __name__ == '__main__':
    day4()
