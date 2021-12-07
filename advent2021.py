from collections import Counter
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
    def transpose(matrix):
        return [list(t) for t in zip(*matrix)]

    def sum2d(matrix):
        return sum(sum(matrix, []))

    def winner():
        original = deepcopy(boards)
        transposed = [transpose(board) for board in original]
        for pick in picks:
            for board in [*original, *transposed]:
                for row in board:
                    if pick in row:
                        row.remove(pick)
                        if len(row) == 0:
                            return sum2d(board) * pick

    def loser():
        original = deepcopy(boards)
        transposed = [transpose(board) for board in original]
        n = len(boards)
        winners = set()
        for pick in picks:
            for i, board in enumerate([*boards, *transposed]):
                for row in board:
                    if pick in row:
                        row.remove(pick)
                        if len(row) == 0:
                            winners.add(i % n)
                            if len(winners) == n:
                                return sum2d(board) * pick

    with open('4.txt') as f:
        picks = [int(val) for val in f.readline().split(',')]
        f.readline()
        boards = [[[int(val) for val in row.split()] for row in board.split('\n')] for board in f.read().split('\n\n')]
    print(winner())
    print(loser())


def day5():
    def ordered_range(start, end):
        if start < end:
            return range(start, end + 1)
        elif start > end:
            return range(start, end - 1, -1)

    def discrete_points(x1, y1, x2, y2, allow_diags):
        x_range = ordered_range(x1, x2)
        y_range = ordered_range(y1, y2)
        if x1 == x2:
            return [(x1, y) for y in y_range]
        elif y1 == y2:
            return [(x, y1) for x in x_range]
        elif allow_diags:
            return [pair for pair in zip(x_range, y_range)]
        return []

    def count_unique_intersections(segments, allow_diags):
        discrete = Counter()
        for segment in segments:
            discrete.update(discrete_points(*segment, allow_diags))
        return sum(count > 1 for count in discrete.values())

    with open('5.txt') as f:
        entries = [[int(v) for p in line.strip().split(' -> ') for v in p.split(',')] for line in f.readlines()]
    print(count_unique_intersections(entries, False))
    print(count_unique_intersections(entries, True))


def day6():
    def simulate_day_fast(fish_counts):
        updated_counts = Counter()
        for k, v in fish_counts.items():
            if k == 0:
                updated_counts[6] += v
                updated_counts[8] += v
            else:
                updated_counts[k - 1] += v
        return updated_counts

    with open('6.txt') as f:
        entries = [int(val) for val in f.read().strip().split(',')]
    counts = Counter(entries)
    for i in range(80):
        counts = simulate_day_fast(counts)
    print(sum(subtotal for subtotal in counts.values()))
    counts = Counter(entries)
    for i in range(256):
        counts = simulate_day_fast(counts)
    print(sum(subtotal for subtotal in counts.values()))


def day7():
    def distance_flat(a, b):
        return abs(a - b)

    def distance_growing(a, b):
        d = distance_flat(a, b)
        return int(d * (d + 1) / 2)

    def closest(values, metric):
        return min(sum(metric(value, i) for value in values) for i in range(min(values), max(values) + 1))

    with open('7.txt') as f:
        entries = [int(val) for val in f.read().strip().split(',')]
    print(closest(entries, distance_flat))
    print(closest(entries, distance_growing))


if __name__ == '__main__':
    day7()
