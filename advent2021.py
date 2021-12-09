from collections import Counter
from copy import deepcopy
from math import prod


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


def day8():
    entries = []
    with open('8.txt') as f:
        for line in f.readlines():
            entries.append([[''.join(sorted(item)) for item in half.strip().split()] for half in line.split(' | ')])
    count = sum(len(display) in {2, 3, 4, 7} for entry in entries for display in entry[1])
    print(count)
    values = []
    for entry in entries:
        len5 = []
        len6 = []
        mapping = {}
        for code in entry[0]:
            items = set(list(code))
            segment_count = len(items)
            if segment_count == 2:
                mapping[1] = items
            elif segment_count == 3:
                mapping[7] = items
            elif segment_count == 4:
                mapping[4] = items
            elif segment_count == 7:
                mapping[8] = items
            elif segment_count == 5:
                len5.append(items)
            elif segment_count == 6:
                len6.append(items)
        for check in len5:
            if mapping[1].issubset(check):
                mapping[3] = check
                len5.remove(check)
                break
        mapping[9] = mapping[3] | mapping[4]
        len6.remove(mapping[9])
        bottom_left = (mapping[8] - (mapping[3] | mapping[4])).pop()
        for check in len6:
            if mapping[1].issubset(check):
                mapping[0] = check
            else:
                mapping[6] = check
        for check in len5:
            if bottom_left in check:
                mapping[2] = check
            else:
                mapping[5] = check
        rev_map = {''.join(sorted(v)): k for k, v in mapping.items()}
        values.append(int(''.join([str(rev_map[display]) for display in entry[1]])))
    print(sum(values))


def day9():
    def at(position):
        return entries[position[0]][position[1]]

    def in_bounds(position):
        i, j = position
        return 0 <= i < len(entries) and 0 <= j < len(entries[i])

    def adjacent(position):
        i, j = position
        return [pos for pos in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] if in_bounds(pos)]

    def build_basin(basin, start):
        basin.add(start)
        options = [option for option in adjacent(start) if at(start) < at(option) < 9]
        for p in options:
            if p not in basin:
                build_basin(basin, p)

    with open('9.txt') as f:
        entries = [[int(val) for val in line.strip()] for line in f.readlines()]
    safe_spots = []
    for i, row in enumerate(entries):
        for j, val in enumerate(row):
            current = (i, j)
            adj_heights = [at(pos) for pos in adjacent(current)]
            if val < min(adj_heights):
                safe_spots.append(current)
    print(sum(at(spot) + 1 for spot in safe_spots))
    basin_sizes = []
    for basin_start in safe_spots:
        current_basin = set()
        build_basin(current_basin, basin_start)
        basin_sizes.append(len(current_basin))
    print(prod(sorted(basin_sizes)[-3:]))


if __name__ == '__main__':
    day9()
