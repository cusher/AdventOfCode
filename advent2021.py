from collections import Counter, defaultdict
from copy import copy, deepcopy
from functools import reduce
from heapq import heappop, heappush
from itertools import count, takewhile
from math import inf, prod


class Matrix:
    def __init__(self, data):
        self.data = data

    def __copy__(self):
        return Matrix([row.copy() for row in self.data])

    def __getitem__(self, key):
        return self.data[key[0]][key[1]]

    def __setitem__(self, key, value):
        self.data[key[0]][key[1]] = value

    def __str__(self):
        return '\n'.join(' '.join([str(item) for item in row]) for row in self.data)

    @classmethod
    def from_string(cls, string, delimiter=None):
        if delimiter == '':
            return cls([[int(val) for val in line] for line in string.split('\n')])
        # No delimiter = split on whitespace
        return cls([[int(val) for val in line.split(delimiter)] for line in string.split('\n')])

    def indexes(self):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                yield i, j

    def in_bounds(self, position):
        i, j = position
        return 0 <= i < len(self.data) and 0 <= j < len(self.data[i])

    def transpose(self):
        return Matrix([list(t) for t in zip(*self.data)])

    def sum(self):
        return sum(sum(self.data, []))

    def adjacent(self, position, diagonals=False):
        i, j = position
        options = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        if diagonals:
            options.extend([(i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)])
        return [position for position in options if self.in_bounds(position)]


# Adapted from https://docs.python.org/3.7/library/heapq.html#priority-queue-implementation-notes
class PriorityQueue:
    REMOVED = '<removed-task>'  # placeholder for a removed task

    def __init__(self, initial_data=None):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.counter = count()  # unique sequence count
        if initial_data:
            for k, v in initial_data.items():
                self.add_task(k, v)

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = PriorityQueue.REMOVED

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not PriorityQueue.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')


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
    def winner():
        original = deepcopy(boards)
        transposed = [board.transpose() for board in original]
        for pick in picks:
            for board in [*original, *transposed]:
                for row in board.data:
                    if pick in row:
                        row.remove(pick)
                        if len(row) == 0:
                            return board.sum() * pick

    def loser():
        original = deepcopy(boards)
        transposed = [board.transpose() for board in original]
        n = len(boards)
        winners = set()
        for pick in picks:
            for i, board in enumerate([*boards, *transposed]):
                for row in board.data:
                    if pick in row:
                        row.remove(pick)
                        if len(row) == 0:
                            winners.add(i % n)
                            if len(winners) == n:
                                return board.sum() * pick

    with open('4.txt') as f:
        picks = [int(val) for val in f.readline().split(',')]
        f.readline()
        boards = [Matrix.from_string(board) for board in f.read().split('\n\n')]
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
    def build_basin(basin, start):
        basin.add(start)
        options = [option for option in data.adjacent(start) if data[start] < data[option] < 9]
        for p in options:
            if p not in basin:
                build_basin(basin, p)

    with open('9.txt') as f:
        data = Matrix.from_string(f.read(), '')
    safe_spots = []
    for current in data.indexes():
        adj_heights = [data[pos] for pos in data.adjacent(current)]
        if data[current] < min(adj_heights):
            safe_spots.append(current)
    print(sum(data[spot] + 1 for spot in safe_spots))
    basin_sizes = []
    for basin_start in safe_spots:
        current_basin = set()
        build_basin(current_basin, basin_start)
        basin_sizes.append(len(current_basin))
    print(prod(sorted(basin_sizes)[-3:]))


def day10():
    def parse(stack, remainder):
        opposite_map = {
            '(': ')',
            '[': ']',
            '{': '}',
            '<': '>',
        }
        while remainder:
            test = remainder.pop(0)
            if test in ['(', '{', '[', '<']:
                stack.append(test)
                return parse(stack, remainder)
            elif test != opposite_map[stack[-1]]:
                raise Exception(test)
            stack.pop()
        return [opposite_map[char] for char in reversed(stack)]

    with open('10.txt') as f:
        entries = [line.strip() for line in f.readlines()]
    illegal_map = {
        ')': 3,
        ']': 57,
        '}': 1197,
        '>': 25137,
    }
    end_map = {
        ')': 1,
        ']': 2,
        '}': 3,
        '>': 4,
    }
    illegal_score = 0
    end_scores = []
    for line in entries:
        try:
            ending = parse([], list(line))
            end_scores.append(reduce(lambda score, char: score * 5 + end_map[char], ending, 0))
        except Exception as e:
            illegal_score += illegal_map[e.args[0]]
    print(illegal_score)
    print(sorted(end_scores)[int(len(end_scores) / 2)])


def day11():
    def simulate(matrix):
        flashers = set()
        recheck = set()
        for position in matrix.indexes():
            matrix[position] += 1
            recheck.add(position)
        while recheck:
            checking = recheck.pop()
            if matrix[checking] > 9 and checking not in flashers:
                flashers.add(checking)
                for adj in matrix.adjacent(checking, True):
                    matrix[adj] += 1
                    recheck.add(adj)
        for flasher in flashers:
            matrix[flasher] = 0
        return flashers

    with open('11.txt') as f:
        data = Matrix.from_string(f.read(), '')
    data_copy = copy(data)
    print(sum(len(simulate(data_copy)) for _ in range(100)))
    i = 0
    flashed = set()
    while len(flashed) != 100:
        flashed = simulate(data)
        i += 1
    print(i)


def day12():
    def traverse(graph, completed, path, allow_twice=False, used_twice=False):
        current = path[-1]
        if current == 'end':
            completed.append(path)
        else:
            options = graph[current]
            for option in options:
                if option.isupper() or option not in path:
                    traverse(graph, completed, path + [option], allow_twice, used_twice)
                elif allow_twice and not used_twice and option != 'start':
                    traverse(graph, completed, path + [option], allow_twice, True)

    with open('12.txt') as f:
        entries = [line.strip().split('-') for line in f.readlines()]
    network = defaultdict(list)
    for entry in entries:
        network[entry[0]].append(entry[1])
        network[entry[1]].append(entry[0])
    paths1 = []
    traverse(network, paths1, ['start'])
    print(len(paths1))
    paths2 = []
    traverse(network, paths2, ['start'], True)
    print(len(paths2))


def day13():
    def do_fold(paper, instruction):
        fold_at = int(instruction[1])
        coord_index = 0 if instruction[0] == 'x' else 1
        for dot in paper:
            if dot[coord_index] > fold_at:
                dot[coord_index] = 2 * fold_at - dot[coord_index]

    with open('13.txt') as f:
        dots = [[int(val) for val in line.strip().split(',')] for line in takewhile(lambda l: l != '\n', f)]
        folds = [line.strip().strip('fold along ').split('=') for line in f]
    do_fold(dots, folds[0])
    print(len(set((dot[0], dot[1]) for dot in dots)))
    for fold in folds[1:]:
        do_fold(dots, fold)
    dot_set = set((dot[0], dot[1]) for dot in dots)
    x_range, y_range = max(x for x, y in dots) + 1, max(y for x, y in dots) + 1
    for y in range(y_range):
        print(''.join('#' if (x, y) in dot_set else ' ' for x in range(x_range)))


def day14():
    def fast_pass(pair_counts, mapping, char_counts):
        temp = Counter(pair_counts)
        for pair, num in pair_counts.items():
            if pair in mapping and num > 0:
                char_counts[mapping[pair]] += num
                temp[pair[0] + mapping[pair]] += num
                temp[mapping[pair] + pair[1]] += num
                temp[pair] -= num
        return temp

    with open('14.txt') as f:
        start = f.readline().strip()
        f.readline()
        entries = dict(line.strip().split(' -> ') for line in f.readlines())
    chars = Counter(start)
    pairs = Counter()
    for i in range(len(start) - 1):
        pairs[start[i] + start[i + 1]] += 1
    for i in range(10):
        pairs = fast_pass(pairs, entries, chars)
    print(chars.most_common()[0][1] - chars.most_common()[-1][1])
    for i in range(30):
        pairs = fast_pass(pairs, entries, chars)
    print(chars.most_common()[0][1] - chars.most_common()[-1][1])


def day15():
    def dijkstra(graph, start_node, end_node):
        dist = {node: inf for node in graph.indexes()}
        dist[start_node] = 0
        unvisited = PriorityQueue(dist)
        while unvisited:
            visiting = unvisited.pop_task()
            if visiting == end_node:
                return dist[end_node]
            for neighbor in graph.adjacent(visiting):
                new_dist = dist[visiting] + graph[neighbor]
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    unvisited.remove_task(neighbor)
                    unvisited.add_task(neighbor, new_dist)

    with open('15.txt') as f:
        graph = Matrix.from_string(f.read(), '')
    start = (0, 0)
    end = (len(graph.data)) - 1, (len(graph.data[0])) - 1
    print(dijkstra(graph, start, end))
    graph = Matrix([[(col + n_col + n_row - 1) % 9 + 1 for n_col in range(5) for col in row]
                    for n_row in range(5) for row in graph.data])
    end = (len(graph.data)) - 1, (len(graph.data[0])) - 1
    print(dijkstra(graph, start, end))


def day16():
    def read_packet(binstring, start, version_sum):
        if int(binstring[start:len(binstring)], 2) == 0:
            return len(binstring) - start, version_sum
        i = start
        version = int(binstring[i:i + 3], 2)
        i += 3
        version_sum += version
        type_id = int(binstring[i:i + 3], 2)
        i += 3
        value = 0
        if type_id == 4:
            read_literal = True
            literal = ''
            while read_literal:
                read_literal = bool(int(binstring[i:i + 1], 2))
                i += 1
                literal += binstring[i:i + 4]
                i += 4
            literal = int(literal, 2)
            value = literal
        else:
            length_type = int(binstring[i:i + 1], 2)
            i += 1
            values = []
            if length_type:
                n_sub_packets = int(binstring[i:i + 11], 2)
                i += 11
                for j in range(n_sub_packets):
                    n, version_sum, value = read_packet(binstring, i, version_sum)
                    values.append(value)
                    i += n
            else:
                length = int(binstring[i:i + 15], 2)
                i += 15
                n = 0
                while n < length:
                    n_more, version_sum, value = read_packet(binstring, i + n, version_sum)
                    values.append(value)
                    n += n_more
                i += length
            if type_id == 0:
                value = sum(values)
            elif type_id == 1:
                value = prod(values)
            elif type_id == 2:
                value = min(values)
            elif type_id == 3:
                value = max(values)
            elif type_id == 5:
                value = int(values[0] > values[1])
            elif type_id == 6:
                value = int(values[0] < values[1])
            elif type_id == 7:
                value = int(values[0] == values[1])
        return i - start, version_sum, value

    with open('16.txt') as f:
        hexstring = f.read().strip()
    binstring = bin(int(hexstring, 16))[2:].zfill(len(hexstring) * 4)
    i = 0
    version_sum = 0
    n, version_sum, value = read_packet(binstring, i, version_sum)
    print(version_sum)
    print(value)


if __name__ == '__main__':
    day16()