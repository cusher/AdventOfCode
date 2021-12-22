from collections import Counter, defaultdict
from copy import copy, deepcopy
from functools import reduce
from heapq import heappop, heappush
from itertools import count, takewhile
from math import ceil, floor, inf, prod


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
    class BitStream:
        def __init__(self, bitstring):
            self.bitstring = bitstring
            self.index = 0

        def read_bits_raw(self, n_bits):
            temp = self.bitstring[self.index:self.index + n_bits]
            self.index += n_bits
            return temp

        def read_bits(self, n_bits):
            return int(self.read_bits_raw(n_bits), 2)

    def read_packet():
        bit_stream.version_sum += bit_stream.read_bits(3)
        type_id = bit_stream.read_bits(3)
        if type_id == 4:
            keep_reading = True
            literal = ''
            while keep_reading:
                keep_reading = bool(bit_stream.read_bits(1))
                literal += bit_stream.read_bits_raw(4)
            return int(literal, 2)
        length_type = bit_stream.read_bits(1)
        values = []
        if length_type:
            n_sub_packets = bit_stream.read_bits(11)
            for j in range(n_sub_packets):
                values.append(read_packet())
        else:
            length = bit_stream.read_bits(15)
            n0 = bit_stream.index
            while bit_stream.index < n0 + length:
                values.append(read_packet())
        if type_id == 0:
            return sum(values)
        elif type_id == 1:
            return prod(values)
        elif type_id == 2:
            return min(values)
        elif type_id == 3:
            return max(values)
        elif type_id == 5:
            return int(values[0] > values[1])
        elif type_id == 6:
            return int(values[0] < values[1])
        elif type_id == 7:
            return int(values[0] == values[1])

    with open('16.txt') as f:
        hexstring = f.read().strip()
    bit_stream = BitStream(bin(int(hexstring, 16))[2:].zfill(len(hexstring) * 4))
    bit_stream.version_sum = 0
    value = read_packet()
    print(bit_stream.version_sum)
    print(value)


def day17():
    def x_pos_at(x_vel, step):
        temp = x_vel - step if step < x_vel else 0
        return (x_vel ** 2 + x_vel - temp ** 2 - temp) / 2

    def y_pos_at(y_vel, step):
        if step <= y_vel:
            return x_pos_at(y_vel, step)
        else:
            peak = x_pos_at(y_vel, y_vel)
            remainder = step - y_vel - 1
            return peak - x_pos_at(remainder, remainder)

    def possible_y_velocities(y_area):
        options = defaultdict(list)
        for i in range(y_area[0], -y_area[0]):
            for j in range(0, -y_area[0] * 2 + 1):
                if y_area[0] <= y_pos_at(i, j) <= y_area[1]:
                    options[i].append(j)
        return options

    def possible_x_velocities(x_area, cap):
        options = defaultdict(list)
        for i in range(x_area[1] + 1):
            for j in range(cap + 1):
                if x_area[0] <= x_pos_at(i, j) <= x_area[1]:
                    options[i].append(j)
        return options

    with open('17.txt') as f:
        areas = [[int(v) for v in r.split('..')] for r in f.readline().strip("target area: x=").split(', y=')]
    vy_options = possible_y_velocities(areas[1])
    biggest_step = max(reduce(lambda a, b: a | set(b), vy_options.values(), set()))
    vx_options = possible_x_velocities(areas[0], biggest_step)
    max_y_vel = 0
    speeds = set()
    for vy, y_steps in vy_options.items():
        for vx, x_steps in vx_options.items():
            if set(x_steps) & set(y_steps):
                max_y_vel = vy
                speeds.add((vx, vy))
    print(int(y_pos_at(max_y_vel, max_y_vel)))
    print(len(speeds))


def day18():
    def parse_tree(string, i):
        if string[i] == '[':
            left, i = parse_tree(string, i + 1)
            right, i = parse_tree(string, i + 1)
            return [left, right], i + 1
        else:
            n = min(string[i:].find(char) for char in (',', ']') if char in string[i:])
            val = int(string[i:i + n])
            return val, i + n

    def add_to_edge(tree, i, number):
        if isinstance(tree[i], int):
            tree[i] += number
        else:
            add_to_edge(tree[i], i, number)

    def get_exploder(tree, depth):
        for i in (0, 1):
            if isinstance(tree[i], int):
                pass
            elif isinstance(tree[i][0], int) and isinstance(tree[i][1], int):
                if depth >= 4:
                    exploded = tree[i]
                    tree[i] = 0
                    return do_explode(tree, i, exploded)
            else:
                exploded = get_exploder(tree[i], depth + 1)
                if exploded is None:
                    continue
                return do_explode(tree, i, exploded)

    def do_explode(tree, i, exploded):
        n = int(not i)
        if isinstance(tree[n], int):
            tree[n] += exploded[n]
        else:
            add_to_edge(tree[n], i, exploded[n])
        return [0, exploded[i]] if i else [exploded[i], 0]

    def get_splitter(tree):
        for i in (0, 1):
            if isinstance(tree[i], int):
                if tree[i] >= 10:
                    tree[i] = [floor(tree[i] / 2), ceil(tree[i] / 2)]
                    return True
            elif get_splitter(tree[i]):
                return True

    def add_reduce(a, b):
        tree = [a, b]
        last_result = True
        while last_result:
            last_result = get_exploder(tree, 1)
            if not last_result:
                last_result = get_splitter(tree)
        return tree

    def magnitude(tree):
        if isinstance(tree, int):
            return tree
        return 3 * magnitude(tree[0]) + 2 * magnitude(tree[1])

    with open('18.txt') as f:
        entries = [line.strip() for line in f.readlines()]
    trees = [parse_tree(entry, 0)[0] for entry in entries]
    print(magnitude(reduce(add_reduce, map(deepcopy, trees))))
    mags = [magnitude(add_reduce(deepcopy(a), deepcopy(b))) for a in trees for b in trees if a != b]
    print(max(mags))


def day21():
    def deterministic_game(positions):
        scores = [0, 0]
        roll = -3
        for turn in count(0):
            player = turn % 2
            roll += 9
            positions[player] = (positions[player] + roll - 1) % 10 + 1
            scores[player] += positions[player]
            if max(scores) >= 1000:
                return min(scores) * (turn + 1) * 3

    odds_3d3 = ((3, 1), (4, 3), (5, 6), (6, 7), (7, 6), (8, 3), (9, 1))
    win_outcomes = ((1, 0), (0, 1))

    def dirac_turn(positions, scores, player):
        wins_0, wins_1 = 0, 0
        for roll, chance in odds_3d3:
            sub_outcomes = dirac_roll(positions.copy(), scores.copy(), player, roll)
            wins_0 += sub_outcomes[0] * chance
            wins_1 += sub_outcomes[1] * chance
        return wins_0, wins_1

    def dirac_roll(positions, scores, player, roll):
        positions[player] = (positions[player] + roll - 1) % 10 + 1
        scores[player] += positions[player]
        if scores[player] >= 21:
            return win_outcomes[player]
        return dirac_turn(positions, scores, not player)

    with open('21.txt') as f:
        starts = [int(line.strip().split(': ')[1]) for line in f.readlines()]
    print(deterministic_game(starts.copy()))
    temp = dirac_turn(starts, [0, 0], 0)
    print(max(temp))


def day22():
    def overlap_1d(a, b):
        if a[0] <= b[1] <= a[1] or b[0] <= a[1] <= b[1]:
            return max(a[0], b[0]), min(a[1], b[1])

    def overlap(a, b):
        overlaps = [overlap_1d(a[i], b[i]) for i in range(3)]
        return overlaps if all(overlaps) else None

    def volume(box):
        return prod(box[i][1] - box[i][0] + 1 for i in range(3))

    procedure = []
    with open('22.txt') as f:
        entries = [line.strip() for line in f.readlines()]
        for entry in entries:
            bit = 1 if entry.split(' ')[0] == 'on' else 0
            pos = [[int(val) for val in coord.split('=')[1].split('..')] for coord in entry.split(' ')[1].split(',')]
            procedure.append((bit, pos))
    on_cubes = set()
    for step in procedure:
        for i in range(max(step[1][0][0], -50), min(step[1][0][1], 50) + 1):
            for j in range(max(step[1][1][0], -50), min(step[1][1][1], 50) + 1):
                for k in range(max(step[1][2][0], -50), min(step[1][2][1], 50) + 1):
                    if step[0]:
                        on_cubes.add((i, j, k))
                    elif (i, j, k) in on_cubes:
                        on_cubes.remove((i, j, k))
    print(len(on_cubes))
    regions = []
    for step in procedure:
        new_regions = []
        if step[0]:
            new_regions.append(step)
        for region in regions:
            region_overlap = overlap(step[1], region[1])
            if region_overlap:
                if region[0] > 0:
                    new_regions.append((-1, region_overlap))
                else:
                    new_regions.append((1, region_overlap))
        regions.extend(new_regions)
    print(sum(volume(region[1]) * region[0] for region in regions))


if __name__ == '__main__':
    day22()
