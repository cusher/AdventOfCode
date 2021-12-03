def day1():
    counter = 0
    with open('1.txt') as f:
        nums = [int(val) for val in f.readlines()]
    for i, val in enumerate(nums):
        if i != 0 and val > nums[i - 1]:
            counter += 1
    print(counter)
    counter = 0
    for i, val in enumerate(nums):
        if i != 0 and i + 2 < len(nums):
            window = nums[i] + nums[i + 1] + nums[i + 2]
            prev = nums[i-1] + nums[i] + nums[i + 1]
            if window > prev:
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
        count = 0
        for v in values:
            if v[position] == '1':
                count += 1
        return str(int((count >= len(values) / 2) == most))

    with open('3.txt') as f:
        entries = [line.strip() for line in f.readlines()]

    n = len(entries[0])
    gamma = ''.join([common_bit(i, entries, True) for i in range(n)])
    epsilon = ''.join([common_bit(i, entries, False) for i in range(n)])
    print(int(gamma, 2) * int(epsilon, 2))

    ratings = [None, None]
    for option in range(2):
        matches = list(entries)
        for i in range(len(entries[0])):
            if ratings[option] is None:
                cb = common_bit(i, matches, bool(option))
                matches = list(filter(lambda entry: entry[i] == cb, matches))
                if len(matches) == 1:
                    ratings[option] = matches[0]
    print(int(ratings[0], 2) * int(ratings[1], 2))


def day4():
    with open('4.txt') as f:
        entries = [line.strip() for line in f.readlines()]


if __name__ == '__main__':
    day4()
