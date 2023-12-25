import random


def random_sum_to(n, num_terms=None):  # return fixed sum randomly distributed
    num_terms = (num_terms or random.randint(2, n)) - 1
    a = random.sample(range(1, n), num_terms) + [0, n]
    list.sort(a)
    return [a[j + 1] - a[j] for j in range(len(a) - 1)]

