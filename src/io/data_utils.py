import random
import string
import numpy as np
import re
from typing import List, Tuple, Union

labels_map = {'correct': 0, 'misspelling': 1, 'segmentation': 2, 'union': 3}


def misspelling(s):
    if len(s) == 0:
        return s, 0
    random_char = random.choice(string.printable[:-6])
    random_position = random.randint(0, len(s))

    action = random.randint(0, 2)
    # inserts
    if action == 0:
        return s[:random_position] + random_char + s[random_position:], 1
    # deletes
    elif action == 1:
        if len(s) == 1:
            return s, 0
        return s[:random_position] + s[random_position + 1:], 1
    # replaces
    else:
        return s[:random_position] + random_char + s[random_position + 1:], 1


def segmentation(s):
    if len(s) <= 1:
        return s, 0
    num_spaces = random.randint(1, 5)
    random_position = random.randint(1, len(s) - 1)
    return s[:random_position] + ' ' * num_spaces + s[random_position:], [2, 2]


def union(s1, s2):
    return s1 + s2, 3


def create_dist(probability: float, n_tokens: int) -> List[int]:
    return np.random.choice([1, 0], size=n_tokens, p=[probability, 1 - probability])


def choose_function(function_probabilities: List[float]) -> str:
    functions = ['misspelling', 'segmentation', 'union']
    return np.random.choice(functions, p=function_probabilities)


def alter_token(token: str, function: str, next_token: str = None) -> Tuple[str, Union[int, List[int]]]:
    functions = {
        'misspelling': misspelling,
        'segmentation': segmentation,
        'union': union
    }

    if function == 'union':
        elaborated_token, label = functions.get(function)(s1=token, s2=next_token)
    else:
        elaborated_token, label = functions.get(function)(s=token)

    return elaborated_token, label


def process_string(input_string: str, probability: float, function_probabilities: List[float]) -> Tuple[str, List[int]]:
    tokens = [match.group() for match in re.finditer(r'\S+', input_string)]
    n_tokens = len(tokens)
    distribution = create_dist(probability, n_tokens)
    modified_tokens = []
    labels = []

    i = 0
    while i < n_tokens:
        if distribution[i] == 1:
            chosen_function = choose_function(function_probabilities)
            while chosen_function == 'union' and i == n_tokens - 1:
                chosen_function = choose_function(function_probabilities)

            if chosen_function == 'union' and i < n_tokens - 1:
                elaborated_token, label = alter_token(tokens[i], chosen_function, tokens[i + 1])
                modified_tokens.append(elaborated_token)
                labels.append(label)
                i += 2
            else:
                elaborated_token, label = alter_token(tokens[i], chosen_function)
                if isinstance(label, list):
                    modified_tokens.append(elaborated_token)
                    labels.extend(label)
                else:
                    modified_tokens.append(elaborated_token)
                    labels.append(label)
                i += 1
        else:
            modified_tokens.append(tokens[i])
            labels.append(0)
            i += 1

    result_string = ' '.join(modified_tokens)

    return result_string, labels


def create_dataset(data, probability, function_probabilities):
    rows = []
    for line in data:
        row = [line]
        row.extend(process_string(line, probability, function_probabilities))
        rows.append(row)
    return rows
