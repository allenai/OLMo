
"""
Script to create downstream eval datasets for math.
"""
import os
import random

import numpy as np
import pandas as pd
from datasets import Dataset
from olmo.torch_util import seed_all

seed_all(6198)


def run(save_to: str):

    num_questions = 1000

    simple_operations = ['+', '-']
    # operations = ['+', '-', '*', '/']

    # easy; only additions and subtractions

    def build_eqn(variables, operations):
        question = ''
        y_eqn = 'y ='
        for i, var in enumerate(variables):
            var_name = chr(ord('a')+i)
            question += f"{var_name} = {str(var)}; "
            y_eqn += f" {var_name}"
            if i < len(variables) - 1:
                y_eqn += f" {operations[i]}"
        question += f"{y_eqn}; print(y);"
        return question

    how_many = int(np.ceil(num_questions / (4 * 3)))
    # how_many = 1
    q_count = 0
    instances = []
    labels = ["A", "B", "C", "D"]
    for var_count in [1, 2, 3, 4]:
        num_ops = var_count - 1
        for num_digits in [1, 2, 3]:
            min_val = -10 ** num_digits
            max_val = 10 ** num_digits
            max_output = max_val * var_count  # only addition and subtraction
            min_output = -max_output
            for _ in range(how_many):
                vars = np.random.randint(min_val, max_val, size=var_count)
                ops = [simple_operations[i] for i in np.random.randint(0, 2, num_ops)]

                # eqn = 'a = 2\nb = 3\nc = -7\ny = a + b - c\nprint(y)\n12
                question = build_eqn(vars, ops)
                outputs = {}
                exec(question, outputs)
                answer = outputs["y"]
                choices = [str(answer)] + [str(random.choice([j for j in range(min_output, max_output) if j != answer])) for _ in range(3)]
                random.shuffle(choices)
                answer_key = labels[choices.index(str(answer))]

                q_count += 1

                id_str = f"q{q_count}_max{num_digits}d_max{var_count}"

                if var_count < 2:
                    type_tag = "easy"
                elif num_digits <= 2:
                    type_tag = "medium"
                else:
                    type_tag = "hard"

                instance = {
                    "id": id_str,
                    "question": question,
                    "choices": {
                        "text": choices,
                        "label": labels
                    },
                    "answerKey": answer_key,
                    "type_tag": type_tag
                }
                instances.append(instance)

    random.shuffle(instances)
    df = pd.DataFrame.from_records(instances)
    df.to_json(save_to, lines=True, compression=None, orient="records")
    return instances


if __name__ == "__main__":
    import sys

    run(sys.argv[1])
