from catwalk.tasks import TASKS
from catwalk.tasks.tasks_lm import TASKS_LM

if __name__ == "__main__":
    for key in TASKS:
        print(key)

    for key in TASKS_LM:
        print(key)
