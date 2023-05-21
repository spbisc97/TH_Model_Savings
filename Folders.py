import os
import re


def move_to_path(p=False):
    file_path = os.path.dirname(__file__)
    os.chdir(file_path)
    if p:
        print(os.path.abspath("."))
    return file_path


def pattern_compile():
    pattern_string = "0[4-9]_[0-9]{2}_[0-9]{2}_[0-9]{2}/run_"
    pattern = re.compile(pattern_string)
    return pattern


if __name__ == "__main__":
    base_path = move_to_path(p=True)
    pattern = pattern_compile()
    algos = ["PPO", "DDPG", "TD3"]
    subfolders = ["models", "logs", "imgs"]
    for root, dirs, files in os.walk(base_path):
        for name in files:
            if pattern.search(root):
                print(root)
                print(os.path.join(root, name))
