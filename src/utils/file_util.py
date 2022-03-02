import yaml


def read_yaml(path_file):
    with open(path_file) as file:
        res = yaml.load(file, Loader=yaml.FullLoader)
        return res
