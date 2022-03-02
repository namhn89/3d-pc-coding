import logging


def init_log(path_log: str):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s][%(name)s:%(lineno)d] - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(path_log, mode='w'),
            logging.StreamHandler()
        ]
    )
    return
