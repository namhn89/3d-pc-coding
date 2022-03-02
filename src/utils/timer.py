import logging
import time

logger = logging.getLogger(__name__)


class Timer:
    def __init__(self):
        self.start_times = []
        self.jobs = []

    def start(self, job, verbal=False):
        self.jobs.append(job)
        self.start_times.append(time.time())
        if verbal:
            logger.info("[I] {job} started.".format(job=job))

    def stop(self):
        if self.jobs:
            elapsed_time = time.time() - self.start_times.pop()
            logger.info(
                "[I] {job} finished in {elapsed_time:0.3f} s.".format(
                    job=self.jobs.pop(), elapsed_time=elapsed_time
                )
            )
