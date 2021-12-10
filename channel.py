import random
import logging

import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Channel:
    COUNT = 0

    def __init__(self, mu, reject_probability, on_reject):
        Channel.COUNT += 1
        self.id = Channel.COUNT

        self.reject_probability = reject_probability
        self.on_reject = on_reject
        self.mu = mu

        self.free = True
        self.end_at = 0
        self.request = None

    def run(self, start_at, request):
        self.free = False
        self.end_at = start_at + np.random.exponential(1 / self.mu)
        self.request = request
        logging.info('[Стартовал] Канал #%d: с %.4f до %.4f' %
                     (self.id, start_at, self.end_at))

    def try_free(self, cur_time):
        if not self.free and self.end_at < cur_time:
            self.free = True

            rejected = random.random() <= self.reject_probability
            if rejected:
                self.on_reject(self.request)
                # print(f"{bcolors.OKBLUE}Warning: No active frommets remain")
                logging.warning('[Отклонено] Канал #%d: освободился в %.4f' %
                             (self.id, cur_time))
            else:
                logging.info('[Выполнено] Канал #%d: осободился в %.4f' %
                             (self.id, cur_time))
                return self.request