import os
import logging

import numpy as np
import pandas as pd
import markdown_generator as mg
import matplotlib.pyplot as plt

from datetime import datetime


from system import System
from helpers import *

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO
)


def generate_report(lambda_, mu, p, n, system):
    path = 'results'
    hists_dir_name = 'hists'
    hists_path = os.path.join(path, hists_dir_name)
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(hists_path):
        os.makedirs(hists_path)
    time_now = datetime.now().strftime('%d%m%Y_%H%M%S')
    filename = 'result_%s.md' % (time_now,)
    hist_name = time_now + '.png'
    hist_name_1 = time_now + '-1' + '.png'
    hist_name_4 = time_now + '-4' + '.png'

    with open(os.path.join(path, filename), 'w', encoding='utf-8') as f:
        doc = mg.Writer(f)
        doc.write_heading('Статистика')
        doc.writelines([
            'λ = %.2f' % lambda_, '',
            'μ = %.2f' % mu, '',
            'n = %d' % n, ''
        ])

        df_c, df_times = system.stats.build()
        doc.writelines([
            df_c.describe().T.to_markdown(), '',
        ])

        doc.writelines([
            'Всего отменено: %d' % system.stats.cancellations, '',
            'Всего выполнено: %d' % len(system.stats.requests), ''
        ])

        states_bins, states_counts = system.stats.get_states_probs()
        _rho = lambda_ / mu

        plt.xticks(states_bins)
        plt.hist(system.stats.total_requests, bins=np.array(states_bins), density=True)
        plt.savefig(os.path.join(hists_path, hist_name))

        k = n + system.max_queue
        probs = get_state_probs(_rho, k)

        theor_requests = []
        for i in range(0, len(probs)):
            for j in range(0, int(probs[i] * 1000)):
                theor_requests.append(i)

        plt.clf()
        plt.xticks(states_bins)
        plt.hist(theor_requests, bins=np.array(states_bins), density=True)
        plt.savefig(os.path.join(hists_path, hist_name_1))

        plt.clf()
        plt.plot(states_bins, probs, label="Theoretic")
        plt.plot(states_bins, states_counts / sum(states_counts), label="Practical")
        plt.legend()
        plt.savefig(os.path.join(hists_path, hist_name_4))

        doc.writelines([
            'Теоретические вероятности для состояний системы:',
            '![hist](%s)' % (os.path.join(hists_dir_name, hist_name_1),), '',
            'Практические вероятности для состояний системы:',
            '![hist](%s)' % (os.path.join(hists_dir_name, hist_name),), '',
            pd.DataFrame(data={
                'Теоретическая вероятность': probs,
                'Практическая вероятность': states_counts / sum(states_counts)
            }).T.to_markdown(), ''
        ])

        cancel_prob = system.stats.get_cancel_prob()
        theor_cancel_prob = 0
        relative_bandwidth = 1 - cancel_prob
        theor_relative_bandwidth = 1 - theor_cancel_prob
        absolute_bandwidth = relative_bandwidth * lambda_
        theor_absolute_bandwidth = lambda_ * theor_relative_bandwidth
        theor_queue_size = get_theor_queue_len(_rho)
        theor_channel_loaded = _rho
        theor_system_load = theor_queue_size + theor_channel_loaded

        doc.writelines([
            pd.DataFrame({
                'Вероятность отказа': [theor_cancel_prob, cancel_prob],
                'Относительная пропускная способность': [theor_relative_bandwidth, relative_bandwidth],
                'Абсолютная пропускная способность': [theor_absolute_bandwidth, absolute_bandwidth],
                'Длина очереди': [theor_queue_size, np.mean(system.stats.queue_sizes)],
                'Количество занятых каналов': [theor_channel_loaded, np.mean(system.stats.working_channels)],
                'Количество заявок в системе': [theor_system_load, np.mean(system.stats.total_requests)],
            }, index=['Теор.', 'Практ.']).T.to_markdown(), ''
        ])

        doc.writelines([
            df_times.describe().T.to_markdown(), ''
        ])

        theor_queue_time = theor_queue_size / lambda_
        theor_overall_request_time = theor_system_load / lambda_

        doc.writelines([
            pd.DataFrame({
                'Теор. среднее время пребывания заявки в очереди': [theor_queue_time],
                'Теор. среднее время пребывания заявки в СМО': [theor_overall_request_time],
            }, index=['Значение']).T.to_markdown(), ''
        ])

        sum_per_day = system.sum * 48 / system.request_limit

        doc.writelines([
            pd.DataFrame({
                'Сумма штрафа за сутки при S = 10 рублей': [sum_per_day],
            }, index=['Значение']).T.to_markdown(), ''
        ])

        plt.clf()


def main():
    lambda_ = 2
    mu = 3
    p = 1
    n = 1
    system = System(n, lambda_, mu, p, 0.01, 1000)
    system.log()
    system.run()
    generate_report(lambda_, mu, p, n, system)

if __name__ == '__main__':
    main()