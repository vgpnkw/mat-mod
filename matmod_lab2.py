import simpy
import numpy as np
from matplotlib import pyplot as plt
from math import factorial

fig, axs = plt.subplots(2)


def get_empiric_probability_of_failure(rejection_list, m, num_channel) :
    rejection_array = np.array(rejection_list)
    P_fail = len(rejection_array[rejection_array == (num_channel + m + 1)]) / len(rejection_array)
    print("Эмпирическая вероятность отказа: ", P_fail)
    return P_fail


def get_empiric_probability_queuing(rejection_list, m, num_channel) :
    P_queque = []
    rejection_array = np.array(rejection_list)
    for i in range(1, num_channel + m + 2):
        if i > num_channel and i <= num_channel + m:
            P_queque.append(len(rejection_array[rejection_array == i]) / len(rejection_array))
    P_queued = sum(P_queque)
    print("Эмпирическая вероятность образования очереди: ", P_queued)
    return P_queued


def get_empiric_probability(rejection_list, m, num_channel) :
    P_x = []
    items = []
    rejection_array = np.array(rejection_list)
    for i in range(1, num_channel + m + 2):
        P_x.append(len(rejection_array[rejection_array == i]) / len(rejection_array))
    for i, item in enumerate(P_x):
        print(f'Эмпирическая P{i}: {item}')
        items.append(item)
    return items


def get_empiric_throughput(rejection_list, m, num_channel) :
    rejection_array = np.array(rejection_list)
    P_fail = len(rejection_array[rejection_array == (num_channel + m + 1)]) / len(rejection_array)
    Q = 1 - P_fail
    throughput = []
    throughput.append(Q)
    print("Эмпирическая относительная пропускная способность: ", Q)
    A = Q * lambd
    throughput.append(A)
    print("Эмпирическая абсолютная пропускная способность: ", A)
    return throughput


def get_n_empiric_people_queue (queued_list) :
    n_people_queque = average_count_people(queued_list)
    print("(эмрипич.)")
    return n_people_queque


def get_empiric_K_av (smo_list) :
    K_av = average_smo_count_people(smo_list)
    print("(эмрипич.)")
    return K_av


def get_empiric_number_of_busy_channels (Q) :
    n_average = Q * lambd / mu
    print("Cреднее число занятых каналов: ", n_average, end=" ")
    print("(эмрипич.)")
    return n_average


def get_empiric_T_queue (queued_time) :
    T_queque = average_queqe_time(queued_time)
    print("(эмрипич.)")
    return T_queque


def get_empiric_T_smo (wait_times) :
    T_smo = average_smo_time(wait_times)
    print("(эмрипич.)")
    return T_smo


def axs_add (queued_list, wait_times) :
    axs[0].hist(wait_times, 50)
    axs[0].set_title('Wait times')
    axs[1].hist(queued_list, 50)


def servicing(env, visitors, smo, mu, v, m, num_channel):
    len_queque_global = len(smo.loader.queue)
    count_active_channel_global = smo.loader.count
    with smo.loader.request() as request:
        len_queque = len(smo.loader.queue)
        count_active_channel = smo.loader.count
        smo.queued_list.append(len_queque_global)
        smo.smo_list.append(len_queque_global + count_active_channel_global)
        if len_queque <= m:
            smo.rejection_list.append(count_active_channel + len_queque)
            arrival_time = env.now
            result = yield request | env.process(smo.waiting(visitors, v))
            smo.queued_time.append(env.now - arrival_time)
            if request in result:
                yield env.process(smo.rejection(visitors, mu))
                smo.wait_times.append(env.now - arrival_time)
            else:
                smo.wait_times.append(env.now - arrival_time)
        else:
            smo.rejection_list.append(m + num_channel + 1)
            smo.queued_time.append(0)
            smo.wait_times.append(0)


class SMO(object):
    def __init__(self, env, num_channel):
        self.env = env
        self.wait_times = []
        self.queued_list = []
        self.queued_time = []
        self.rejection_list = []
        self.smo_list = []
        self.loader = simpy.Resource(env, num_channel)

    def rejection(self, visitors, mu):
        yield self.env.timeout(np.random.exponential(1 / mu))

    def waiting(self, visitors, v):
        yield self.env.timeout(np.random.exponential(1 / v))


def run_SMO(env, smo, num_channel, lambd, mu, v, m):
    visitors = 0
    while True:
        yield env.timeout(np.random.exponential(1 / lambd))
        env.process(servicing(env, visitors, smo, mu, v, m, num_channel))
        visitors = visitors + 1


def generate_SMO(num_channel, lambd, mu, v, m, test_time):
    env = simpy.Environment()
    smo = SMO(env, num_channel)
    env.process(run_SMO(env, smo, num_channel, lambd, mu, v, m))
    env.run(until=test_time)
    return smo.wait_times, smo.queued_list, smo.queued_time, smo.rejection_list, smo.smo_list


def average_count_people(queued_list):
    average_L = np.array(queued_list).mean()
    print("Среднее число заявок в очереди: ", average_L, end=" ")
    return average_L


def average_smo_count_people(smo_list):
    average_L = np.array(smo_list).mean()
    print("Среднее число заявок, обслуживаемых в СМО: ", average_L, end=" ")
    return average_L


def average_queqe_time(queued_time):
    average_queued_time = np.array(queued_time).mean()
    print("Среднее время пребывания заявки в очереди: %s" % (average_queued_time), end=" ")
    return average_queued_time


def average_smo_time(wait_times):
    average_T = np.array(wait_times).mean()
    print("Среднее время пребывания заявки в СМО: %s " % (average_T), end=" ")
    return average_T


def get_teoretic_probability (num_channel, m, lambd, mu, v) :
    ro = lambd / mu
    betta = v / mu
    teoretic_propability = []
    sum_p = 0
    p0 = (sum([ro ** i / factorial(i) for i in range(num_channel + 1)]) +
          (ro ** num_channel / factorial(num_channel)) *
          sum([ro ** i / (np.prod([num_channel + t * betta for t in range(1, i + 1)])) for i in range(1, m + 1)])) ** -1
    print('Теоретическая P0:', p0)
    teoretic_propability.append(p0)
    sum_p += p0
    for i in range(1, num_channel + 1):
        px = (ro ** i / factorial(i)) * p0
        sum_p += px
        teoretic_propability.append(px)
        print(f'Теоретическая P{i}: {px}')

    pn = px
    p_queque = px
    for i in range(1, m + 1):
        px = (ro ** (i) / np.prod([num_channel + t * betta for t in range(1, i + 1)])) * pn
        sum_p += px
        if i < m:
            p_queque += px
        print(f'Теоретическая P{num_channel + i}: {px}')
        teoretic_propability.append(px)
    return teoretic_propability, px, p_queque, pn, p0

def get_teoretic_probability_of_failure (px) :
    P = px
    print(f'Теоретическая вероятность отказа: {P}')
    return P

def get_teoretic_probability_queuing(p_queque) :
    print("Теоретическая вероятность образования очереди: ", p_queque)
    return p_queque

def get_teoretic_throughput (P, lambd) :
    throughput = []
    Q = 1 - P
    throughput.append(Q)
    print("Теоретическая относительная пропускная способность: ", Q)
    A = Q * lambd
    throughput.append(A)
    print("Теоретическая абсолютная пропускная способность : ", A)
    return throughput

def get_n_teoretic_people_queue (pn, num_channel, m, lambd, mu, v) :
    ro = lambd / mu
    betta = v / mu
    n_people_queque = sum([i * pn * (ro ** i) / np.prod([num_channel + l * betta for l in range(1, i + 1)]) for
                           i in range(1, m + 1)])
    print("Среднее число заявок в очереди: ", n_people_queque, end=" ")
    print("(теоретич.)")
    return n_people_queque

def get_teoretic_K_av (num_channel, m, lambd, mu, v, p0) :
    ro = lambd / mu
    betta = v / mu
    K_av = sum([index * p0 * (ro ** index) / factorial(index) for index in range(1, num_channel + 1)]) + sum(
        [(num_channel + index) * pn * ro ** index / np.prod(
            np.array([num_channel + l * betta for l in range(1, index + 1)])) for
         index in range(1, m + 1)])
    print("Среднее число заявок, обслуживаемых в СМО : ", K_av, end=" ")
    print("(теоретич.)")
    return K_av

def get_teorecit_number_of_busy_channels (Q, lambd, mu) :
    ro = lambd / mu
    n_average = Q * ro
    print("Cреднее число занятых каналов: ", n_average,end=" ")
    print("(теоретич.)")
    return n_average

def get_teoretic_T_queue (n_people_queque, lambd) :
    T_queque = n_people_queque / lambd
    print("Среднее время пребывания заявки в очереди: ", T_queque,end="")
    print("(теоретич.)")
    return T_queque

def get_teoretic_T_smo (K_av, lambd) :
    T_smo = K_av / lambd
    print("Среднее время пребывания заявки в СМО: ", T_smo, end=" ")
    print("(теоретич.)")
    return T_smo




print("----------------------1--------------------------------------------")
print("------------------------------------------------------------------")
num_channel = 2
lambd = 2
mu = 1
m = 2
v = 1

wait_times, queued_list, queued_time, rejection_list, smo_list = generate_SMO(num_channel, lambd, mu, v, m, 10000)

empiric_probability_of_failure = get_empiric_probability_of_failure(rejection_list, m, num_channel)

empiric_probability_queuing = get_empiric_probability_queuing(rejection_list, m, num_channel)

empiric_probability = get_empiric_probability(rejection_list, m, num_channel)

empiric_throughput = get_empiric_throughput(rejection_list, m, num_channel)

n_empiric_people_queue = get_n_empiric_people_queue(queued_list)

empiric_K_av = get_empiric_K_av(smo_list)

empiric_number_of_busy_channels = get_empiric_number_of_busy_channels(empiric_throughput[0])

empiric_T_queue = get_empiric_T_queue(queued_time)

T_smo = get_empiric_T_smo(wait_times)


axs_add(queued_list, wait_times)
plt.show()

print("------------------------------------------------------------------")
print("------------------------------------------------------------------")


teoretic_probability, px, p_queue, pn, p0 = get_teoretic_probability(num_channel, m, lambd, mu, v)

teoretic_probability_of_failure = get_teoretic_probability_of_failure(px)

teoretic_probability_queuing = get_teoretic_probability_queuing(p_queue)

teoretic_throughput = get_teoretic_throughput(teoretic_probability_of_failure, lambd)

n_teoretic_people_queue = get_n_teoretic_people_queue(pn, num_channel, m, lambd, mu, v)

teoretic_K_av = get_teoretic_K_av(num_channel, m, lambd, mu, v, p0)

teoretic_T_queue = get_teoretic_T_queue(n_teoretic_people_queue, lambd)

teoretic_T_smo = get_teoretic_T_smo(teoretic_K_av, lambd)


print("---------------------2---------------------------------------------")
print("------------------------------------------------------------------")


num_channel = 5
lambd = 4
mu = 3
m = 2
v = 1

wait_times, queued_list, queued_time, rejection_list, smo_list = generate_SMO(num_channel, lambd, mu, v, m, 10000)

empiric_probability_of_failure2 = get_empiric_probability_of_failure(rejection_list, m, num_channel)

empiric_probability_queuing2 = get_empiric_probability_queuing(rejection_list, m, num_channel)

empiric_probability2 = get_empiric_probability(rejection_list, m, num_channel)

empiric_throughput2 = get_empiric_throughput(rejection_list, m, num_channel)

n_empiric_people_queue2 = get_n_empiric_people_queue(queued_list)

empiric_K_av2 = get_empiric_K_av(smo_list)

empiric_number_of_busy_channels2 = get_empiric_number_of_busy_channels(empiric_throughput[0])

empiric_T_queue2 = get_empiric_T_queue(queued_time)

T_smo2 = get_empiric_T_smo(wait_times)


axs_add(queued_list, wait_times)
plt.show()

print("------------------------------------------------------------------")
print("------------------------------------------------------------------")

# P_teoretic = get_teoretic_probabilities(num_channel, m, lambd, mu, v)

teoretic_probability2, px, p_queue, pn, p0 = get_teoretic_probability(num_channel, m, lambd, mu, v)

teoretic_probability_of_failure2 = get_teoretic_probability_of_failure(px)

teoretic_probability_queuing2 = get_teoretic_probability_queuing(p_queue)

teoretic_throughput2 = get_teoretic_throughput(teoretic_probability_of_failure, lambd)

n_teoretic_people_queue2 = get_n_teoretic_people_queue(pn, num_channel, m, lambd, mu, v)

teoretic_K_av2 = get_teoretic_K_av(num_channel, m, lambd, mu, v, p0)

teoretic_T_queue2 = get_teoretic_T_queue(n_teoretic_people_queue, lambd)

teoretic_T_smo2 = get_teoretic_T_smo(teoretic_K_av, lambd)


print("-----------------------3-------------------------------------------")
print("------------------------------------------------------------------")


num_channel = 10
lambd = 12
mu = 11
m = 5
v = 4

wait_times, queued_list, queued_time, rejection_list, smo_list = generate_SMO(num_channel, lambd, mu, v, m, 10000)

empiric_probability_of_failure3 = get_empiric_probability_of_failure(rejection_list, m, num_channel)

empiric_probability_queuing3 = get_empiric_probability_queuing(rejection_list, m, num_channel)

empiric_probability3 = get_empiric_probability(rejection_list, m, num_channel)

empiric_throughput3 = get_empiric_throughput(rejection_list, m, num_channel)

n_empiric_people_queue3 = get_n_empiric_people_queue(queued_list)

empiric_K_av3 = get_empiric_K_av(smo_list)

empiric_number_of_busy_channels3 = get_empiric_number_of_busy_channels(empiric_throughput[0])

empiric_T_queue3 = get_empiric_T_queue(queued_time)

T_smo3 = get_empiric_T_smo(wait_times)


axs_add(queued_list, wait_times)
plt.show()

print("------------------------------------------------------------------")
print("------------------------------------------------------------------")


teoretic_probability3, px, p_queue, pn, p0 = get_teoretic_probability(num_channel, m, lambd, mu, v)

teoretic_probability_of_failure3 = get_teoretic_probability_of_failure(px)

teoretic_probability_queuing3 = get_teoretic_probability_queuing(p_queue)

teoretic_throughput3 = get_teoretic_throughput(teoretic_probability_of_failure, lambd)

n_teoretic_people_queue3 = get_n_teoretic_people_queue(pn, num_channel, m, lambd, mu, v)

teoretic_K_av3 = get_teoretic_K_av(num_channel, m, lambd, mu, v, p0)

teoretic_T_queue3 = get_teoretic_T_queue(n_teoretic_people_queue, lambd)

teoretic_T_smo3 = get_teoretic_T_smo(teoretic_K_av, lambd)


print("------------------------------------------------------------------")
print("------------------------------------------------------------------")


