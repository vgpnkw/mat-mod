from sympy import *
import matplotlib.pyplot as plt
import scipy.stats as sta
import numpy as np

N, M = 2, 3
counter = 10
round = 1000

X = np.zeros(N)
Y = np.zeros(M)

xy = np.random.dirichlet(np.ones(N * M))


def x_y_row(N, x_y, counter):
    for i in range(N):
        do = True
        while (do):
            temp_item = np.random.randint(1, counter)
            if temp_item not in x_y:
                x_y[i] = temp_item
                do = False
    return np.sort(x_y)


X = x_y_row(N, X, counter)
Y = x_y_row(M, Y, counter)
print('X = ', X)
print('Y = ', Y)

# распределение по x
p_x = np.zeros(N)
sum = 0
for i in range(N * M):
    sum += xy[i]
    if (i + 1) % M == 0:
        p_x[int(i / M)] = sum
        sum = 0


# функция распределения для x
def distribution_F(N, p):
    f = np.zeros(N)
    sum = 0
    for i in range(N):
        sum += p[i]
        f[i] = sum
    return f


# функция распределения для двумерной
def distribution_F_y_x(N, M, xy, p_x):
    y_x_f = np.zeros((N, M))

    for i in range(N):
        sum = 0
        for j in range(M):
            sum += xy[i * M + j]
            y_x_f[i, j] = sum / p_x[i]

    return y_x_f


F_x = distribution_F(N, p_x)
F_yx = distribution_F_y_x(N, M, xy, p_x)
discrete_f = np.zeros((round, 2))


# ДСВ
def discrete_random_value(round, X, Y, F_x, F_yx, d_f):
    for i in range(round):
        x_y_random = np.random.uniform(size=2)
        x = x_y_random[0]
        y = x_y_random[1]

        x_index = np.searchsorted(F_x, x)
        y_index = np.searchsorted(F_yx[x_index], y)

        d_f[i][0] = X[x_index]
        d_f[i][1] = Y[y_index]
    return d_f


DSV = discrete_random_value(round, X, Y, F_x, F_yx, discrete_f)

print('discrete_random_value:')
print(DSV)


def empiric_probability_matrix(round, DSV, X, Y):
    matrix = np.zeros((N, M))
    unique_discreteSV, SV_counts = np.unique(DSV, return_counts=True, axis=0)
    count_diff_elem_SV = len(unique_discreteSV)

    for i in range(count_diff_elem_SV):
        x_ind = np.where(X == unique_discreteSV[i][0])
        y_ind = np.where(Y == unique_discreteSV[i][1])
        matrix[x_ind, y_ind] = SV_counts[i] / round
    return matrix


empiric_probability_matrix = empiric_probability_matrix(round, DSV, X, Y)
print('empiric_probability_matrix: ', empiric_probability_matrix)

teoretical_probability_matrix = np.hsplit(xy, N)
print('teoretical_probability_matrix', teoretical_probability_matrix)


def plot_histogram(DSV, title, measures):
    if measures == 'x':
        measure = [SV[0] for SV in DSV]
    else:
        measure = [SV[1] for SV in DSV]
    diff_measure, count_measure = np.unique(measure, return_counts=True)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_title(title)
    rects1 = ax.bar(diff_measure, count_measure / len(measure), tick_label=diff_measure)
    plt.show()


plot_histogram(DSV, 'Гистограмма  X', 'x')
plot_histogram(DSV, 'Гистограмма  Y', 'y')


# эмпирич мат ожид
def expected_value(DSV, round, measures):
    if measures == 'x':
        return np.sum([SV[0] for SV in DSV]) / round
    else:
        return np.sum([SV[1] for SV in DSV]) / round


expected_value_x = expected_value(DSV, round, 'x')
expected_value_y = expected_value(DSV, round, 'y')


# теорит мат ожид
def theoretical_expected_value(xy, measure, measures):
    if measures == 'x':
        return np.sum(np.sum(xy, axis=1) * measure)
    else:
        return np.sum(np.sum(xy, axis=0) * measure)


theoretical_expected_value_x = theoretical_expected_value(teoretical_probability_matrix, X, 'x')
theoretical_expected_value_y = theoretical_expected_value(teoretical_probability_matrix, Y, 'y')


print('Эмпирическое М[X] = ', expected_value_x)
print('Теоретическое М[X] = ', theoretical_expected_value_x)


print('Эмпирическое М[Y] = ', expected_value_y)
print('Теоретическое М[Y] = ', theoretical_expected_value_y)


# дисперсия
def dispersion_value(DSV, type_measures):
    if type_measures == 'x':
        return np.var([SV[0] for SV in DSV], ddof=1)
    else:
        return np.var([SV[1] for SV in DSV], ddof=1)


dispersion_value_x = dispersion_value(DSV, 'x')
dispersion_value_y = dispersion_value(DSV, 'y')


def theoretical_dispersion_value(xy, measures, type_measures):
    if type_measures == 'x':
        sum_by_x = np.sum(xy, axis=1)
        return np.sum(sum_by_x * (measures ** 2)) - np.sum(sum_by_x * measures) ** 2
    else:
        sym_by_y = np.sum(xy, axis=0)
        return np.sum(sym_by_y * (measures ** 2)) - np.sum(sym_by_y * measures) ** 2


theoretical_dispersion_value_x = theoretical_dispersion_value(teoretical_probability_matrix, X, 'x')
theoretical_dispersion_value_y = theoretical_dispersion_value(teoretical_probability_matrix, Y, 'y')


print('Эмпирическое D[X] = ', dispersion_value_x)
print('Теоретическое D[X] = ', theoretical_dispersion_value_x)


print('Эмпирическое D[Y] = ', dispersion_value_y)
print('Теоретическое D[Y] = ', theoretical_dispersion_value_y)


def intervals_expected(DSV, measures, round, r=0.95):
    normal_quantile = sta.norm.ppf((1 + r) / 2)
    if measures == 'x':
        value = [SV[0] for SV in DSV]
    else:
        value = [SV[1] for SV in DSV]

    sv_mean = np.mean(value)
    sv_var = np.var(value, ddof=1)

    return (sv_mean - np.sqrt(sv_var / round) * normal_quantile,
            sv_mean + np.sqrt(sv_var / round) * normal_quantile)


interfal_x = intervals_expected(DSV, 'x', round)
interfal_y = intervals_expected(DSV, 'y', round)


print('Доверительный интервал мат ожидания X: ', interfal_x)
print('Доверительный интервал мат ожидания Y: ', interfal_y)


def intervals_dispersion(discrete_SV, measures, n, confidence_level=0.95):
    if measures == 'x':
        value = [SV[0] for SV in discrete_SV]

    else:
        value = [SV[1] for SV in discrete_SV]

    sv_var = np.var(value, ddof=1)


    xi = sta.chi2(n - 1)
    array = xi.rvs(100000)
    temp = sta.mstats.mquantiles(array, prob=[(1 - confidence_level) / 2, (1 + confidence_level) / 2])
    xi_plus = temp[0]
    xi_minus = temp[1]

    return ((n - 1) * sv_var / xi_minus,
            (n - 1) * sv_var / xi_plus)


interval_dispersion_x = intervals_dispersion(DSV, 'x', round)
interval_dispersion_y = intervals_dispersion(DSV, 'y', round)


print('Доверительный интервал дисперсии X: ', interval_dispersion_x)
print('Доверительный интервал дисперсии Y: ', interval_dispersion_y)


def correlation(X, Y, matr, mX, mY, dispepsionX, dispersionY):
    cov = 0
    for i in range(len(X)):
        for j in range(len(Y)):
            cov = cov + (X[i] * Y[j] * matr[i][j])

    cov -= mX * mY
    c = cov / np.sqrt(dispepsionX * dispersionY)
    return cov, c


covarilation_theoretical, correlation_theoretical = correlation(X, Y, teoretical_probability_matrix, theoretical_expected_value_x, theoretical_expected_value_y, theoretical_dispersion_value_x, theoretical_dispersion_value_y)
covar_empiric, correlation_empiric = correlation(X, Y, empiric_probability_matrix, expected_value_x, expected_value_y, dispersion_value_x, dispersion_value_y)


print('Теоретическая ковариация = ', covarilation_theoretical)
print('Теоретический коэфф корреляции = ', correlation_theoretical)


print('Эмпиричекая ковариация = ', covar_empiric)
print('Эмпирический коэффициент корреляция = ', correlation_empiric)
