import numpy as np

import matplotlib.pyplot as plt





# -----------------------------------

# Question 1 – Histograms

# -----------------------------------



def normal_histogram(n):

    data = np.random.normal(0, 1, n)

    plt.figure()

    plt.hist(data, bins=10)

    plt.xlabel("Value")

    plt.ylabel("Frequency")

    plt.title("Histogram of Normal(0,1)")

    plt.show()

    return data





def uniform_histogram(n):

    data = np.random.uniform(0, 10, n)

    plt.figure()

    plt.hist(data, bins=10)

    plt.xlabel("Value")

    plt.ylabel("Frequency")

    plt.title("Histogram of Uniform(0,10)")

    plt.show()

    return data





def bernoulli_histogram(n):

    data = np.random.binomial(1, 0.5, n)

    plt.figure()

    plt.hist(data, bins=10)

    plt.xlabel("Value")

    plt.ylabel("Frequency")

    plt.title("Histogram of Bernoulli(0.5)")

    plt.show()

    return data





# -----------------------------------

# Question 2 – Sample Mean & Variance

# -----------------------------------



def sample_mean(data):

    data = np.array(data)

    return np.sum(data) / len(data)





def sample_variance(data):

    data = np.array(data)

    mean = sample_mean(data)

    n = len(data)

    return np.sum((data - mean) ** 2) / (n - 1)





# -----------------------------------

# Question 3 – Order Statistics

# -----------------------------------



def order_statistics(data):

    data = np.sort(np.array(data))



    minimum = data[0]

    maximum = data[-1]

    median = np.median(data)



    q1 = np.percentile(data, 25, method="midpoint")

    q3 = np.percentile(data, 75, method="midpoint")



    return (minimum, maximum, median, q1, q3)





# -----------------------------------

# Question 4 – Sample Covariance

# -----------------------------------



def sample_covariance(x, y):

    x = np.array(x)

    y = np.array(y)



    mean_x = sample_mean(x)

    mean_y = sample_mean(y)



    n = len(x)

    return np.sum((x - mean_x) * (y - mean_y)) / (n - 1)





# -----------------------------------

# Question 5 – Covariance Matrix

# -----------------------------------



def covariance_matrix(x, y):

    var_x = sample_variance(x)

    var_y = sample_variance(y)

    cov_xy = sample_covariance(x, y)



    return np.array([[var_x, cov_xy],

                     [cov_xy, var_y]])
