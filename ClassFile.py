__author__ = 'magniear'


class Clusters(object):

    def __init__(self):
        self.clusters = []

    def add_cluster(self, cluster):
        """

        :type cluster: Cluster
        """
        self.clusters.append(cluster)


class Cluster(object):

    def __init__(self, mean, scatter):
        self.mean = float(mean)
        self.scatter = float(scatter)

    def display_mean(self):
        print ("Cluster mean %d" % self.mean)

    def display_scatter(self):
        print ("Scatter : ", self.scatter)


class Samples(object):

    def __init__(self):
        self.samples = []

    def add_sample(self, sample):
        self.samples.append(sample)


class Sample(object):

    def __init__(self, mean, scatter, omega, size, name):
        self.mean = float(mean)
        self.scatter = float(scatter)
        self.omega = float(omega)
        self.size = int(size)
        self.name = name
        self.scatter_maturity = []

    def add_scatter_maturity(self, scatter_maturity):
        self.scatter_maturity.append(scatter_maturity)

    def add_all_scatter_maturity(self, scatter_maturity):
        self.scatter_maturity = scatter_maturity


class ScatterMaturity(object):

    def __init__(self, scatter, maturity):
        self.scatter = float(scatter)
        self.maturity = int(maturity)


class ClusterResults(object):

    number = 0

    def __init__(self):
        self.commodities = []
        self.distances = []

    def add_commodity(self, commodity):
        self.commodities.append(commodity)
        self.number += 1

    def add_distance(self, distance):
        self.distances.append(distance)


class Results(object):

    def __init__(self):
        self.cluster_results = []

    def add_cluster_result(self, cluster):
        self.cluster_results.append(cluster)


class ProjectedSeries(object):

    def __init__(self, name):
        self.components = []
        self.name = name

    def add_components(self, component):
        self.components.append(component)


class Component(object):

    def __init__(self, contribution):
        self.series = []
        self.contribution = contribution

    def add_series(self, series):
        self.series.append(series)
