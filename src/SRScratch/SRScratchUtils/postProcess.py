# plotting of u_best on testing data. Show the function found vs the data
import matplotlib.pyplot as plt
from alpine.data import Dataset
from sklearn.metrics import r2_score
import numpy as np

from sympy import symbols, lambdify
from sympy.abc import x


class postProcess():
    def __init__(self,
                 ind_str_norm,  # string of the simplified expression
                 ind_str_unnorm: str = None  # un-normalised string
                 ):

        self.ind = ind_str_norm
        self.ind_unnorm = ind_str_unnorm
        self.func = lambdify([x], self.ind)
        if ind_str_unnorm is not None:
            self.func_unnorm = lambdify([x], self.ind_unnorm)

    def plotBestIndividualOnTestData(self, X, y):
        # takes either train or testing data
        plt.figure()
        plt.plot(self.func(X), '-', label='Best individual')
        plt.plot(y, '+', label='Test data')
        plt.legend()
        plt.grid()
        plt.show()

    def plotFunction(self, x, y):
        # takes non-shuffled data and plots
        plt.figure()
        plt.plot(x, self.func(x), '-', label='Best individual')
        plt.plot(x, y, '-', label='Data')
        plt.xlabel(r'Crack length, $a$')
        plt.ylabel(r'f(a/b)')
        plt.legend()
        plt.grid()
        plt.show()

    def plotting(self, x_norm, y_norm, data_name):

        fig, ax = plt.subplots(2, 1, sharex=True)

        x = x_norm
        if data_name == 'DEN' or data_name == 'CEN':  # if data is DEN or CEN
            y = y_norm / ((1 - x)**(1/2))
            ax[0].set_ylabel(r'$(1-a/b)^{1/2}f(a/b)$')
        else:  # if data is SEN1 or SEN2
            y = y_norm / ((1 - x)**(3/2))
            ax[0].set_ylabel(r'$(1-a/b)^{3/2}f(a/b)$')

        ax[0].plot(x_norm, self.func(x_norm), '-', label='Best individual')
        ax[0].plot(x_norm, y_norm, '+', label='Data')
        ax[0].set_title('Normalised model')
        # ax[0].set_xlabel(r'Crack length, $a$')
        ax[0].set_xlim([0, 1])
        # ax[0].set_ylim([0, 1])
        ax[0].grid()

        ax[1].plot(x, self.func_unnorm(x), '-', label='Best individual')
        ax[1].plot(x, y, '+', label='Data')
        ax[1].set_title('Un-normalised model')
        ax[1].set_ylabel(r'$f(a/b)$')
        # ax[1].set_xlabel(r'')
        # ax[1].set_ylim([0, 1])
        ax[1].grid()

        fig.supxlabel(r'Crack length, $a$')
        # fig.supylabel(r'$f(a/b)$')
        ax[0].legend(loc='upper center')
        ax[1].legend(loc='upper center')
        plt.tight_layout()
        plt.show()

    def fitnessOnTraining(self, individual):
        """
        individual is the gpsr individual
        """
        fitness = individual.fitness
        print(f"Fitness on the training set: {fitness}")
        return fitness

    def mseAndR2(self, X, y):
        y_pred = self.func(X)
        MSE = np.sum((y_pred - y) ** 2) / len(y_pred)
        r2 = r2_score(y, y_pred)
        # print("MSE on the test set = ", MSE)
        # print("R^2 on the test set = ", r2)
        return MSE, r2

    def printConsts(self):
        if hasattr(self.pipeline[-1].gpsr.best, "consts"):
            print("Best parameters = ", self.pipeline[-1].gpsr.best.consts)
        return self.pipeline[-1].gpsr.best.consts

    def printTiming(self, tic, toc):
        print("Elapsed time = ", toc - tic)
        time_per_individual = (toc - tic) / (
            self.pipeline[-1].gpsr.NGEN * self.pipeline[-1].gpsr.NINDIVIDUALS *
            self.pipeline[-1].gpsr.num_islands
        )
        individuals_per_sec = 1 / time_per_individual
        print("Time per individual = ", time_per_individual)
        print("Individuals per sec = ", individuals_per_sec)
