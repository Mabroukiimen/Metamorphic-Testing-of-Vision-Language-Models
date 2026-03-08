"""
Simplified Genetic Algorithm (based on rmsolgi/geneticalgorithm) adapted for
generic optimisation: the objective function receives a population of
candidate vectors or single vectors, depending on how you implement it.
"""

import numpy as np
import sys
import matplotlib.pyplot as plt


class geneticalgorithm():
    """  Genetic Algorithm (Elitist version) for Python

    This version is generic: it does NOT assume patches, PSNR, trackers, etc.
    Your objective function is given in `function` and is called inside
    `evaluate()`.
    """

    #############################################################
    def __init__(self, function, dimension, args=(), kwargs={},
                 variable_type='bool', variable_boundaries=None,
                 variable_type_mixed=None,
                 function_timeout=10,
                 algorithm_parameters={'max_num_iteration': None,
                                       'population_size': 100,
                                       'mutation_probability': 0.1,
                                       'elit_ratio': 0.01,
                                       'crossover_probability': 0.5,
                                       'parents_portion': 0.3,
                                       'crossover_type': 'uniform',
                                       'max_iteration_without_improv': None},
                 convergence_curve=True, progress_bar=True):

        self.__name__ = geneticalgorithm

        # function: the objective function to be minimized
        #NB: this implementation minimizes the given objective function.
        assert callable(function), "function must be callable"
        self.f = function

        # dimension = length of the transformation vector / the number of decision variables (decision variables = genes)
        self.dim = int(dimension)

        # extra args / kwargs forwarded to f
        self.args = args
        self.kwargs = kwargs   #keyword arguments

        # variable type
        assert (variable_type in ['bool', 'int', 'real']), \
            "variable_type must be 'bool', 'int', or 'real'"

        # variable_type_mixed
        if variable_type_mixed is None:
            if variable_type == 'real':
                self.var_type = np.array([['real']] * self.dim)
            else:
                self.var_type = np.array([['int']] * self.dim)
        else:
            assert type(variable_type_mixed).__module__ == 'numpy', \
                "variable_type_mixed must be numpy array"
            assert len(variable_type_mixed) == self.dim, \
                "variable_type_mixed must have length == dimension"
            for i in variable_type_mixed:
                assert i in ['real', 'int'], \
                    "variable_type_mixed elements must be 'int' or 'real'"
            self.var_type = variable_type_mixed

        # boundaries
        if variable_type != 'bool' or type(variable_type_mixed).__module__ == 'numpy':
            assert type(variable_boundaries).__module__ == 'numpy', \
                "variable_boundaries must be numpy array"
            assert len(variable_boundaries) == self.dim, \
                "variable_boundaries length must equal dimension"
            for i in variable_boundaries:
                assert len(i) == 2, \
                    "each boundary must be a tuple [lower, upper]"
                assert i[0] <= i[1], \
                    "lower_boundaries must be <= upper_boundaries"
            self.var_bound = variable_boundaries
        else:
            self.var_bound = np.array([[0, 1]] * self.dim)

        # Timeout (not actively used in this version)  #if the given function does not provide output before function_timeout (unit is seconds) the algorithm raise error.
        #For example, when there is an infinite loop in the given function.
        self.funtimeout = float(function_timeout)

        # Convergence_curve: <True/False> - Plot the convergence curve or not Default is True.
        # Progress_bar <True/False> - Show progress bar or not. Default is True.
        self.convergence_curve = bool(convergence_curve)
        self.progress_bar = bool(progress_bar)
        
        
        
        #algorithm_parameters:
            #max_num_iteration <int> - stoping criteria of the genetic algorithm (GA)
            #population_size <int> - number of individuals/chromosomes/instances in each generation
            #mutation_probability <float in [0,1]> - probability of mutation for each gene in the offspring
            #elit_ration <float in [0,1]> - portion of the population to be copied as elites to the next generation; these elites are selected from the best individuals in the current generation; for example, if elit_ratio=0.01 and population_size=100, then 1 elite (the best individual) is copied to the next generation without mutation
            #crossover_probability <float in [0,1]> - probability of crossover for each pair of parents selected for mating
            #parents_portion <float in [0,1]> - portion of the population selected as parents for mating; for example, if parents_portion=0.3 and population_size=100, then 30 individuals are selected as parents to produce offspring for the next generation
            #crossover_type <string> - Default is 'uniform'; 'one_point' or 'two_point' are other options
            #max_iteration_without_improv <int> - maximum number of successive iterations without improvement. If None it is ineffective

        # algorithm params
        self.param = algorithm_parameters
        self.pop_s = int(self.param['population_size'])

        assert 0 <= self.param['parents_portion'] <= 1, \
            "parents_portion must be in [0,1]"
        self.par_s = int(self.param['parents_portion'] * self.pop_s)
        trl = self.pop_s - self.par_s
        if trl % 2 != 0:
            self.par_s += 1

        self.prob_mut = self.param['mutation_probability']
        assert 0 <= self.prob_mut <= 1, "mutation_probability must be in [0,1]"

        self.prob_cross = self.param['crossover_probability']
        assert 0 <= self.prob_cross <= 1, "crossover_probability must be in [0,1]"

        assert 0 <= self.param['elit_ratio'] <= 1, \
            "elit_ratio must be in [0,1]"
        trl = self.pop_s * self.param['elit_ratio']
        if trl < 1 and self.param['elit_ratio'] > 0:
            self.num_elit = 1
        else:
            self.num_elit = int(trl)

        assert self.par_s >= self.num_elit, \
            "number of parents must be >= number of elites"

        # iterations
        if self.param['max_num_iteration'] is None:
            self.iterate = 0
            for i in range(0, self.dim):
                if self.var_type[i] == 'int':
                    self.iterate += (self.var_bound[i][1] - self.var_bound[i][0]) \
                                    * self.dim * (100 / self.pop_s)
                else:
                    self.iterate += (self.var_bound[i][1] - self.var_bound[i][0]) \
                                    * 50 * (100 / self.pop_s)
            self.iterate = int(self.iterate)
            if (self.iterate * self.pop_s) > 10000000:
                self.iterate = 10000000 / self.pop_s
        else:
            self.iterate = int(self.param['max_num_iteration'])

        # crossover type
        self.c_type = self.param['crossover_type']
        assert self.c_type in ['uniform', 'one_point', 'two_point'], \
            "crossover_type must be 'uniform', 'one_point', or 'two_point'"

        # stopping based on no improvement
        if self.param['max_iteration_without_improv'] is None:
            self.mniwi = self.iterate + 1
        else:
            self.mniwi = int(self.param['max_iteration_without_improv'])
        self.stop_mniwi = False

    #############################################################
    def run(self):

        # variable indices by type
        self.integers = np.where(self.var_type == 'int')
        self.reals = np.where(self.var_type == 'real')

        # population: last column is fitness
        pop = np.zeros((self.pop_s, self.dim + 1))
        var = np.zeros((self.pop_s, self.dim))

        # initialize population
        for i in self.integers[0]:
            var[:, i] = np.random.randint(
                self.var_bound[i][0],
                self.var_bound[i][1] + 1,
                size=self.pop_s    #self.pop_s = population size
            )
        for i in self.reals[0]:
            var[:, i] = self.var_bound[i][0] + np.random.random(
                size=self.pop_s
            ) * (self.var_bound[i][1] - self.var_bound[i][0])                          #self.var_bound = lower/upper bound for each gene

        obj = self.evaluate(var)
        pop[:, :self.dim] = var   #self.dim = dimension (= length of the transformation vector)
        pop[:, self.dim] = obj

        # reporting
        self.report = []
        self.best_variable = pop[0, :self.dim].copy()
        self.best_function = pop[0, self.dim]
        self.test_obj = self.best_function

        t = 1
        counter = 0

        while t <= self.iterate:
            if self.progress_bar:
                self.progress(t, self.iterate, status="GA is running...")

            # sort by fitness (ascending)
            pop = pop[pop[:, self.dim].argsort()]
            #Proof that it's a minimizer: lower fitness is better
            if pop[0, self.dim] < self.best_function:
                self.best_function = pop[0, self.dim].copy()
                self.best_variable = pop[0, :self.dim].copy()
                counter = 0
            else:
                counter += 1

            self.report.append(pop[0, self.dim])

            # normalizing objective function
            minobj = pop[0, self.dim]
            if minobj < 0:
                normobj = pop[:, self.dim] + abs(minobj)
            else:
                normobj = pop[:, self.dim].copy()

            maxnorm = np.max(normobj)
            normobj = maxnorm - normobj + 1

            # selection probabilities
            sum_normobj = np.sum(normobj)
            prob = normobj / sum_normobj
            cumprob = np.cumsum(prob)

            # select parents
            par = np.zeros((self.par_s, self.dim + 1))
            par[:self.num_elit] = pop[:self.num_elit].copy()
            index = np.searchsorted(
                cumprob,
                np.random.random(size=self.par_s - self.num_elit)
            )
            par[self.num_elit:self.par_s] = pop[index].copy()

            par_count = 0
            while par_count == 0:
                ef_par_list = np.random.choice(
                    [True, False],
                    (self.par_s,),
                    p=[self.prob_cross, 1 - self.prob_cross]
                )
                if np.any(ef_par_list):
                    par_count += 1
            ef_par = par[ef_par_list].copy()

            # new generation
            num_children_pairs = int(np.round((self.pop_s - self.par_s) / 2))
            mutations = np.zeros((num_children_pairs, self.dim + 1))
            indices = np.arange(self.par_s, self.pop_s, 2)
            pop_new = np.zeros_like(pop)
            pop_new[:self.par_s] = par.copy()

            r1 = np.random.randint(0, par_count, size=num_children_pairs)
            r2 = np.random.randint(0, par_count, size=num_children_pairs)
            pvar1 = ef_par[r1, :self.dim].copy()
            pvar2 = ef_par[r2, :self.dim].copy()

            ch = self.cross(pvar1, pvar2, self.c_type)
            ch1 = ch[0].copy()
            ch2 = ch[1].copy()

            ch1 = self.mut(ch1)
            ch2 = self.mutmidle(ch2, pvar1, pvar2)

            # evaluate children 1
            obj1 = self.evaluate(ch1)
            mutations[:, :self.dim] = ch1
            mutations[:, self.dim] = obj1
            pop_new[indices] = mutations.copy()

            # evaluate children 2
            obj2 = self.evaluate(ch2)
            mutations[:, :self.dim] = ch2
            mutations[:, self.dim] = obj2
            pop_new[indices + 1] = mutations.copy()

            pop = pop_new

            if counter > self.mniwi:
                if self.progress_bar:
                    self.progress(t, self.iterate,
                                  status="GA is running (no improv stop)...")
                self.stop_mniwi = True
                break

            t += 1

        # final sort
        pop = pop[pop[:, self.dim].argsort()]
        if pop[0, self.dim] < self.best_function:
            self.best_function = pop[0, self.dim].copy()
            self.best_variable = pop[0, :self.dim].copy()

        self.report.append(pop[0, self.dim])

        self.output_dict = {
            'variable': self.best_variable,
            'function': self.best_function
        }

        if self.progress_bar:
            show = ' ' * 100
            sys.stdout.write('\r%s' % (show))
        sys.stdout.write('\r The best solution found:\n %s' % self.best_variable)
        sys.stdout.write('\n\n Objective function:\n %s\n' % self.best_function)
        sys.stdout.flush()

        if self.convergence_curve:
            re = np.array(self.report)
            plt.plot(re)
            plt.xlabel('Iteration')
            plt.ylabel('Objective function')
            plt.title('Genetic Algorithm')
            plt.show()

        return pop

    ##################################################################
    def cross(self, x, y, c_type):

        ofs1 = x.copy()
        ofs2 = y.copy()

        if c_type == 'one_point':
            ran = np.random.randint(0, self.dim, size=x.shape[0])
            for i, el in enumerate(ran):
                ofs1[i, :el] = y[i, :el].copy()
                ofs2[i, :el] = x[i, :el].copy()

        if c_type == 'two_point':
            ran1 = np.random.randint(0, self.dim, size=x.shape[0])
            ran2 = np.random.randint(ran1, self.dim)
            for i in range(len(ran1)):
                ofs1[i, ran1[i]:ran2[i]] = y[i, ran1[i]:ran2[i]].copy()
                ofs2[i, ran1[i]:ran2[i]] = x[i, ran1[i]:ran2[i]].copy()

        if c_type == 'uniform':
            ran = np.random.choice(
                [True, False],
                (x.shape[0], self.dim),
                p=[0.5, 0.5]
            )
            ofs1[ran] = y[ran].copy()
            ofs2[ran] = x[ran].copy()

        return np.array([ofs1, ofs2])

    ##################################################################
    def mut(self, x):
        mask = np.random.choice(
            [True, False],
            (x.shape[0], self.dim),
            p=[self.prob_mut, 1 - self.prob_mut]
        )
        for i in range(self.dim):
            indices = np.where(mask[:, i] == True)[0]
            if i in self.integers[0]:
                x[indices, i] = np.random.randint(
                    self.var_bound[i][0],
                    self.var_bound[i][1] + 1,
                    size=len(indices)
                )
            else:
                x[indices, i] = self.var_bound[i][0] + np.random.random(
                    size=len(indices)
                ) * (self.var_bound[i][1] - self.var_bound[i][0])
        return x

    ##################################################################
    def mutmidle(self, x, p1, p2):
        mask = np.random.choice(
            [True, False],
            (x.shape[0], self.dim),
            p=[self.prob_mut, 1 - self.prob_mut]
        )
        for i in range(self.dim):
            idx1 = np.where((mask[:, i] == True) & (p1[:, i] < p2[:, i]))
            idx2 = np.where((mask[:, i] == True) & (p1[:, i] > p2[:, i]))
            idx3 = np.where((mask[:, i] == True) & (p1[:, i] == p2[:, i]))

            if i in self.integers[0]:
                if len(idx1[0]) > 0:
                    x[idx1[0], i] = np.random.randint(
                        p1[idx1[0], i], p2[idx1[0], i] + 1, size=len(idx1[0])
                    )
                if len(idx2[0]) > 0:
                    x[idx2[0], i] = np.random.randint(
                        p2[idx2[0], i], p1[idx2[0], i] + 1, size=len(idx2[0])
                    )
                if len(idx3[0]) > 0:
                    x[idx3[0], i] = np.random.randint(
                        self.var_bound[i][0],
                        self.var_bound[i][1] + 1,
                        size=len(idx3[0])
                    )
            else:
                if len(idx1[0]) > 0:
                    x[idx1, i] = p1[idx1, i] + np.random.random(
                        size=len(idx1[0])
                    ) * (p2[idx1, i] - p1[idx1, i])
                if len(idx2[0]) > 0:
                    x[idx2, i] = p2[idx2, i] + np.random.random(
                        size=len(idx2[0])
                    ) * (p1[idx2, i] - p2[idx2, i])
                if len(idx3[0]) > 0:
                    x[idx3, i] = self.var_bound[i][0] + np.random.random(
                        size=len(idx3[0])
                    ) * (self.var_bound[i][1] - self.var_bound[i][0])

        return x

    ##################################################################
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """

        X: shape (population_size, dim)
        each row X[i] is one chromosome (one transformation vector)
        returns: shape (population_size,)
        """
        fx = np.zeros((X.shape[0],), dtype=float) #Creates an array fx to store fitness values
        for i in range(X.shape[0]):
            fx[i] = self.f(X[i], *self.args, **self.kwargs)
        return fx
    #So output fx is a list of fitness scores, one per chromosome, used by the GA to decide which individuals are better and should be selected as parents.

    ##################################################################
    def progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))
        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)
        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()
