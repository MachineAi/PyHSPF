# autocalibrator.py
#
# David J. Lampert (djlampert@gmail.com)
#
# Contains the AutoCalibrator class that can be used to calibrate a model.
# The class requires and HSPFModel class, start and end dates, and an output
# location to work in while running simulations. The primary function is
# autocalibrate, and it takes a list of HSPF variables, perturbations (as a
# percentage, optimization parameter, and flag for parallelization as 
# keyword arguments. The calibration routine can be summarized as follows:
# 1. Set up a series of simulations with a small perturbation to the current
#    parameter values for the parameters of interest
# 2. Make copies of the input HSPFModel and adjust the parameter values
# 3. Run the simulations and get the effect of the optimization parameter
# 4. Adjust the baseline parameter values if they improve performance
# 5. Repeat until a maximum is achieved.

# The class should be adaptable to other methodologies.

import os, pickle, datetime

from multiprocessing import Pool, cpu_count
from pyhspf.core import HSPFModel, Postprocessor

class AutoCalibrator:
    """Autocalibrates an HSPF model."""

    def __init__(self, hspfmodel, start, end, output):

        self.hspfmodel = hspfmodel
        self.start     = start
        self.end       = end
        self.output    = output

    def copymodel(self, name):
        """Returns a copy of the HSPFModel."""

        model = HSPFModel()
        model.build_from_existing(self.hspfmodel, name)

        for f in self.hspfmodel.flowgages:
            start, tstep, data = self.hspfmodel.flowgages[f]
            model.add_timeseries('flowgage', f, start, data, tstep = tstep)

        for p in self.hspfmodel.precipitations: 
            start, tstep, data = self.hspfmodel.precipitations[p]
            model.add_timeseries('precipitation', p, start, data, tstep = tstep)

        for e in self.hspfmodel.evaporations: 
            start, tstep, data = self.hspfmodel.evaporations[e]
            model.add_timeseries('evaporation', e, start, data, tstep = tstep)

        for tstype, identifier in self.hspfmodel.watershed_timeseries.items():

            model.assign_watershed_timeseries(tstype, identifier)

        for tstype, d in self.hspfmodel.subbasin_timeseries.items():

            for subbasin, identifier in d.items():
                
                if subbasin in model.subbasins:

                    model.assign_subbasin_timeseries(tstype, subbasin, 
                                                     identifier)

        return model

    def adjust(self, model, variable, adjustment):
        """Adjusts the values of the given parameter by the adjustment.""" 

        if variable == 'LZSN':
            for p in model.perlnds: p.LZSN   *= adjustment
        if variable == 'UZSN':
            for p in model.perlnds: p.UZSN   *= adjustment
        if variable == 'LZETP':
            for p in model.perlnds: p.LZETP  *= adjustment
        if variable == 'INFILT':
            for p in model.perlnds: p.INFILT *= adjustment
        if variable == 'INTFW':
            for p in model.perlnds: p.INTFW  *= adjustment
        if variable == 'IRC':
            for p in model.perlnds: p.IRC    *= adjustment
        if variable == 'AGWRC':
            for p in model.perlnds: p.AGWRC  *= adjustment
    
    def run(self, 
            model,
            targets = ['reach_outvolume',
                       'groundwater',
                       ],
            start = None,
            end = None,
            verbose = False,
            ):
        """Creates a copy of the base model, adjusts a parameter value, runs
        the simulation, calculates and returns the perturbation."""

        # if dates provided use, otherwise use default
        
        if start is None: start = self.start
        if end   is None: end   = self.end

        # build the input files and run

        
        model.build_wdminfile()
        model.build_uci(targets, start, end, hydrology = True)
        model.run(verbose = verbose)

        # get the regression information using the postprocessor

        p = Postprocessor(model, (start, end))
        p.get_calibration()
        p.calculate_errors(output = None, verbose = False)

        dr2, logdr2, dNS, logdNS, mr2, logmr2, mN2, logMS = p.regression

        p.close()

        model = None
        p     = None
        perturbation = None

        if self.optimization == 'Nash-Sutcliffe Product':    return dNS * logdNS
        if self.optimization == 'Nash-Sutcliffe Efficiency': return dNS

    def simulate(self, simulation):
        """Performs a simulation and returns the optimization value."""

        name, perturbation, adjustments = simulation

        # create a copy of the original HSPFModel to modify

        filename = '{}/{}{:4.3f}'.format(self.output, name, perturbation)

        model = self.copymodel(filename)
        model.add_hydrology()
                                 
        # adjust the values of the parameters

        for variable, adjustment in zip(self.variables, adjustments):
            self.adjust(model, variable, adjustment)

        # run and pass back the result
                                 
        return self.run(model)

    def perturb(self, parallel):
        """Performs the perturbation analysis."""

        print('perturbing the model\n')

        # adjust the parameter values for each variable for each simulation

        its = range(len(self.variables)), self.variables, self.perturbations
        adjustments = []
        for i, v, p in zip(*its):
            adjustment = self.values[:]
            adjustment[i] += p
            adjustments.append(adjustment)
                                 
        # run a baseline simulation and perturbation simulations for 
        # each of calibration variables

        its = self.variables, self.perturbations, adjustments
        simulations = ([['baseline', 0, self.values]] + 
                       [[v, p, a] for v, p, a in zip(*its)])

        if parallel:

            # create a pool of workers equal to one less than the total CPUs

            n = 4 * cpu_count()
            with Pool(cpu_count(), maxtasksperchild = n) as pool:

                # run the simulations to get the optimization parameter values

                results = pool.map_async(self.simulate, simulations)
                optimizations = results.get(timeout = 100)

        else:

            # run the simulations to get the optimization parameter values

            optimizations = [self.simulate(s) for s in simulations]

        # calculate the sensitivities for the perturbations

        sensitivities = [o - optimizations[0] for o in optimizations[1:]]

        # save the current value of the optimization parameter

        self.value = optimizations[0]

        return sensitivities

    def get_default(self, variable):
        """Gets the default value of the perturbation for the variable."""

        if   variable == 'LZSN':   return 0.05
        elif variable == 'UZSN':   return 0.05
        elif variable == 'LZETP':  return 0.02
        elif variable == 'INFILT': return 0.05
        elif variable == 'INTFW':  return 0.01
        elif variable == 'IRC':    return 0.05
        elif variable == 'AGWRC':  return 0.005
        else:
            print('error: unknown variable specified\n')
            raise

    def optimize(self, parallel):
        """Optimizes the objective function for the parameters."""

        current = self.value - 1
        t = 'increasing {:6s} {:5.1%} increases {} {:6.3f}'
        while current < self.value:

            # update the current value

            current = self.value

            print('\ncurrent optimization value: {:4.3f}\n'.format(self.value))

            # perturb the values positively

            sensitivities = self.perturb(parallel)

            # iterate through the calibration variables and update if they
            # improve the optimization parameter

            for i in range(len(self.values)):

                if sensitivities[i] > 0: self.values[i] += self.perturbations[i]
           
                its = (self.variables[i], self.perturbations[i], 
                       self.optimization, sensitivities[i])
                print(t.format(*its))

            # perturb the values negatively

            print('\ncurrent optimization value: {:4.3f}\n'.format(self.value))

            self.perturbations = [-p for p in self.perturbations]
            sensitivities = self.perturb(parallel)

            # iterate through the calibration variables and update if they
            # improve the optimization parameter

            for i in range(len(self.values)):

                if sensitivities[i] > 0:

                    self.values[i] = round(self.values[i] + 
                                           self.perturbations[i], 4)
           
                its = (self.variables[i], self.perturbations[i], 
                       self.optimization, sensitivities[i])
                print(t.format(*its))

            self.perturbations = [-p for p in self.perturbations]

    def autocalibrate(self, 
                      output,
                      variables = ['LZSN',
                                   'UZSN',
                                   'LZETP',
                                   'INFILT',
                                   'INTFW',
                                   'IRC',
                                   'AGWRC',
                                   ],
                      optimization = 'Nash-Sutcliffe Efficiency',
                      perturbations = [4, 2, 1, 0.5],
                      parallel = True,
                      ):
        """Autocalibrates the hydrology for the hspfmodel by modifying the 
        values of the HSPF PERLND parameters contained in the vars list."""

        # set up the current values of the variables, the amount to perturb
        # them by in each iteration, and the optimization parameter

        self.variables    = variables
        self.values       = [1. for v in variables]
        self.optimization = optimization

        # current value of the optimization parameter

        self.value = -10 

        # perturb until reaching a maximum (start with large perturbations)

        for p in perturbations:

            self.perturbations = [p * self.get_default(v) for v in variables]
            self.optimize(parallel)

        print('optimization complete, saving model\n')

        model = self.copymodel('calibrated')
        model.add_hydrology()
                                 
        # adjust the values of the parameters

        for variable, adjustment in zip(self.variables, self.values):
            self.adjust(model, variable, adjustment)

        with open(output, 'wb') as f: pickle.dump(model, f)
