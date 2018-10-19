# coding: utf-8
"""This script performs the same optimisation process as NLP_IDD2s, but within a loop so that a daily revenue is returned
 over a longer timescale. Author: Diarmid Roberts."""
########################################################################################################################
# Import required tools
########################################################################################################################

# import off-the-shelf scripting modules
from __future__ import division     # Without this, rounding errors occur in python 2.7, but apparently not in 3.4
import pandas as pd                 # CSV handling
import datetime                     # For time stamping optimisation process

# import PYOMO

from pyomo.environ import *
from pyomo.opt import SolverFactory

years = [2017]
print(datetime.datetime.now())      # Time stamp start of optimisation
for i in years:
    #raw_data = pd.read_csv('clean_n2ex_' + str(i) + '_hourly.csv')
    raw_data = pd.read_csv('clean_n2ex_2017_hourly_Q_not_PSD_days_removed.csv')
    price_data = raw_data.ix[:, 'price_sterling'].tolist()
    price = [float(x) for x in price_data]
    daily_revenue_list = []   # Empty receptacle for daily revenue
    # Below loop goes through data 24h at a time
    for i in range(int(len(price)/24)):
        price_in_period = price[i*24:(i+1)*24]

        ####################################################################################################################
        # PYOMO model starts here.
        ####################################################################################################################

        model = ConcreteModel()

        ####################################################################################################################
        # Scalar parameters (parameters used directly in the optimisation problem formulation are given the PYOMO Param class.)
        ####################################################################################################################
        # Intrinsic parameters (taken from Reed et al. 2016, except VOCV50SOC and LossBOP, taken from Kim 2011 and Weber 2013.
        model.VOCV50SOC = Param(initialize=1.46)               # OCV at 50% SOC (PNNL)
        model.LossBOP = Param(initialize=0.02)                 # One way fractional power losses due to BOP, Weber et al. 2013
        model.EffCoul = Param(initialize=0.975)                # Assumed constant with I
        model.ASR = Param(initialize=0.0000538)                # Ohm.m2 from linear fit on over-potential v current density
        model.Vfaradaic = Param(initialize=0.030)              # Faradaic over-potential, y axis intercept of linear fit.
        BESSMaxCurrentDensity = 3200                           # Maximum at which polarization curve is still linear
        model.VMax = 1.65                                      # Max permitted cell voltage
        a = 0.2667					                           # Gradient of OCV v SOC line
        b = 1.333					                           # SOC = 0 intercept of OCV v SOC line

        # RFB system sizing for absolute output of scheduling problem is done as follows:

        # 1: Define power rating (kW)
        BESSRatedPower = 1

        # 2: From experimental data, find VE that gives EffDC ( = EffV*EffCoul*(1-LossBOP)^2 ) >= 0.75 (round-trip)
        EffVRated = 0.801

        # 3: Identify current density that corresponds to above EffV
        BESSRatedCurrentDensity = 2188                          # A/m2, current reaching external stack terminals

        # 4: Set stack area such that system can output 1kW DC despite voltaic and balance of plant losses.
        model.StackArea = Param(initialize=1000 * BESSRatedPower / (BESSRatedCurrentDensity * model.VOCV50SOC *
                                                                    sqrt(EffVRated) * (1 - model.LossBOP)))

        # 5: Set energy to power ratio (a.k.a discharge time)
        EtoP = 4

        # 6: Calculate required coulombic capacity, considering coulombic losses during discharge.
        model.BESSCapacity = Param(initialize=BESSRatedCurrentDensity * EtoP * model.StackArea / sqrt(model.EffCoul))

        # Data-set
        model.TimeStep = Param(initialize=1)  # Hours

        ####################################################################################################################
        # indexed parameters
        ####################################################################################################################

        # Index
        model.I = range(len(price_in_period))

        # Input price data


        def init_price(model, t):
            return price_in_period[t]


        model.Price = Param(model.I, initialize=init_price)

        #  For constraints


        def max_discharge_current_density_bess_init(model, i):
            return BESSMaxCurrentDensity


        model.MaxDischargeCurrentDensityBESSIndexed = Param(model.I, initialize=max_discharge_current_density_bess_init)


        def max_charge_current_density_bess_init(model, i):
            return BESSMaxCurrentDensity


        model.MaxChargeCurrentDensityBESSIndexed = Param(model.I, initialize=max_charge_current_density_bess_init)


        def min_soc_bess_init(model, i):
            return 0.15


        model.MinSOCBESSIndexed = Param(model.I, initialize=min_soc_bess_init)


        def max_soc_bess_init(model, i):
            return 0.85


        model.MaxSOCBESSIndexed = Param(model.I, initialize=max_soc_bess_init)


        def eff_coul_RT_init(model, i):
            return model.EffCoul


        model.EffCoulIndexed = Param(model.I, initialize=eff_coul_RT_init)

        def max_cell_voltage_indexed(model, i):
            return model.VMax


        model.VMaxIndexed = Param(model.I, initialize=max_cell_voltage_indexed)



        ####################################################################################################################
        # variables
        ####################################################################################################################

        def current_density_bess_charge_init(model, i):
            return 800  # In A/m2


        model.CurrentDensityCharge = Var(model.I, within=NonNegativeReals, initialize=current_density_bess_charge_init)


        def current_density_bess_discharge_init(model, i):
            return 800  # In A/m2


        model.CurrentDensityDischarge = Var(model.I, within=NonNegativeReals, initialize=current_density_bess_charge_init)


        ####################################################################################################################
        # constraints
        ####################################################################################################################

        # BESS

        def max_charge_current_density_rule(model, t):
            return model.CurrentDensityCharge[t] - model.MaxChargeCurrentDensityBESSIndexed[t] <= 0


        model.MaxChargeCurrentDensityBESSConstraint = Constraint(model.I, rule=max_charge_current_density_rule)


        def max_discharge_current_density_rule(model, t):
            return model.CurrentDensityDischarge[t] - model.MaxDischargeCurrentDensityBESSIndexed[t] <= 0


        model.MaxDischargeCurrentDensityBESSConstraint = Constraint(model.I, rule=max_discharge_current_density_rule)

        # The below loop generates a list of SOC values to which constraints are subsequently applied
        model.SOCAtStart = Param(initialize=0.5)
        SOCtracker = []
        SOC = model.SOCAtStart
        for i in model.I:
            SOC_increment = model.StackArea * model.TimeStep \
                            * (model.CurrentDensityCharge[i] * sqrt(model.EffCoulIndexed[i])
                               - model.CurrentDensityDischarge[i] / sqrt(model.EffCoulIndexed[i])) \
                            / model.BESSCapacity
            SOC = SOC + SOC_increment
            SOCtracker = SOCtracker + [SOC]  # This appends the ith SOC value to the list


        # These constraints are defined w.r.t to the above set of SOC values

        def min_soc_rule(model, t):
            return SOCtracker[t] - model.MinSOCBESSIndexed[t] >= 0


        model.MinSOCBESSConstraint = Constraint(model.I, rule=min_soc_rule)


        def max_soc_rule(model, t):
            return SOCtracker[t] - model.MaxSOCBESSIndexed[t] <= 0

        model.MaxSOCBESSConstraint = Constraint(model.I, rule=max_soc_rule)

        model.SOCAtEnd = SOCtracker[len(model.I) - 1]  # The below constraint is not indexed - hence a function doesn't work
        model.BESSEnergyConservation = Constraint(expr=model.SOCAtStart - model.SOCAtEnd == 0)

        # The below code generates a cell voltage list (average across sub-period) which may be constrained, or used in the 	objective function.

        # First, the average SOC across each sub-period is calculated

        mid_point_SOC_exc_initial = []  # Need to perform separate calc for the first sub-period
        for i in range(1, len(SOCtracker)):
            mid_point_SOC = (SOCtracker[i - 1] + SOCtracker[i]) / 2
            mid_point_SOC_exc_initial = mid_point_SOC_exc_initial + [mid_point_SOC]
            mid_point_SOC_data = [(model.SOCAtStart + SOCtracker[0]) / 2] + \
                                    mid_point_SOC_exc_initial

        # Then the SOC is used to calculate the average OCV in each sub-period
        OCV_data = [a * i + b for i in mid_point_SOC_data]  # From OCV vs. SOC in Kim2011
	
	#The below code generates a constraint on the working cell voltage during charging 
	#First, the cell voltage is calculated as the sum of the OCV and the over-potential
        VCelltracker = []
        for i in model.I:
    	    VCell = OCV_data[i] + model.Vfaradaic + model.ASR * model.CurrentDensityCharge[i]
    	    VCelltracker = VCelltracker + [VCell]

        def max_cell_voltage_rule(model, t):
            return VCelltracker[t] <= model.VMaxIndexed[i]

        model.MaxVCellConstraint = Constraint(model.I, rule=max_cell_voltage_rule)

        ####################################################################################################################
        # objective
        ####################################################################################################################
        def objective_expression(model):
    	    return model.StackArea * model.TimeStep * (1 / 1000000) \
                            * sum((model.Price[t] *
                                ((OCV_data[t] - model.Vfaradaic) * model.CurrentDensityDischarge[t] *
                                    (1 - model.LossBOP) -
                                    (OCV_data[t] + model.Vfaradaic) * model.CurrentDensityCharge[t] *
                                    (1 / (1 - model.LossBOP))
                                    - model.ASR * (model.CurrentDensityDischarge[t] ** 2 +
                                                   model.CurrentDensityCharge[t] ** 2))) for t in model.I)

        model.Objective = Objective(rule=objective_expression, sense=maximize)

        opt = SolverFactory('gurobi')

        results = opt.solve(model)

        ####################################################################################################################
        # Terminal window output of optimization results
        ####################################################################################################################
        daily_revenue = value(model.Objective)
        print(daily_revenue)
        daily_revenue_list = daily_revenue_list + [daily_revenue]
    #Writes list of daily revenue for year i to a CSV file
    daily_revenue_output = pd.DataFrame({"Revenue, £": daily_revenue_list})
    daily_revenue_output.to_csv(str(int(i)) + "daily_revenue_NLP_IDD2s.csv", sep=',')
    annual_revenue = sum(daily_revenue_list)
    print("Annual revenue: £", str(annual_revenue))
print(datetime.datetime.now())

