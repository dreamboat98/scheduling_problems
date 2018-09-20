# coding: utf-8
"""This is a scheduling optimisation problem with a quadratic objective function, capturing the non linear relationship
between power and current due to ohmic losses in the cell. The objective is to maximise revenue from
BM price arbitrage."""
########################################################################################################################
# Import required tools
########################################################################################################################

# import off-the-shelf scripting modules
from __future__ import division  # Without this, rounding errors occur in python 2.7, but apparently not in 3.4
import pandas as pd
import math
import datetime


# import raw temporal data
raw_data = pd.read_csv('https://github.com/diarmidr/scheduling_problems/blob/master/n2ex_2018_12_11.csv')
price = [float(x) for x in raw_data.ix[:, 'price_dollar']]


# import PYOMO

from pyomo.environ import *
from pyomo.opt import SolverFactory

print(datetime.datetime.now())
########################################################################################################################
# PYOMO model starts here
########################################################################################################################

model = ConcreteModel()

########################################################################################################################
# scalar BESS parameters - Based on Reed et al. 2016). IDD2s config at 400 ml/min flow rate
########################################################################################################################

# BESS
BESSRatedPower = 1                              # kW, discharge power that gives 75% stack EERT
EtoP = 0.5                                      # Energy to Power ratio of RFB
model.BESSCapacity = Param(initialize=EtoP*BESSRatedPower)        # kWh
model.ASR = Param(initialize=0.0000538)         # Ohm.m2 from linear fit
model.VOCV50SOC = Param(initialize=1.4)         # OCV at 50% SOC (PNNL)
model.Vfaradaic = Param(initialize=0.030)       # Faradaic potential, from extrapolation to I = 0
EffCoulRT = 0.975                               # Assumed constant with I
LossBOP = 0.02                                  # One way losses due to BOP, Weber et al. 2013
BESSRatedCurrentDensity = 2188                  # A/m2, current density at EffDC = 75%  (EffV*EffC*(1-LossBOP))
StackArea = 1000 * BESSRatedPower / (BESSRatedCurrentDensity * model.VOCV50SOC * (1-LossBOP))
model.StackArea = Param(initialize=StackArea)

# Data-set
model.TimeStep = Param(initialize=1)          # Hours

########################################################################################################################
# indexed parameters
########################################################################################################################

# Index
model.I = range(len(price))

# Input data

def init_priceBM(model,t):
    return price[t]
model.PriceBM = Param(model.I, initialize=init_priceBM)

#  For constraints

def max_discharge_current_density_bess_init(model,i):
    return 3200  # In A/m2
model.MaxDischargeCurrentDensityBESSIndexed = Param(model.I, initialize=max_discharge_current_density_bess_init)

def max_charge_current_density_bess_init(model,i):
    return 3200  # In A/m2
model.MaxChargeCurrentDensityBESSIndexed = Param(model.I, initialize=max_charge_current_density_bess_init)

def min_soc_bess_init(model,i):
    return 0.15
model.MinSOCBESSIndexed = Param(model.I, initialize=min_soc_bess_init)

def max_soc_bess_init(model,i):
    return 0.85
model.MaxSOCBESSIndexed = Param(model.I, initialize=max_soc_bess_init)

def eff_coul_RT_init(model,i):
    return EffCoulRT
model.EffCoulRTIndexed = Param(model.I, initialize=eff_coul_RT_init)


########################################################################################################################
# variables
########################################################################################################################

def current_density_bess_charge_init(model,i):
    return 1600  # In A/m2
model.CurrentDensityCharge = Var(model.I, within=NonNegativeReals, initialize=current_density_bess_charge_init)

def current_density_bess_discharge_init(model,i):
    return 1600  # In A/m2
model.CurrentDensityDischarge = Var(model.I, within=NonNegativeReals, initialize=current_density_bess_charge_init)

########################################################################################################################
# constraints
########################################################################################################################

#BESS

def max_charge_current_density_rule(model,t):
    return model.CurrentDensityCharge[t] - model.MaxChargeCurrentDensityBESSIndexed[t] <= 0
model.MaxChargeCurrentDensityBESSConstraint = Constraint(model.I, rule=max_charge_current_density_rule)

def max_discharge_current_density_rule(model,t):
    return model.CurrentDensityDischarge[t] - model.MaxDischargeCurrentDensityBESSIndexed[t] <= 0
model.MaxDischargeCurrentDensityBESSConstraint = Constraint(model.I, rule=max_discharge_current_density_rule)

# The below loop generates a list of SOC values to which constraints are applied
model.SOCAtStart = Param(initialize=0.5)
SOCtracker = []
SOC = model.SOCAtStart
for i in model.I:
    SOC_increment = model.StackArea * model.TimeStep\
                    * (model.CurrentDensityCharge[i] * math.sqrt(model.EffCoulRTIndexed[i])
                        - model.CurrentDensityDischarge[i] / math.sqrt(model.EffCoulRTIndexed[i]))\
                        / (1000 * model.BESSCapacity / model.VOCV50SOC)
    SOC = SOC + SOC_increment
    SOCtracker = SOCtracker + [SOC]  # This appends the ith SOC value to the set

# These constraints are defined w.r.t to the above set of SOC values

def min_soc_rule(model,t):
    return SOCtracker[t] - model.MinSOCBESSIndexed[t] >= 0
model.MinSOCBESSConstraint = Constraint(model.I, rule=min_soc_rule)

def max_soc_rule(model,t):
    return SOCtracker[t] - model.MaxSOCBESSIndexed[t] <= 0
model.MaxSOCBESSConstraint = Constraint(model.I, rule=max_soc_rule)

model.SOCAtEnd = SOCtracker[len(model.I)-1]  # The below constraint is not indexed - hence a function doesn't work
model.BESSEnergyConservation = Constraint(expr=model.SOCAtStart - model.SOCAtEnd == 0)

########################################################################################################################
# objective
########################################################################################################################
def objective_expression(model):
    return model.StackArea * model.TimeStep * (1 / 1000000) \
            * sum(model.PriceBM[t] * ((model.VOCV50SOC - model.Vfaradaic) * model.CurrentDensityDischarge[t] \
                  - (model.VOCV50SOC + model.Vfaradaic)* model.CurrentDensityCharge[t] \
           - model.ASR * (model.CurrentDensityDischarge[t]**2 + model.CurrentDensityCharge[t]**2)) for t in model.I)
model.Objective = Objective(rule=objective_expression, sense=maximize)

opt = SolverFactory('gurobi')

results = opt.solve(model)

print(datetime.datetime.now())  # For measurement of solve time
########################################################################################################################
# Terminal window output of optimization results
########################################################################################################################
for i in model.I:
    print(i, "%.2f" % model.CurrentDensityCharge[i].value, "%.2f" % model.CurrentDensityDischarge[i].value,
          "%.2f" % value(SOCtracker[i]))
print("Revenue from", BESSRatedPower, "kW, ", value(model.BESSCapacity), "kWh system: $", "%.2f" % value(model.Objective))

########################################################################################################################
# CSV output of optimization results
########################################################################################################################
# First convert indexed parameters and variables to lists:
price_data = []
dis_current_density_data = []
chg_current_density_data = []
current_density_data = []
SOC_data = []
half_hour_revenue = []
power_output = []
for i in model.I:
    price_data = price_data + [model.PriceBM[i]]
    dis_current_density_data = dis_current_density_data + [model.CurrentDensityDischarge[i].value]
    chg_current_density_data = chg_current_density_data + [model.CurrentDensityCharge[i].value]
    current_density_data = current_density_data + [(model.CurrentDensityCharge[i].value - model.CurrentDensityDischarge[i].value)]
    SOC_data = SOC_data + [value(SOCtracker[i])]
    half_hour_revenue = half_hour_revenue + [value(model.StackArea * model.TimeStep * (1 / 1000000)
            * model.PriceBM[i] * ((model.VOCV50SOC - model.Vfaradaic) * model.CurrentDensityDischarge[i]
                  - (model.VOCV50SOC + model.Vfaradaic)* model.CurrentDensityCharge[i]
           - model.ASR * (model.CurrentDensityDischarge[i]**2 + model.CurrentDensityCharge[i]**2))
    )]
    power_output = power_output + [value(model.StackArea *
                    ((model.VOCV50SOC - model.Vfaradaic) * model.CurrentDensityDischarge[i]
                    - (model.VOCV50SOC + model.Vfaradaic)* model.CurrentDensityCharge[i]
                    - model.ASR * (model.CurrentDensityDischarge[i]**2 + model.CurrentDensityCharge[i]**2))
    )]
# Then stick lists in pandas dataframe format and export to CSV
results_table = pd.DataFrame({"Price": price_data,
                              "Discharge Current Density": dis_current_density_data,
                              "Charge Current Density": chg_current_density_data,
                              "Current Density": current_density_data,
                              "SOC": SOC_data,
                              "Power (W)": power_output,
                              "Half hour revenue, £": half_hour_revenue})

results_table.to_csv('CR_NLP_IDD2s_results.csv', sep=',')
