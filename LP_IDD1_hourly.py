# coding: utf-8
"""LP scheduling optimisation using IDD1 parameters."""
########################################################################################################################
# Import required tools
########################################################################################################################

# import off-the-shelf scripting modules
from __future__ import division  # Without this, rounding errors occur in python 2.7, but apparently not in 3.4
import pandas as pd
import math
import datetime


# import raw temporal data
raw_data = pd.read_csv('n2ex_2018_12_11.csv')
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
EtoP = 4                                        # Energy to Power ratio of RFB
model.BESSCapacity = Param(initialize=EtoP*BESSRatedPower)        # kWh
model.VOCV50SOC = Param(initialize=1.4)         # OCV at 50% SOC (PNNL)
EffCoulRT = 0.964                               # Assumed constant with I
EffV = 0.763                                    # Round trip VE at 1600 A/m2 (midpoint!)
LossBOP = 0.02                                  # One way losses due to BOP, Weber et al. 2013
BESSRatedCurrentDensity = 1100                  # A/m2, current density at EffDC = 75%  (EffV*EffC*(1-LossBOP)^2)
StackArea = 1000 * BESSRatedPower / (BESSRatedCurrentDensity * model.VOCV50SOC * math.sqrt(EffV) * (1-LossBOP))
model.StackArea = Param(initialize=StackArea)
model.ASR = Param(initialize=0.0000894)         # Ohm.m2 from linear fit
model.Vfaradaic = Param(initialize=0.035)       # Faradaic potential, from extrapolation to I = 0

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
    return 3200  # In A/m2: highest current density tested for IDD2s
model.MaxDischargeCurrentDensityBESSIndexed = Param(model.I, initialize=max_discharge_current_density_bess_init)

def max_charge_current_density_bess_init(model,i):
    return 3200  # In A/m2: highest current density tested for IDD2s
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
    return 800  # In A/m2
model.CurrentDensityCharge = Var(model.I, within=NonNegativeReals, initialize=current_density_bess_charge_init)

def current_density_bess_discharge_init(model,i):
    return 800  # In A/m2
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
    return model.StackArea * model.TimeStep \
            * sum(model.PriceBM[t] *
                  ((model.VOCV50SOC * math.sqrt(EffV) * (1-LossBOP) * model.CurrentDensityDischarge[t]\
                    - (model.VOCV50SOC / (math.sqrt(EffV) * (1-LossBOP))) * model.CurrentDensityCharge[t])
                    ) for t in model.I) / 1000000
model.Objective = Objective(rule=objective_expression, sense=maximize)

opt = SolverFactory('gurobi')

results = opt.solve(model)

print(datetime.datetime.now())

########################################################################################################################
# Terminal window output of optimization results
########################################################################################################################
# In terminal window
for i in model.I:
    print(i, "%.2f" % model.CurrentDensityCharge[i].value, "%.2f" % model.CurrentDensityDischarge[i].value,
          "%.2f" % value(SOCtracker[i]))
print("Predicted revenue from", BESSRatedPower, "kW, ", value(model.BESSCapacity), "kWh system: £", "%.3f" % value(model.Objective), "under fixed efficiency assumption.")

#Actual revenue (considering real losses, as per the NLP formulation)
actual_revenue = []
for i in model.I:
    acutal_hourly_revenue = model.StackArea * model.TimeStep * (1 / 1000000) \
            * sum(model.PriceBM[t] *
                  ((model.VOCV50SOC - model.Vfaradaic) * model.CurrentDensityDischarge[i].value * (1-LossBOP)\
                  - (model.VOCV50SOC + model.Vfaradaic) * model.CurrentDensityCharge[i].value * (1/(1-LossBOP)) \
           - model.ASR * (model.CurrentDensityDischarge[i].value**2 + model.CurrentDensityCharge[i].value**2))
    actual_revenue = actual_revenue + [actual_hourly_revenue]
actual_revenue =sum(actual_revenue)
print("Actual revenue from", BESSRatedPower, "kW, ", value(model.BESSCapacity), "kWh system: £", "%.3f" % actual_revenue, "under real efficiency.")                  
                  
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
for i in model.I:
    price_data = price_data + [model.PriceBM[i]]
    dis_current_density_data = dis_current_density_data + [model.CurrentDensityDischarge[i].value]
    chg_current_density_data = chg_current_density_data + [model.CurrentDensityCharge[i].value]
    current_density_data = current_density_data + [(model.CurrentDensityCharge[i].value - model.CurrentDensityDischarge[i].value)]
    SOC_data = SOC_data + [value(SOCtracker[i])]
    half_hour_revenue = half_hour_revenue + [value(model.StackArea * model.TimeStep \
            * model.PriceBM[i] * ((model.VOCV50SOC * math.sqrt(EffV) * model.CurrentDensityDischarge[i] - (
                model.VOCV50SOC / math.sqrt(EffV)) * model.CurrentDensityCharge[i])
                                      ) / 1000000)]

# Then stick lists in pandas dataframe format and export to CSV
results_table = pd.DataFrame({"Price": price_data,
                              "Discharge Current Density": dis_current_density_data,
                              "Charge Current Density": chg_current_density_data,
                              "Current Density": current_density_data,
                              "SOC": SOC_data,
                              "Half hour revenue, £": half_hour_revenue})

results_table.to_csv('CR_LP_D_IDD1_results.csv', sep=',')
