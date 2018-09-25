# coding: utf-8
"""This script optimises a RFB schedule against electrical price data to maximise revenue using linear programming.
The LP approach requires an assumption of fixed efficiency w.r.t. power input/output. At the end of the script, the
revenue predicted under this assumption is compared to the actual revenue that would result once real voltaic losses
are factored in. Author: Diarmid Roberts."""
########################################################################################################################
# Import required tools
########################################################################################################################

# import off-the-shelf scripting modules
from __future__ import division     # Without this, rounding errors occur in python 2.7, but apparently not in 3.4
import pandas as pd                 # CSV handling
import datetime                     # For time stamping optimisation process

# import raw temporal data
raw_data = pd.read_csv('high_revenue_day_2017-10-01.csv')
price = [float(x) for x in raw_data.ix[:, 'price_sterling']]

# import PYOMO

from pyomo.environ import *
from pyomo.opt import SolverFactory

print(datetime.datetime.now())      # Time stamp start of optimisation

########################################################################################################################
# PYOMO model starts here.
########################################################################################################################

model = ConcreteModel()

########################################################################################################################
# Scalar parameters (parameters used directly in the optimisation problem formulation are given the PYOMO Param class.)
########################################################################################################################
# Intrinsic parameters (taken from Reed et al. 2016, except LossBOP. taken from Weber 2013.
model.VOCV50SOC = Param(initialize=1.4)                # OCV at 50% SOC (PNNL)
model.LossBOP = Param(initialize=0.02)                 # One way fractional power losses due to BOP, Weber et al. 2013
model.EffCoul = Param(initialize=0.964)                # Assumed constant with I
model.ASR = Param(initialize=0.0000894)                # Ohm.m2 from linear fit on over-potential v current density
model.Vfaradaic = Param(initialize=0.035)              # Faradaic over-potential, y axis intercept of linear fit.
BESSMaxCurrentDensity = 3200                           # Maximum at which polarization curve is still linear

# RFB system sizing for absolute output of scheduling problem is done as follows:

# 1: Define power rating (kW)
BESSRatedPower = 1

# 2: From experimental data, find VE that gives EffDC ( = EffV*EffCoul*(1-LossBOP)^2 ) >= 0.75 (round-trip)
EffVRated = 0.810

# 3: Identify current density that corresponds to above EffV
BESSRatedCurrentDensity = 1133                         # A/m2, current reaching external stack terminals

# 4: Set stack area such that system can output 1kW DC despite voltaic and balance of plant losses.
model.StackArea = Param(initialize=1000 * BESSRatedPower / (BESSRatedCurrentDensity * model.VOCV50SOC *
                                                            sqrt(EffVRated) * (1 - model.LossBOP)))

# 5: Set energy to power ratio (a.k.a discharge time)
EtoP = 4

# 6: Calculate required coulombic capacity, considering coulombic losses during discharge.
model.BESSCapacity = BESSRatedCurrentDensity * EtoP / sqrt(model.EffCoul)

# For the LP formulation it is necessary to choose a representative (average) VE that will apply during scheduling.
# Here, the VE corresponding to a midpoint current-density of 1600 A/m2 was used.
model.EffV = Param(initialize=0.763)

# Data-set
model.TimeStep = Param(initialize=1)  # Hours

########################################################################################################################
# indexed parameters
########################################################################################################################

# Index
model.I = range(len(price))

# Input price data


def init_price(model, t):
    return price[t]


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


########################################################################################################################
# variables
########################################################################################################################

def current_density_bess_charge_init(model, i):
    return 800  # In A/m2


model.CurrentDensityCharge = Var(model.I, within=NonNegativeReals, initialize=current_density_bess_charge_init)


def current_density_bess_discharge_init(model, i):
    return 800  # In A/m2


model.CurrentDensityDischarge = Var(model.I, within=NonNegativeReals, initialize=current_density_bess_charge_init)


########################################################################################################################
# constraints
########################################################################################################################

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


########################################################################################################################
# objective
########################################################################################################################
def objective_expression(model):
    return model.StackArea * model.TimeStep * (1 / 1000000) \
           * sum(model.Price[t] *
                 ((model.VOCV50SOC * model.CurrentDensityDischarge[t] * sqrt(model.EffV) * (1 - model.LossBOP)
                   - (model.VOCV50SOC * model.CurrentDensityCharge[t] / (sqrt(model.EffV) * (1 - model.LossBOP))))
                  ) for t in model.I)


model.Objective = Objective(rule=objective_expression, sense=maximize)

opt = SolverFactory('gurobi')

results = opt.solve(model)

print(datetime.datetime.now())

########################################################################################################################
# Terminal window output of optimization results
########################################################################################################################
# Predicted revenue, i.e. based on the assumed constant voltaic efficiency
for i in model.I:
    print(i, "%.2f" % model.CurrentDensityCharge[i].value, "%.2f" % model.CurrentDensityDischarge[i].value,
          "%.2f" % value(SOCtracker[i]))
print("Predicted revenue from", BESSRatedPower, "kW, ", str(BESSRatedPower*EtoP), "kWh system: £",
      "%.3f" % value(model.Objective), "under fixed efficiency assumption.")

# Actual revenue considering the I2R losses, as in the NLP formulation
actual_hourly_revenue = []
for i in model.I:
    actual_rev_hour_t = model.StackArea * model.TimeStep * (1 / 1000000) \
                            * (model.Price[i] *
                                ((model.VOCV50SOC - model.Vfaradaic) * model.CurrentDensityDischarge[i].value *
                                    (1 - model.LossBOP) -
                                    (model.VOCV50SOC + model.Vfaradaic) * model.CurrentDensityCharge[i].value *
                                    (1 / (1 - model.LossBOP))
                                    - model.ASR * (model.CurrentDensityDischarge[i].value ** 2 +
                                                   model.CurrentDensityCharge[i].value ** 2)))
    actual_hourly_revenue = actual_hourly_revenue + [actual_rev_hour_t]
actual_revenue = sum(actual_hourly_revenue)
print("Actual revenue from", BESSRatedPower, "kW, ", str(BESSRatedPower*EtoP), "kWh system: £",
          "%.3f" % actual_revenue, "under real efficiency.")

########################################################################################################################
# CSV output of optimization results
########################################################################################################################
# First convert indexed parameters and variables to lists:
price_data = []
dis_current_density_data = []
chg_current_density_data = []
current_density_data = []
SOC_data = []
predicted_hourly_revenue = []
for i in model.I:
    price_data = price_data + [model.Price[i]]
    dis_current_density_data = dis_current_density_data + [model.CurrentDensityDischarge[i].value]
    chg_current_density_data = chg_current_density_data + [model.CurrentDensityCharge[i].value]
    current_density_data = current_density_data + \
        [(model.CurrentDensityCharge[i].value - model.CurrentDensityDischarge[i].value)]
    SOC_data = SOC_data + [value(SOCtracker[i])]
    predicted_hourly_revenue = predicted_hourly_revenue + [value(model.StackArea * model.TimeStep * model.Price[i] *
                    ((model.VOCV50SOC * sqrt(model.EffV) * (1-model.LossBOP) * model.CurrentDensityDischarge[i] -
                      (model.VOCV50SOC / (sqrt(model.EffV)*(1-model.LossBOP))) * model.CurrentDensityCharge[i]))
                                                                 / 1000000)]

# Then arrange lists in pandas dataframe format and export to CSV
results_table = pd.DataFrame({"Price": price_data,
                              "Discharge Current Density": dis_current_density_data,
                              "Charge Current Density": chg_current_density_data,
                              "Current Density": current_density_data,
                              "SOC": SOC_data,
                              "Predicted Hourly Revenue, £": predicted_hourly_revenue,
                              "Actual Hourly Revenue": actual_hourly_revenue})

results_table.to_csv('LP_IDD1_hourly_results.csv', sep=',')
