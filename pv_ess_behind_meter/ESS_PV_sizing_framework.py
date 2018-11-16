# coding: utf-8
"""This script imports price, load and solar data for an electrical retail customer, then uses LP to minimise costs by
optimal ESS scheduling. PV power goes by default to suppply load, with any excess being split between export and ESS
charging, with a variable, r, denoting the fraction going to the ESS.

The optimisation is performed within a triple loop, to probe:
* PV installed power
* ESS installed power
* ESS installed energy
Author: Diarmid Roberts. Date: 4th October 2018"""
########################################################################################################################
# Import required tools
########################################################################################################################

# import off-the-shelf scripting modules
from __future__ import division     # Without this, rounding errors occur in python 2.7, but apparently not in 3.4
import pandas as pd                 # CSV handling
import datetime                     # For time stamping optimisation process

#Import price and load data
price_data = pd.read_csv('price.csv')
load_data = pd.read_csv('load.csv')
solar_data = pd.read_csv('solar.csv')
# import PYOMO
from pyomo.environ import *
from pyomo.opt import SolverFactory

########################################################################################################################
# User entered parameters
########################################################################################################################
# Constraints on installed ratings of ESS and PV
AvailableArea = 1800                # m2 available for PV installation
#MaxPowerPV = AvailableArea * 0.150  # Convert area to peak power assuming a certain kWp/m2
MaxPowerPV = 150                    # Set to be double max load, so export not greater than current peak load
MaxPowerESS = 30                    # Set a maximum for ESS power rating (kW)
MaxDurationESS = 4                  # Maximum energy to power ratio of ESS (h)
EffESS = 0.9                        # Round trip energy efficiency of ESS
FiT = 0.042                         # FiT for PV (£/kWh)
ExportTariff = 0.052                # FiT export Tariff (£/kWh)

# Define granularity of sizing matrix (fraction of max install)
StepESSPower = 0.5
StepESSDuration = 0.2
StepSolar = 0.5

########################################################################################################################
# Optimisation Framework
########################################################################################################################
# This loop runs through different PV installation sizes
df_by_PV_install = pd.DataFrame()
for s in range(0,int(1+1/StepSolar)):
    PVRating = s * StepSolar * MaxPowerPV
    print('PV', PVRating, 'kW')
    # this loop goes through different ESS power ratings
    df_by_ESS_power = pd.DataFrame()
    for p in range(0,int(1+1/StepESSPower)):
        print('ESS', p*StepESSPower*MaxPowerESS, "kW")
        # This loop goes through the ESS duration options
        annual_avoided_cost_by_ESS_duration = []  # Receptacle for data by duration
        for e in range(0,int(1+1/StepESSDuration)):
            print(e*StepESSDuration*MaxDurationESS, "hour")
            annual_avoided_cost = 0  # Start count of daily avoided cost
            # This loop goes through the CSV file and returns a day of data against which ESS schedule is optimised.
            for d in range(1,3):
                pv_profile = [i*PVRating for i in solar_data.ix[:, d].tolist()]     # Convert fractional value to kW
                price_profile = [i/100 for i in price_data.ix[:, d].tolist()]       # Convert p/kWh to £/kWh
                load_profile = [2*i for i in load_data.ix[:,d].tolist()]            # Convert half hourly consumption to power

                ########################################################################################################
                #PYOMO model starts here
                ########################################################################################################

                model = ConcreteModel()

                ########################################################################################################
                # Model Index and timestep
                ########################################################################################################

                model.I = range(len(price_profile))
                model.Tau = Param(initialize=0.5)  # Timestep in hours

                ########################################################################################################
                #Scalar Parameters
                ########################################################################################################
                model.ESS_P = Param(initialize=MaxPowerESS*(p+0.0001)*StepESSPower)          # kW rating of ESS, p can't be 0!
                model.ESS_E = Param(initialize=model.ESS_P*(e+0.0001)*StepESSDuration*MaxDurationESS)  # kWh rating of ESS, e can't be 0!
                model.ESS_Eff = Param(initialize=0.9)                                   # Round trip AC-AC efficiency of ESS
                model.ExportTariff = Param(initialize=ExportTariff)                     # Tariff for export of PV power
                model.FiT = Param(initialize=FiT)                                       # Feed in Tariff

                ########################################################################################################
                # Indexed Parameters
                ########################################################################################################

                def init_price(model, i):  # This initiates an indexed price param list
                    return price_profile[i]

                model.Price = Param(model.I, initialize=init_price)

                def init_power_load(model, i):  # This initiates an indexed load power param list
                    return load_profile[i]

                model.PowerLoad = Param(model.I, initialize=init_power_load)

                def init_available_pv_output_rule(model, i):
                    return pv_profile[i]

                model.PVavailable = Param(model.I, initialize=init_available_pv_output_rule)

                # Conditional parameters generated from raw data

                def init_pv_surplus(model, i):
                    if load_profile[i] < pv_profile[i]:
                        return pv_profile[i] - load_profile[i]
                    else:
                        return 0
                model.PVsurplus = Param(model.I, initialize=init_pv_surplus)

                def init_pv_self(model, i):
                    if load_profile[i] < pv_profile[i]:
                        return load_profile[i]
                    else:
                        return pv_profile[i]

                model.PVself = Param(model.I, initialize=init_pv_self)

                # Indexed parameters required for constraints

                def init_max_power(model, i):
                    return model.ESS_P

                model.MaxPowerESS = Param(model.I, initialize=init_max_power)

                def init_min_soc_ess(model, i):
                    return 0.15

                model.MinSOCESS = Param(model.I, initialize=init_min_soc_ess)

                def init_max_soc_ess(model, i):
                    return 1.00

                model.MaxSOCESS = Param(model.I, initialize=init_max_soc_ess)

                def init_efficiency(model, i):
                    return model.ESS_Eff

                model.Efficiency = Param(model.I, initialize=init_efficiency)


                ########################################################################################################
                # Variables
                ########################################################################################################

                def init_power_ess_discharge(model, i):
                    return model.ESS_P / 2 # Initialize discharge power variable at mid point.

                model.PowerESSDischarge = Var(model.I, within=NonNegativeReals, initialize=init_power_ess_discharge)

                def init_power_ess_charge_from_grid(model, i):
                    return model.ESS_P / 2  # Initialize ESS charge variable at mid point.

                model.PowerESSChargeFromGrid = Var(model.I, within=NonNegativeReals, initialize=init_power_ess_charge_from_grid)

                def init_fraction_pv_surplus_stored(model, i):
                    return 0
                model.r = Var(model.I, within=NonNegativeReals, bounds=(0, 1), initialize=init_fraction_pv_surplus_stored)


                ########################################################################################################
                # Constraints
                ########################################################################################################

                # ESS
                # Max ESS output
                def max_discharge_power_rule(model, i):
                    return model.PowerESSDischarge[i] - model.MaxPowerESS[i] <= 0

                model.MaxDischargePowerESSConstraint = Constraint(model.I, rule=max_discharge_power_rule)

                # Max ESS input
                def max_charge_power_rule(model, i):
                    return model.PowerESSChargeFromGrid[i] + model.r[i] * model.PVsurplus[i] - model.MaxPowerESS[i] <= 0

                model.MaxChargePowerESSConstraint = Constraint(model.I, rule=max_charge_power_rule)

                # No export rule

                def no_export_rule(model, i):
                    return model.PowerESSDischarge[i] + model.PVself[i] - model.PowerLoad[i] <= 0

                model.NoExportFromESSConstraint = Constraint(model.I, rule=no_export_rule)

                # ESS SOC constraints

                model.SOCAtStart = Param(initialize=0.5)
                SOCTracker = []
                SOC = model.SOCAtStart
                for i in model.I:
                    SOC_increment = (model.Tau / model.ESS_E) *\
                                        ((model.r[i] * model.PVsurplus[i] + model.PowerESSChargeFromGrid[i])
                                         * sqrt(model.Efficiency[i]) -
                                         model.PowerESSDischarge[i] / sqrt(model.Efficiency[i]))
                    SOC = SOC + SOC_increment
                    SOCTracker = SOCTracker + [SOC]

                def min_soc_rule(model, i):
                    return SOCTracker[i] - model.MinSOCESS[i] >= 0

                model.MinSOCESSConstraint = Constraint(model.I, rule=min_soc_rule)

                def max_soc_rule(model, i):
                    return SOCTracker[i] - model.MaxSOCESS[i] <= 0

                model.MaxSOCESSConstraint = Constraint(model.I, rule=max_soc_rule)

                # Return to initial SOC each day

                model.SOCAtEnd = SOCTracker[len(model.I) - 1]  # Grab SOC value in last period

                model.ESSEnergyConservation = Constraint(expr=model.SOCAtStart - model.SOCAtEnd == 0)

                #########################################################################################
                # Objective
                #########################################################################################

                def objective_expression(model):
                    # Avoided cost
                    return model.Tau * sum(model.Price[i] * (model.PowerESSDischarge[i] - model.PowerESSChargeFromGrid[i]
                                                             + model.PVself[i])+
                                           model.FiT * model.PVavailable[i] +
                                           model.ExportTariff * (1-model.r[i]) * model.r[i] * model.PVsurplus[i]
                                           for i in model.I)

                model.Objective = Objective(rule=objective_expression, sense=maximize)

                opt = SolverFactory('gurobi')

                results = opt.solve(model)

                ########################################################################################################
                # Terminal output of optimisation results
                ########################################################################################################

                print(value(model.Objective))
                annual_avoided_cost =annual_avoided_cost + int(value(model.Objective))


            annual_avoided_cost_by_ESS_duration = annual_avoided_cost_by_ESS_duration + [annual_avoided_cost]
            df_by_ESS_duration = pd.DataFrame([annual_avoided_cost_by_ESS_duration])
        df_by_ESS_power = pd.concat([df_by_ESS_power,df_by_ESS_duration])
    df_by_PV_install = pd.concat([df_by_PV_install,df_by_ESS_power])

print(df_by_PV_install)
df_by_PV_install.to_csv('sizing_matrix.csv', sep=',', encoding='utf-8')
        ################################################################################################################
        # CSV output of results
        ################################################################################################################
