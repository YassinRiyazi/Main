""" 
    Author: Chris Westbrook 
    
    http://www.met.reading.ac.uk/~sws04cdw/viscosity_calc.html

    parameterization in Cheng (2008) Ind. Eng. Chem. Res. 47 3285-3288, with a number of adjustments (see below), 
    which are described in Volk and Kähler (2018) Experiments in Fluids 59 75.

"""

import numpy
import math

def Viscosity_calc(glycerolMass: float,
                   waterMass: float,
                   temperature: float = 27, 
                   verbose: bool = True) -> float:

    """
    Calculate the viscosity of a glycerol-water mixture based on mass fractions and temperature.
    
    Args:
        glycerolMass (float): Mass of glycerol in grams.
        waterMass (float): Mass of water in grams.
        temperature (float): Temperature in degrees Celsius. Default is 27.
        verbose (bool): If True, prints intermediate results. Default is True.  

    Returns:
        float: Viscosity of the glycerol-water mixture in Ns/m².
    """
    #Densities ----------------
    glycerolDen = (1273.3-0.6121*temperature)/1000 			    #Density of Glycerol (g/cm3)
    waterDen    = (1-math.pow(((abs(temperature-4))/622),1.7)) 	#Density of water (g/cm3)
    glycerolVol = glycerolMass/glycerolDen
    waterVol    = waterMass/waterDen

    # #Fraction calculator ---------------
    totalMass       = glycerolMass + waterMass
    mass_fraction   = glycerolMass / totalMass
    vol_fraction    = glycerolVol / (glycerolVol + waterVol)

    if verbose:
        print ("Mass fraction of mixture =", round(mass_fraction,5))
        print ("Volume fraction of mixture =", round(vol_fraction,5))

    #Viscosity calculator ----------------
    glycerolVisc= 0.001*12100*numpy.exp((-1233+temperature)*temperature/(9900+70*temperature))
    waterVisc   = 0.001*1.790*numpy.exp((-1230-temperature)*temperature/(36100+360*temperature))
    a=0.705-0.0017*temperature
    b=(4.9+0.036*temperature)*numpy.power(a,2.5)
    alpha=1-mass_fraction+(a*b*mass_fraction*(1-mass_fraction))/(a*mass_fraction+b*(1-mass_fraction))
    A=numpy.log(waterVisc/glycerolVisc)
    viscosity_mix=glycerolVisc*numpy.exp(A*alpha)

    if verbose:
        print ("Viscosity of mixture =",round(viscosity_mix,5), "Ns/m2")

    return viscosity_mix

