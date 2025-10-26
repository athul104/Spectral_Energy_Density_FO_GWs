#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#					                     	     Spectral Energy Density of First-order Inflationary Gravitational Waves
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Owner        : Athul K. Soman 
# Collaborators: Swagat S. Mishra, Mohammed Shafi, Soumen Basak
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

# This python script is to compute the spectral energy density of first-order inflationary gravitational waves (GWs) for a multi-epoch reheating scenario. 
# The transition from one epoch to the next epoch is assumed to be instantaneous. 
# This code is associated with the paper "Inflationary Gravitational Waves as a probe of the unknown post-inflationary primordial Universe" 
# (arXiv.2407.07956 [https://arxiv.org/abs/2407.07956]).

# This code will generate the plot of the spectral energy density of first-order inflationary GWs as a function of present-day frequency of GWs 
# for a multi-epoch reheating scenario as specified by the user. The png image will be saved in the same folder as the code. 
# The instructions to provide the inputs are given in the comments. 
# Your inputs are only needed in the section "YOU HAVE TO PROVIDE INPUT ONLY HERE" (line 68-111). 



#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Importing the required packages
# The required packages are numpy, matplotlib, scipy, mpmath, shapely, and fractions.
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.special import gamma
from matplotlib import rcParams
from matplotlib import rc
import matplotlib as mpl
import mpmath
from fractions import Fraction
from shapely.geometry import LineString
from shapely import intersects
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

# #font
mpl.rcParams['font.family'] = 'Times New Roman'

# activate latex text rendering
plt.rc('text', usetex=True)

#LaTex setting
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams['text.latex.preamble'] = r'\boldmath'

#Plot setting:

plt.rcParams['figure.figsize'] = (9, 6)
plt.rcParams['font.size'] = 11

plt.rcParams['xtick.major.size'] = 3
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = 1

plt.rcParams['axes.linewidth'] = 1

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# To set the directory to the folder where the code is saved
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# YOU HAVE TO PROVIDE INPUT ONLY HERE
#################################################################################################################################################################


# 1) EQUATION OF STATE (EoS) DURING REHEATING
# -----------------------------------------------------------------------------------
#   +) Provide the equation of state of each epochs during reheating in the following list in order from the EARLIEST epoch to the LATEST epoch.
#   +) The values must be between -0.28 <= w < 1.

EoS_list = [-0.2, -0.1, 0.0, 0.4, 0.6, 0.8, 0.9, 0.99] # [w_1, w_2, w_3, ..., w_n]  


# 2) TEMPERATURE OR ENERGY SCALE AT THE END OF REHEATING
# -----------------------------------------------------------------------------------
#   +) Provide the temperature achieved at the end of reheating (T_Rh) in GeV or the energy scale of the universe at the end of reheating (E_Rh) in GeV. 
#      Comment the other one.
#   +) In accordance to the BBN constarint, the temperature at the end of reheating must be greater than 10^(-3) GeV, i.e., T_Rh > T_BBN = 10^(-3) GeV

# T_Rh = 1e-3 #GeV
E_Rh = 0.3 #GeV


# 3) ENERGY SCALES DURING REHEATING
# -----------------------------------------------------------------------------------
#   +) Provide the energy scales marking the end of each epoch during reheating in GeV in the following list.
#   +) The energy scale at the end of reheating (E_Rh) should NOT be included in this list as it is already provided above. Even if you provide the 
#      temperature at the end of reheating (T_Rh), you should NOT include the corresponding energy scale (E_Rh) in this list.
#   +) The energy scales must be in the order from the LATEST epoch to the EARLIEST epoch during reheating, i.e., [E_{n-1}, E_{n-2}, ..., E_1],
#      where n is the number of epochs during reheating.
#   +) If there is only one epoch of reheating, provide an empty list.
#   +) For example, if you have 3 epochs during reheating, you have to provide 2 values in the list, corresponding to the end of the second and
#      first epochs in that order.
#   +) The values must be between 10^(-3) GeV and 10^(16) GeV.

Energy_list = [1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14] #GeV [E_{n-1}, E_{n-2}, ..., E_1]



# 4) TENSOR-TO-SCALAR RATIO OR ENERGY SCALE DURING INFLATION
# -----------------------------------------------------------------------------------
#   +) Provide either the value of tensor-to-scalar ratio 'r', or the value of energy scale during inflation 'E_inf' in GeV. Comment the other one.
#   +) The value of tensor-to-scalar ratio must be less than the current upper bound r < 0.036 [https://arxiv.org/abs/2110.00483]

r = 1e-3 
# E_inf = 5.76*10**15 #GeV


# 5) BBN CONSTRAINT CHECKING METHOD
# -----------------------------------------------------------------------------------
#   +) Choose the method you want to use for checking the BBN constraint from the following options:

#       1. piecewise    : (default) the method to check the BBN constraint is the equation which approximates the
#                         BBN constraint integral as piecewise integrals, [see Eq. (2.64)].
#       2. intersection : Check whether the curve of spectral energy density intersects the horizontal BBN constraint line at 1.13 * 10^{-6}.
#       3. weaker       : To cross-check with the results in Fig. 5, 6 and 12 of our paper [https://arxiv.org/abs/2407.07956], [see Eq. (3.1)].

#   +) Note that it is recommended to use the 'piecewise' method for checking the BBN constraint as it is more accurate for the general case.
#   +) The other two methods are advised to be used only for cross-checking the results in the paper or to get a rough estimate of the BBN constraint
#      for specific cases as mentioned in the paper.

BBN_method = 'piecewise'
# BBN_method = 'intersection'
# BBN_method = 'weaker'


# 6) NUMBER OF DATA POINTS
# -----------------------------------------------------------------------------------
#   +) Provide the number of data points (frequency) you want in the plot of GW spectral energy density in x-axis.

num_of_points = 1000

#################################################################################################################################################################


# defining the values of the constants
Omega_rad_0 = 4.16*10**(-5)                         # Present radiation density parameter (This is actually Omega_{rad, 0}*h^2) [see Eq. (E.7)]
T_0 = 2.35*10**(-13)     #GeV                       # Present temperature [https://arxiv.org/abs/0911.1955]
m_P = 2.44*10**18        #GeV                       # Reduced Planck mass
BBN_constraint = 1.13*10**(-6)                      # Upper bound on Omega_GW from BBN observations [see Eq. (2.59)]
A_S = 2.1*10**(-9)                                  # Amplitude of scalar perturbations on CMB scales [Planck 2018: https://arxiv.org/abs/1807.06209]

#................................................................................................................................................................

try:
    r
    # If r is defined, execute this block

    if r >= 0.036: # The current upper bound on the tensor-to-scalar ratio r [https://arxiv.org/abs/2110.00483]
        print('The value of tensor-to-scalar ratio r is greater than the current upper bound. Please provide a lower value')
        exit()
    
    H_inf = m_P * np.pi * np.sqrt(A_S * r/2)    #GeV    # Hubble parameter during inflation (in GeV)
    E_inf = 3**(1/4) * np.sqrt(m_P * H_inf)     #GeV    # Energy scale during inflation

except NameError:
    # If r is not defined, execute this block
    try:
        E_inf
        # If E_inf is defined, execute this block

        if E_inf >= 1.39 * 10**16: #GeV # The current upper bound on the energy scale during inflation [see Eq. (A.31)]
            print('The value of energy scale during inflation E_inf is greater than the current upper bound. Please provide a lower value')
            exit()

        H_inf = E_inf**2 / (np.sqrt(3) * m_P)       #GeV    # Hubble parameter during inflation (in GeV)
    
    except NameError:
        print('Please provide either the value of tensor-to-scalar ratio r, or the value of energy scale during inflation E_inf in GeV')
        exit()
   
#.................................................................................................................................................................

#Checking the validity of the equations of state provided
for i in range(len(EoS_list)):
    if i == 0:
        if EoS_list[i] < -0.28 or EoS_list[i] > 1:
            print(f'The equation of state given in position {i+1} of EoS_list is invalid. EoS must be between -0.28 and 1')
            exit()
    else:
        if EoS_list[i] < -0.28 or EoS_list[i] >= 1:
            print(f'The equation of state given in position {i+1} of EoS_list is invalid. EoS must be between -0.28 and 1')
            exit()

#Checking the validity of the energy scales provided

Energy_list.sort(reverse=True) # Sorting the energy scales in descending order

if Energy_list == []: # If reheating has only single epoch
    pass

else:
    for i in range(len(Energy_list)):
        if Energy_list[i] < 10**(-3) or Energy_list[i] > 10**16:
            print(f'The energy scale given in position {i+1} of Energy_list is invalid. Energy scale must be between 10^(-3) GeV and 10^16 GeV')
            exit()


    #checking the validity of the energy scale during inflation
    if E_inf < Energy_list[0]:
        print('The energy scale during inflation is less than the energy scale at the end of the first epoch of reheating')
        print('Please provide a higher value for tensor-to-scalar ratio r')
        exit()

#.................................................................................................................................................................

# Loading the data for the effective relativistic degrees of freedom in energy and entropy. The data is in the form of [Temperature in GeV, g_star, g_s]. 
Eff_rel_dof_data = np.loadtxt('eff_rel_dof.txt')

# Converting the data to numpy arrays
Temp_in_GeV = np.array(Eff_rel_dof_data[:,0])
Eff_rel_dof_in_energy = np.array(Eff_rel_dof_data[:,1])
Eff_rel_dof_in_entropy = np.array(Eff_rel_dof_data[:,2])


#.................................................................................................................................................................

def g_star_k(T):
    '''This function returns the effective relativistic degrees of freedom in energy at a given temperature in GeV. 
    The data is taken from the external file eff_rel_dof.txt'''
    argument = np.where(Temp_in_GeV - T >= 0, Temp_in_GeV - T, np.inf).argmin()
    return Eff_rel_dof_in_energy[argument]

def g_s_k(T):
    '''This function returns the effective relativistic degrees of freedom in entropy at a given temperature in GeV.
    The data is taken from the external file eff_rel_dof.txt'''
    argument = np.where(Temp_in_GeV - T >= 0, Temp_in_GeV - T, np.inf).argmin()
    return Eff_rel_dof_in_entropy[argument]

#.................................................................................................................................................................

def temp_to_energy(T):
    '''This function takes temperature in GeV as input and returns energy in GeV, temperature in GeV and g_star_k(T) as output'''
    return (np.pi**2 / 30 * g_star_k(T))**(1/4) * T, T, g_star_k(T)

vec_temp_to_energy = np.vectorize(temp_to_energy)

# Creating a table for energy in GeV, temperature in GeV and g_star_k(T) for temperatures between 0.1 GeV and 10^16 GeV
temp_arr = np.logspace(np.log10(10**(-3)), 16, 100000)
temperature_table = np.column_stack((vec_temp_to_energy(temp_arr)[0], vec_temp_to_energy(temp_arr)[1], vec_temp_to_energy(temp_arr)[2]))


#  Function for converting energy in GeV to temperature in GeV
def Temp(E):
    '''This function takes energy in GeV as input and returns temperature in GeV as output. It uses the temperature_table created above. This can
    only be used for temperatures in between 10^(-3) GeV and 10^16 GeV. For temperatures outside this range, the function will return an error. This function 
    is mainly built for effective temperatures during reheating. If one need to include temperatures outside this range, the temperature_arr should be
    updated accordingly.'''

    argument = np.where(temperature_table[:,0] - E >= 0, temperature_table[:,0] - E, np.inf).argmin()
    g_star = temperature_table[argument,2]
    return E / (np.pi**2 / 30 * g_star)**(1/4)

vec_Temp = np.vectorize(Temp)   # Vectorizing the function Temp

try:
    T_Rh
    # If T_Rh is defined, execute this block
    pass

except NameError:
    # If T_Rh is not defined, execute this block
    try:
        E_Rh
        # If E_Rh is defined, execute this block
        T_Rh = Temp(E_Rh) #GeV
    except NameError:
        print('Please provide either the value of temperature at the end of reheating T_Rh in GeV, or the value of energy scale at the end of reheating E_Rh in GeV')
        exit()

#checking the validity of the temperature at the end of reheating
if T_Rh < 10**(-3):
    print('The temperature at the end of reheating must be greater than 10^(-3) GeV')
    exit()

if len(Energy_list) != 0:
    if T_Rh >= Temp(Energy_list[-1]):
        print('The temperature at the end of reheating is greater than the effective temperature at the end of the second-last epoch of reheating')
        exit()

#.................................................................................................................................................................

def freq(T):
    """Function for converting temperature in GeV to present-frequency of GWs in Hz.
    [See Eq. (2.50)]"""
    return 7.43 * 10**(-8) * (g_s_k(T_0)/g_s_k(T))**(1/3) * (g_star_k(T)/90)**(1/2) * T


def E(T):
    """Function to convert temperature in GeV to energy scale of the universe in GeV.
    [See Eq. (2.51)]"""
    return T*(np.pi**2 * g_star_k(T)/ 30)**(1/4)


#.................................................................................................................................................................

# Converting the energy scales provided into effective temperatures in GeV

if len(Energy_list) == 0: # If reheating has only single epoch 
    Temperature_list = [] #GeV

else:
    Temperature_list = vec_Temp(Energy_list) #GeV
    Temperature_list = Temperature_list.tolist()

Temperature_list.append(T_Rh) #GeV # adding the end of reheating temperature



if len(Temperature_list) != len(EoS_list):
    print('There is a discrepancy in the number of epochs and the number of energy scales provided. Please check the inputs.')
    exit()

#.................................................................................................................................................................

EoS_list.extend([Fraction(1, 3), 0])                # Adding the EoS of radiation and matter domination epochs after reheating
Temperature_list.append(10**(-9)) #GeV              # Adding the temperature at the matter-radiation equality

freq_list = [freq(T) for T in Temperature_list]     #Hz #creating a frequency list corresponding to the temperature list

# Converting EoS list to alpha list                 #alpha = 2/(1+3*w)
alpha_arr = 2/(1+3*np.array(EoS_list))

#.................................................................................................................................................................

# return the coefficients A_k_n and B_k_n for the final epoch
def coeff(f):
    """Function to calculate the coefficients A_k_n and B_k_n for the MD epoch after reheating
    corresponding to a particular frequency f. This function returns a tuple (A_k_n, B_k_n)"""
    A_k_arr = np.zeros(len(alpha_arr))
    B_k_arr = np.zeros(len(alpha_arr))

    y_arr = np.zeros(len(alpha_arr)-1)

    for i in range(len(y_arr)):
        y_arr[i] = f/freq_list[i]

    A_k_arr[0] = 2**(alpha_arr[0]- 1/2 ) * gamma(alpha_arr[0] + 1/2)    #A_{k, 1} [see Eq. (2.26)]
    B_k_arr[0] = 0                                                      #B_{k, 1} [see Eq. (2.26)]

    for i in range(1, len(alpha_arr)):
        an_ym = alpha_arr[i] * y_arr[i-1]       # \alpha_n * y_m, where m = n-1
        am_ym = alpha_arr[i-1] * y_arr[i-1]     # \alpha_m * y_m, where m = n-1 

        an_m_half = alpha_arr[i] - 1/2          # \alpha_n - 1/2
        am_m_half = alpha_arr[i-1] - 1/2        # \alpha_m - 1/2

        an_p_half = alpha_arr[i] + 1/2          # \alpha_n + 1/2
        am_p_half = alpha_arr[i-1] + 1/2        # \alpha_m + 1/2

        C = ((an_ym)**(an_m_half))/((am_ym)**(am_m_half)) # The coefficient in Eq. (2.21) and (2.22)

        f_1 = mpmath.besselj(-(an_m_half), an_ym) #[see Eq. (2.17)]
        f_2 = mpmath.besselj(-(am_m_half), am_ym) #[see Eq. (2.18)]
        f_3 = mpmath.besselj(-(an_p_half), an_ym) #[see Eq. (2.19)]
        f_4 = mpmath.besselj(-(am_p_half), am_ym) #[see Eq. (2.20)

        g_1 = mpmath.besselj(an_m_half, an_ym) #[see Eq. (2.17)]
        g_2 = mpmath.besselj(am_m_half, am_ym) #[see Eq. (2.18)]
        g_3 = mpmath.besselj(an_p_half, an_ym) #[see Eq. (2.19)]
        g_4 = mpmath.besselj(am_p_half, am_ym) #[see Eq. (2.20)]
        
        
        Deno = f_1 * g_3 + g_1 * f_3 # Denominator in Eq. (2.21) and (2.22)

        K = C / Deno

        Num_A1 = g_2 * f_3 + g_4 * f_1
        Num_B1 = f_2 * f_3 - f_4 * f_1

        Num_A2 = g_2 * g_3 - g_4 * g_1
        Num_B2 = f_2 * g_3 + f_4 * g_1


        A_k_arr[i] = K * (A_k_arr[i-1] * Num_A1 + B_k_arr[i-1] * Num_B1) # [see Eq. (2.21)]
        B_k_arr[i] = K * (A_k_arr[i-1] * Num_A2 + B_k_arr[i-1] * Num_B2) # [see Eq. (2.22)]

    return A_k_arr[-1], B_k_arr[-1]

#.................................................................................................................................................................

# Relativistic correction factor at beginning of last RD epoch
G_R = (g_star_k(Temperature_list[-2])/ g_star_k(T_0)) * (g_s_k(T_0)/g_s_k(Temperature_list[-2]))**(4/3) # (g_{*, r*}/g_{*, 0}) * (g_{s, 0}/g_{s, r*})^{4/3} 

const_coeff = 1/(96*(np.pi)**3) * G_R * Omega_rad_0 * (H_inf/m_P)**2 # The coefficient in Eq. (2.53)

#Function for calculating the spectral energy density at present time
def Omega_GW_0(f):
    """Function to calculate the present spectral energy density of GWs for a given frequency f.
    [See Eq. (2.53)]"""

    y_eq = f/freq_list[-1] # y_eq = f/f_eq
    

    return const_coeff * y_eq**(-2) * (coeff(f)[0]**2 + coeff(f)[1]**2)

vec_Omega_GW_0 = np.vectorize(Omega_GW_0)   #vectorizing the function Omega_GW_0

#.................................................................................................................................................................

# Loading the data for the sensitive curves. These are power-law integrated sensitivity curves (PLIS) for various GW detectors (except Planck)
# obtained from https://zenodo.org/records/3689582 in relation to the paper https://arxiv.org/abs/2002.04615. 

# Column 1: Log10 of the gravitational-wave frequency f, redshifted to today, in units of Hz
# Column 2: Log10 of the gravitational-wave energy density power spectrum h^2\Omega_PLIS


DATA1 = np.loadtxt("PLIS/aLIGO.txt")
DATA2 = np.loadtxt("PLIS/DECIGO.txt")
DATA3 = np.loadtxt("PLIS/LISA.txt")
DATA4 = np.loadtxt("PLIS/IPTA.txt")
DATA5 = np.loadtxt("PLIS/BBO.txt")
DATA6 = np.loadtxt("PLIS/Planck.txt")
DATA7 = np.loadtxt("PLIS/LV.txt")
DATA8 = np.loadtxt("PLIS/SKA.txt")
DATA9 = np.loadtxt("PLIS/PPTA.txt")
DATA10 = np.loadtxt("PLIS/ET.txt")
DATA11 = np.loadtxt("PLIS/CE.txt")
DATA12 = np.loadtxt("PLIS/LVK.txt")
DATA13 = np.loadtxt("PLIS/LVO2.txt")
DATA14 = np.loadtxt("PLIS/EPTA.txt")
DATA15 = np.loadtxt("PLIS/NANOGrav.txt")

# Separating the data for x and y axes
X1 = np.array(DATA1[:,0])
Y1 = np.array(DATA1[:,1])

X2 = np.array(DATA2[:,0])
Y2 = np.array(DATA2[:,1])

X3 = np.array(DATA3[:,0])
Y3 = np.array(DATA3[:,1])

X4 = np.array(DATA4[:,0])
Y4 = np.array(DATA4[:,1])

X5 = np.array(DATA5[:,0])
Y5 = np.array(DATA5[:,1])

X6 = np.array(DATA6[:,0])
Y6 = np.array(DATA6[:,1])

X7 = np.array(DATA7[:,0])
Y7 = np.array(DATA7[:,1])

X8 = np.array(DATA8[:,0])
Y8 = np.array(DATA8[:,1])

X9 = np.array(DATA9[:,0])
Y9 = np.array(DATA9[:,1])

X10 = np.array(DATA10[:,0])
Y10 = np.array(DATA10[:,1])

X11 = np.array(DATA11[:,0])
Y11 = np.array(DATA11[:,1])

X12 = np.array(DATA12[:,0])
Y12 = np.array(DATA12[:,1])

X13 = np.array(DATA13[:,0])
Y13 = np.array(DATA13[:,1])

X14 = np.array(DATA14[:,0])
Y14 = np.array(DATA14[:,1])

X15 = np.array(DATA15[:,0])
Y15 = np.array(DATA15[:,1])


# As the data is in Log10 scale, we need to convert it to normal scale
X1_DATA = 10**X1
Y1_DATA = 10**Y1

X2_DATA = 10**X2
Y2_DATA = 10**Y2

X3_DATA = 10**X3
Y3_DATA = 10**Y3

X4_DATA = 10**X4
Y4_DATA = 10**Y4

X5_DATA = 10**X5
Y5_DATA = 10**Y5

X6_DATA = 10**X6
Y6_DATA = 10**Y6

X7_DATA = 10**X7
Y7_DATA = 10**Y7

X8_DATA = 10**X8
Y8_DATA = 10**Y8

X9_DATA = 10**X9
Y9_DATA = 10**Y9

X10_DATA = 10**X10
Y10_DATA = 10**Y10

X11_DATA = 10**X11
Y11_DATA = 10**Y11

X12_DATA = 10**X12
Y12_DATA = 10**Y12

X13_DATA = 10**X13
Y13_DATA = 10**Y13

X14_DATA = 10**X14
Y14_DATA = 10**Y14

X15_DATA = 10**X15
Y15_DATA = 10**Y15

#.................................................................................................................................................................
# Plotting the GW spectral energy density

f_inf = freq(Temp(E_inf)) #Hz # Present frequency corresponding to the energy scale during inflation

start_freq = np.log10(2*10**(-20))#Hz       #starting frequency for the plot in Log10 scale
end_freq = np.log10(f_inf)#Hz               #end frequency for the plot in Log10 scale
f = np.logspace(start_freq, end_freq, num_of_points, endpoint=True, base=10.0)#Hz #frequency range for the plot
plt.loglog(f, vec_Omega_GW_0(f), color = 'k',lw = 2, zorder = 10)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Uncomment from the following lines as desired to plot the sensitivity curves of various GW detectors

# Sensitivity curve for aLIGO
plot1 = plt.plot(X1_DATA,Y1_DATA, color = 'red', label = 'aLIGO', ls = '-')
plt.setp(plot1, color='red', linewidth=1, linestyle='-')
plt.fill_between(X1_DATA, Y1_DATA, 10**(-2), color='red', alpha=0.1)

plt.text(30,1.95e-7, r"\textbf{aLIGO}", style='normal', fontsize=10, color='red',
    verticalalignment='center', horizontalalignment='center', bbox={'facecolor': 'white', 'edgecolor': 'black', 'pad': 0.1, 'boxstyle': 'round',
                                                                     'linewidth': 0.5}, zorder = 20)

#--------------------------------------------------------------------------------

# Sensitivity curve for DECIGO
plot1 = plt.plot(X2_DATA,Y2_DATA, color = 'dodgerblue', label = 'DECIGO', ls = '-')
plt.setp(plot1, color='dodgerblue', linewidth=1, linestyle='-')
plt.fill_between(X2_DATA, Y2_DATA, 10**(-2), color='dodgerblue', alpha=0.1)

plt.text(0.12,1.95e-12, r"\textbf{DECIGO}", style='normal', fontsize=10, color='dodgerblue',
    verticalalignment='center', horizontalalignment='center', bbox={'facecolor': 'white', 'edgecolor': 'black', 'pad': 0.1, 'boxstyle': 'round', 
                                                                    'linewidth': 0.5}, zorder = 20)

#--------------------------------------------------------------------------------

# Sensitivity curve for LISA
plot1 = plt.plot(X3_DATA,Y3_DATA, color = 'green', label = 'LISA', ls = '-')
plt.setp(plot1, color='green', linewidth=1, linestyle='-')
plt.fill_between(X3_DATA, Y3_DATA, 10**(-2), color='green', alpha=0.1)

plt.text(1e-3,1.95e-10, r"\textbf{LISA}", style='normal', fontsize=10, color='green',
    verticalalignment='center', horizontalalignment='center', bbox={'facecolor': 'white', 'edgecolor': 'black', 'pad': 0.1, 'boxstyle': 'round',
                                                                     'linewidth': 0.5}, zorder = 20)

#--------------------------------------------------------------------------------

# Sensitivity curve for IPTA
plot1 = plt.plot(X4_DATA,Y4_DATA, color = 'indigo', label = 'IPTA', ls = '-')
plt.setp(plot1, color='indigo', linewidth=1, linestyle='-')
plt.fill_between(X4_DATA, Y4_DATA, 10**(-2), color='indigo', alpha=0.1)

plt.text(1e-9,1.95e-10, r"\textbf{IPTA}", style='normal', fontsize=10, color='indigo',
    verticalalignment='center', horizontalalignment='center', bbox={'facecolor': 'white', 'edgecolor': 'indigo', 'pad': 0.1, 'boxstyle': 'round',
                                                                     'linewidth': 0.5}, zorder = 20)

#--------------------------------------------------------------------------------

# Sensitivity curve for BBO
plot1 = plt.plot(X5_DATA,Y5_DATA, color = 'purple', label = 'BBO', ls = '-')
plt.setp(plot1, color='purple', linewidth=1, linestyle='-')
plt.fill_between(X5_DATA, Y5_DATA, 10**(-2), color='purple', alpha=0.1)

plt.text(2,4e-17, r"\textbf{BBO}", style='normal', fontsize=10, color='purple',
    verticalalignment='center', horizontalalignment='center', bbox={'facecolor': 'white', 'edgecolor': 'black', 'pad': 0.1, 'boxstyle': 'round',
                                                                     'linewidth': 0.5}, zorder = 20)

#--------------------------------------------------------------------------------

# Sensitivity curve for Planck
plot1 = plt.plot(X6_DATA,Y6_DATA, color = 'teal', label = 'Planck', ls = '-')
plt.setp(plot1, color='teal', linewidth=1, linestyle='-')
plt.fill_between(X6_DATA, Y6_DATA, 10**(-2), color='teal', alpha=0.1)

plt.text(6e-18,1.95e-9, r"\textbf{Planck}", style='normal', fontsize=10, color='teal',
    verticalalignment='center', horizontalalignment='center', bbox={'facecolor': 'white', 'edgecolor': 'black', 'pad': 0.1, 'boxstyle': 'round',
                                                                     'linewidth': 0.5}, zorder = 20)

#--------------------------------------------------------------------------------

# Sensitivity curve for ET
plot1 = plt.plot(X10_DATA,Y10_DATA, color = 'darkorange', label = 'ET', ls = '-')
plt.setp(plot1, color='darkorange', linewidth=1, linestyle='-')
plt.fill_between(X10_DATA, Y10_DATA, 10**(-2), color='darkorange', alpha=0.1)

plt.text(30,1.95e-11, r"\textbf{ET}", style='normal', fontsize=10, color='darkorange',
    verticalalignment='center', horizontalalignment='center', bbox={'facecolor': 'white', 'edgecolor': 'black', 'pad': 0.1, 'boxstyle': 'round',
                                                                     'linewidth': 0.5}, zorder = 20)

#--------------------------------------------------------------------------------

# Sensitivity curve for CE
plot1 = plt.plot(X11_DATA,Y11_DATA, color = 'maroon', label = 'CE', ls = '-')
plt.setp(plot1, color='maroon', linewidth=1, linestyle='-')
plt.fill_between(X11_DATA, Y11_DATA, 10**(-2), color='maroon', alpha=0.1)

plt.text(2e2,1e-12, r"\textbf{CE}", style='normal', fontsize=10, color='maroon',
    verticalalignment='center', horizontalalignment='center', bbox={'facecolor': 'white', 'edgecolor': 'black', 'pad': 0.1, 'boxstyle': 'round',
                                                                     'linewidth': 0.5}, zorder = 20)

#--------------------------------------------------------------------------------

## Sensitivity curve for aLIGO + aVirgo
# plot1 = plt.plot(X7_DATA,Y7_DATA, color = 'darkslategrey', label = 'LV', ls = '-')
# plt.setp(plot1, color='darkslategrey', linewidth=1, linestyle='-')
# plt.fill_between(X7_DATA, Y7_DATA, 10**(-2), color='darkslategrey', alpha=0.1)

# plt.text(30,1.95e-8, r"\textbf{LV}", style='normal', fontsize=10, color='darkslategrey',
#     verticalalignment='center', horizontalalignment='center')

#--------------------------------------------------------------------------------

## Sensitivity curve for SKA
# plot1 = plt.plot(X8_DATA,Y8_DATA, color = 'cyan', label = 'SKA', ls = '-')
# plt.setp(plot1, color='cyan', linewidth=1, linestyle='-')
# plt.fill_between(X8_DATA, Y8_DATA, 10**(-2), color='cyan', alpha=0.1)

# plt.text(1e-9,1.95e-13, r"\textbf{SKA}", style='normal', fontsize=10, color='cyan',
#     verticalalignment='center', horizontalalignment='center')

#--------------------------------------------------------------------------------

## Sensitivity curve for PPTA
# plot1 = plt.plot(X9_DATA,Y9_DATA, color = 'brown', label = 'PPTA', ls = '-')
# plt.setp(plot1, color='brown', linewidth=1, linestyle='-')
# plt.fill_between(X9_DATA, Y9_DATA, 10**(-2), color='brown', alpha=0.1)

# plt.text(1e-9,1.95e-8, r"\textbf{PPTA}", style='normal', fontsize=10, color='brown',
#     verticalalignment='center', horizontalalignment='center')

#--------------------------------------------------------------------------------

## Sensitivity curve for aLIGO + aVirgo + KAGRA
# plot1 = plt.plot(X12_DATA,Y12_DATA, color = 'magenta', label = 'LVK', ls = '-')
# plt.setp(plot1, color='magenta', linewidth=1, linestyle='-')
# plt.fill_between(X12_DATA, Y12_DATA, 10**(-2), color='magenta', alpha=0.1)

# plt.text(30,1e-7, r"\textbf{LVK}", style='normal', fontsize=10, color='magenta',
#     verticalalignment='center', horizontalalignment='center')

#--------------------------------------------------------------------------------

## Sensitivity curve for aLIGO + aVirgo (O2)
# plot1 = plt.plot(X13_DATA,Y13_DATA, color = 'grey', label = 'LVO', ls = '-')
# plt.setp(plot1, color='grey', linewidth=1, linestyle='-')
# plt.fill_between(X13_DATA, Y13_DATA, 10**(-2), color='grey', alpha=0.1)

# plt.text(30,2.95e-7, r"\textbf{LVO}", style='normal', fontsize=10, color='grey',
#     verticalalignment='center', horizontalalignment='center')

#--------------------------------------------------------------------------------

## Sensitivity curve for EPTA
# plot1 = plt.plot(X14_DATA,Y14_DATA, color = 'teal', label = 'EPTA', ls = '-')
# plt.setp(plot1, color='teal', linewidth=1, linestyle='-')
# plt.fill_between(X14_DATA, Y14_DATA, 10**(-2), color='teal', alpha=0.1)

# plt.text(1e-9,3e-7, r"\textbf{EPTA}", style='normal', fontsize=10, color='teal',
#     verticalalignment='center', horizontalalignment='center')

#--------------------------------------------------------------------------------

## Sensitivity curve for NANOGrav
# plot1 = plt.plot(X15_DATA,Y15_DATA, color = 'darkorange', label = 'NANOGrav', ls = '-')
# plt.setp(plot1, color='darkorange', linewidth=1, linestyle='-')
# plt.fill_between(X15_DATA, Y15_DATA, 10**(-2), color='darkorange', alpha=0.1)

# plt.text(1e-9,1.e-7, r"\textbf{NANOGrav}", style='normal', fontsize=10, color='darkorange',
#     verticalalignment='center', horizontalalignment='center')

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plotting the BBN constraint
plt.axhline(y=BBN_constraint, color='grey', linestyle=':', lw = 1, zorder = 50)
plt.fill_between(np.logspace(start_freq, end_freq), BBN_constraint, 10**(-4), color='grey', alpha=0.8, zorder = 30)
plt.text(10e-6,8e-6, r"\textbf{BBN Constraint}", style='normal', fontsize=10, color='black',
    verticalalignment='center', horizontalalignment='center', bbox={'facecolor': 'white', 'edgecolor': 'black', 'pad': 0.3, 'boxstyle': 'round', 'linewidth': 0.5}, 
    zorder = 40)

f_eq = freq_list[-1] #Hz # Present frequency corresponding to matter-radiation equality

# Texts for matter-radiation equality and end of reheating
plt.text(f_eq - 0.6*f_eq, 1e-17, r"\textbf{Matter-radiation}", style='normal', fontsize=9, color='brown', rotation = 90)
plt.text(f_eq + 0.4*f_eq, 1e-16, r"\textbf{equality}", style='normal', fontsize=9, color='brown', rotation = 90)

plt.text(0.35* freq_list[-2], 1e-17, r"\textbf{End of reheating}", style='normal', fontsize=9, color='brown', rotation = 90)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# For labelling the EoS  of each epochs

# Boundary_list is to contain the frequencies corresponding to the end and beginnings of different epochs
Boundary_list = [10**end_freq]
Boundary_list.extend(freq_list)
Boundary_list.append(2*10**(-20))

Center_arr = np.zeros(len(Boundary_list)-1)
for i in range(len(Center_arr)):
    Center_arr[i] = np.exp((np.log(Boundary_list[i]) +  np.log(Boundary_list[i+1]))/2)

for i in range(len(Center_arr)-2):
    if np.abs(np.log10(Boundary_list[i+1]) - np.log10(Boundary_list[i])) <= 3:
        rot_new = 90
    else:
        rot_new = 0
    plt.text(x = Center_arr[i], y = 5*10**(-20), s=rf"$w_{{{i+1}}}$ = $\mathbf{{{EoS_list[i]}}}$", rotation=rot_new, fontsize=15
             , color='navy', horizontalalignment='center', verticalalignment='bottom',
                 bbox={'facecolor':'white', 'edgecolor':'black', 'boxstyle':'round', 'pad':0.1, 'alpha':1, 'linewidth':0.1}, zorder=20)

epoch_list = [ r'\textbf{RD}', r'\textbf{MD}']
for i in range(len(epoch_list)):
    plt.text(x=Center_arr[i+len(Energy_list)+1], y=5*10**(-20), s=epoch_list[i], rotation=0, fontsize=12, color='navy'
             , horizontalalignment='center', verticalalignment='bottom', bbox={'facecolor':'white', 'edgecolor':'black'
                                                                               , 'boxstyle':'round', 'pad':0.1, 'alpha':1, 'linewidth':0.1}, zorder=20)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Vertical Boundary lines separating each epochs
for i in range(len(freq_list)):
    plt.axvline(x=freq_list[i], color='grey', linestyle='--', lw = 1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

plt.xlabel(r'$f [\textbf{Hz}]$')
plt.ylabel(r'$h^2 \, \Omega_{\rm{GW}}^{(0)}(f)$')

plt.ylim(10**(-20), 5*10**(-5))
plt.xlim(10**start_freq, 10**end_freq)

plt.tick_params(direction='in')
plt.savefig("FO_GWs_spectral_energy_density.png", bbox_inches='tight', dpi = 300)
plt.show()

#.................................................................................................................................................................

aLIGO_curve = LineString(np.column_stack((X1_DATA, Y1_DATA))) # Creating a line string for the aLIGO sensitivity curve
Spectrum = LineString(np.column_stack((f, vec_Omega_GW_0(f)))) # Creating a line string for the GW spectral energy density

# Creating a line string for the horizontal BBN constraint
BBN_constraint_line = LineString([(10**start_freq, BBN_constraint), (10**end_freq, BBN_constraint)])


f_BBN = freq(1e-3) #Hz # Present frequency of GWs corresponging to BBN 


# Function to check the BBN constraint
def BBN_integ_approx(freq_list, EoS_list):
    """Function to check whether the BBN constraint is satisfied or not. This function return the piecewise integral approximation for the
    BBN constraint integral in Eq. (2.59). The expression for this equation is given in Eq. (2.64). The function returns the approximate
    value of the integral."""

    new_freq_list = [f_inf] + freq_list[:-1] + [f_BBN]
    new_EoS_list = EoS_list[:-1]

    integral = 0
    for i in range(len(new_freq_list)-1):
        f1 = new_freq_list[i]
        f2 = new_freq_list[i+1]
        w = new_EoS_list[i]


        alpha = 2/(1+3*w)

        integral += (Omega_GW_0(f1) - Omega_GW_0(f2))/(2*(1-alpha) + np.heaviside(-abs(w-1/3), 1)) + Omega_GW_0(f1) * np.log(f1/f2) * np.heaviside(-abs(w-1/3), 1)

    return integral


# The 'weaker' BBN bound on Omega_GW from BBN observations
def BBN_bound(EoS_list):
    """Function to check if the WEAKER BBN bound is satisfied. This function returns True if the BBN bound is satisfied, and False otherwise
    [see Eq. (3.1)]"""

    Omega_GW_inf = Omega_GW_0(f_inf)
    w1 = EoS_list[0]

    if EoS_list[0] >= 0.34:
        alpha1 = 2/(1+3*w1)
        bound = (1 - alpha1) * 2.26*10**(-6)
    else:
        alpha1 = 2/(1+3*0.34)
        bound = (1 - alpha1) * 2.26*10**(-6)

    return Omega_GW_inf <= bound



formatted_Energy_list = [f'{Energy:.2e}' for Energy in Energy_list] # Formatting the energy scales to be displayed

print('-------------------------------------------------------------------------------------------------------------------------------------------------------')
print('')
print(f'The equation of state parameters during reheating provided are: {EoS_list[:-2]}')

print(f'The energy scales corresponding to the end of epochs during reheating (except last epoch) provided are: {formatted_Energy_list} GeV')
print('')
try:
    r
    print(f'Tensor-to-scalar ratio, r                   = {r}')
    print(f'The energy scale during inflation, E_inf    = {E_inf:.2e} GeV')
except NameError:
    print(f'The Energy scale during inflation, E_inf    = {E_inf:.2e} GeV')

try:
    E_Rh
    print(f'The energy scale at the end of reheating, E_Rh  = {E_Rh:.2e} GeV')
except NameError:
    pass

print(f'The temperature at the end of reheating, T_Rh   = {T_Rh:.2e} GeV')

print('')

if intersects(Spectrum, aLIGO_curve):
    print('The GW spectral energy density curve intersects the aLIGO sensitivity curve.')
else:
    print('The GW spectral energy density curve does not intersect the aLIGO sensitivity curve.')


if BBN_method == 'piecewise':
    integral = BBN_integ_approx(freq_list, EoS_list)
    if integral < 1.13*10**(-6):
        print(f'The piecewise integral approximation for the BBN constraint is satisfied: {integral:.2e} < 1.13*10^(-6)')
    else:
        print(f'The piecewise integral approximation for the BBN constraint is not satisfied: {integral:.2e} > 1.13*10^(-6)')

elif BBN_method == 'intersection':
    if intersects(Spectrum, BBN_constraint_line):
        print('The GW spectral energy density curve intersects the BBN constraint bound.')
    else:
        print('The GW spectral energy density curve does not intersect the BBN constraint bound.')

elif BBN_method == 'weaker':
    if BBN_bound(EoS_list):
        print('The weaker BBN constraint is satisfied.')
    else:
        print('The weaker BBN constraint is not satisfied.')


print('')
print('--------------------------------------------------------------------------------------------------------------------------------------------------------')

