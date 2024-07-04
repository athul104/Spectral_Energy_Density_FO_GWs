#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#					                     	          First-order Gravitational Wave Energy Density Spectrum
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Owner:         Athul K. Soman 
# Collaborators: Swagat S. Mishra, Mohammed Shafi, Soumen Basak
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

# This python script is to compute the energy density spectrum of first-order inflationary gravitational waves (GWs) for a multi-epoch reheating scenario. The 
# transition from one epoch to the next epoch is assumed to be instantaneous. This code is associated with the paper "Inflationary Gravitational Waves as a 
# probe of the unknown post-inflationary primordial Universe".

# The code plots the image and saves it to the same folder as the code. The instructions to provide the inputs are given in the comments. 
# Your inputs must be provided only in the section "YOU HAVE TO PROVIDE INPUT ONLY HERE". 



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# The required packages are numpy, matplotlib, scipy, mpmath, shapely, and fractions.
# Importing the required packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from matplotlib import rcParams
from matplotlib import rc
import matplotlib as mpl
import mpmath
from fractions import Fraction
from shapely.geometry import LineString
from shapely import intersects

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#font
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

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



# YOU HAVE TO PROVIDE INPUT ONLY HERE
#################################################################################################################################################################

# PROVIDE THE FOLDER PATH where the code is saved.
Folder_path = 'C:/Users/ATHUL/Cosmology/Major Project/Codes/Phase 2'


# Provide the equation of state of each epochs during reheating in the following list in order from the EARLIEST epoch to the LATEST epoch.
# The values must be between -0.28 and 0.99.
EoS_list = [0.8, Fraction(1,3), 0.5] # [w_1, w_2, w_3, ..., w_n]  


# Provide the energy scales marking the end of each epoch of reheating in GeV in the following list (Do not include the energy scale at the end of the last epoch).
# If there is only one epoch of reheating, provide an empty list.
# For example, if you have 3 epochs during reheating, you have to provide 2 values in the list.
# The values must be greater than or equal to ~10^2 GeV **.
Energy_list = [10**8, 10**5] #GeV #[E_1, E_2, ...., E_{n-1}]


# Provide the temperature acheived at the end of reheating in GeV.
# In accordance to the BBN constarint, the temperature at the end of reheating must be greater than 10^(-3) GeV, i.e., T_Rh > T_BBN = 10^(-3) GeV
T_Rh = 0.45 #GeV


# Provide either the value of tensor-to-scalar ratio 'r', or the value of energy scale during inflation 'E_inf' in GeV. Comment the other one out.
r = 0.001
# E_inf = 10**16 #GeV


# Choose the method you want to use for checking the BBN constraint.
# 1. If you want to cross check the results we presented or want to use the weaker constraint which we have used in our paper [see Eq. (3.59)], provide 'weaker'
# 2. If you are changing the values of 'r' (or 'E_inf') and/or 'T_Rh' from the values we have used in our paper, we advise you to use the intersection method, 
#    which checks whether the GW energy density spectrum intersects with the BBN constraint curve. Provide 'intersection' for this.
BBN_method = 'weaker' #['weaker', 'intersection']

# Provide the number of data points you want in the plot of GW energy density spectrum
num_of_points = 1000


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# **: This is because above E >= 10^2 GeV, the relativistic degrees of freedom are almost constant, g_star = g_s = 106.75, and we use 
#     this value in converting the energy scales to temperature in GeV. If the energy scale is less than 10^2 GeV, the data will be inaccurate.
#################################################################################################################################################################




# defining the values of the constants
Omega_rad_0 = 4.16*10**(-5)                         # Present radiation density parameter (This is actually Omega_{rad, 0}*h^2) [see Eq. (E.7)]
T_0 = 2.35*10**(-13)     #GeV                         Present temperature [https://arxiv.org/abs/0911.1955]
m_P = 2.44*10**18        #GeV                         Reduced Planck mass
BBN_constraint = 1.13*10**(-6)                      # Upper bound on Omega_GW from BBN observations [see Eq. (3.61)]
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

        if E_inf >= 1.39 * 10**16: #GeV # The current upper bound on the energy scale during inflation [see Eq. (2.27)]
            print('The value of energy scale during inflation E_inf is greater than the current upper bound. Please provide a lower value')
            exit()

        H_inf = E_inf**2 / (np.sqrt(3) * m_P)       #GeV    # Hubble parameter during inflation (in GeV)
    
    except NameError:
        print('Please provide either the value of tensor-to-scalar ratio r, or the value of energy scale during inflation E_inf in GeV')
        exit()
   
#.................................................................................................................................................................

#Checking the validity of the equations of state provided
for i in range(len(EoS_list)):
    if EoS_list[i] <= -1/3 or EoS_list[i] >= 1:
        print(f'The equation of state given in position {i+1} of EoS_list is invalid. EoS must be between -1/3 and 1')
        exit()

#checking the validity of the energy scales provided
for i in range(len(Energy_list)):
    if Energy_list[i] < 10**2:
        print(f'The code is not designed for the scale given in position {i+1}. Energy scale must be greater than 10^2 GeV')
        exit()

#checking the validity of the temperature at the end of reheating
if T_Rh < 10**(-3):
    print('The temperature at the end of reheating must be greater than 10^(-3) GeV')
    exit()

#checking the validity of the energy scale during inflation
if E_inf < Energy_list[0]:
    print('The energy scale during inflation is less than the energy scale at the end of the first epoch of reheating')
    print('Please provide a higher value for tensor-to-scalar ratio r')
    exit()

#.................................................................................................................................................................

# Loading the data for the effective relativistic degrees of freedom in energy and entropy. The data is in the form of [Temperature in GeV, g_star, g_s]. 
Eff_rel_dof_data = np.loadtxt(Folder_path + '/eff_rel_dof.txt')

# Converting the data to numpy arrays
Temp_in_GeV = np.array(Eff_rel_dof_data[:,0])
Eff_rel_dof_in_energy = np.array(Eff_rel_dof_data[:,1])
Eff_rel_dof_in_entropy = np.array(Eff_rel_dof_data[:,2])

#.................................................................................................................................................................

def Temp(E):
    """Function to convert energy scale (E >= 10**2 GeV) of universe to effective temperature in GeV
    [See Eq. (3.53)]"""
    # for E >= 10^2 GeV, g_* = g_s = 106.75
    return E/(np.pi**2 * 106.75/ 30)**(1/4)

vec_Temp = np.vectorize(Temp)   # Vectorizing the function Temp

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

def freq(T):
    """Function for converting temperature in GeV to present-frequency of GWs in Hz.
    [See Eq. (3.52)]"""
    return 7.43 * 10**(-8) * (g_s_k(T_0)/g_s_k(T))**(1/3) * (g_star_k(T)/90)**(1/2) * T


def E(T):
    """Function to convert temperature in GeV to energy scale of the universe in GeV.
    [See Eq. (3.53)]"""
    return T*(np.pi**2 * g_star_k(T)/ 30)**(1/4)

#.................................................................................................................................................................

# Converting the energy scales provided into effective temperatures in GeV

if len(Energy_list) == 0: # If reheating has only single epoch 
    Temperature_list = [] #GeV

else:
    Temperature_list = vec_Temp(Energy_list) #GeV
    Temperature_list = Temperature_list.tolist()

Temperature_list.append(T_Rh) #GeV # adding the end of reheating temperature

if Temperature_list[-1] > Temperature_list[-2]:
    print('The temperature at the end of reheating you provided is greater than the effective temperature at the end of the second last epoch of reheating')
    exit()

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

    A_k_arr[0] = 2**(alpha_arr[0]- 1/2 ) * gamma(alpha_arr[0] + 1/2)    #A_{k, 1} [see Eq. (3.36)]
    B_k_arr[0] = 0                                                      #B_{k, 1} [see Eq. (3.36)]

    for i in range(1, len(alpha_arr)):
        an_ym = alpha_arr[i] * y_arr[i-1]       # \alpha_n * y_m, where m = n-1 [see Eq. (3.20), (3.21)]
        am_ym = alpha_arr[i-1] * y_arr[i-1]     # \alpha_m * y_m, where m = n-1 [see Eq. (3.20), (3.21)]

        an_m_half = alpha_arr[i] - 1/2          # \alpha_n - 1/2
        am_m_half = alpha_arr[i-1] - 1/2        # \alpha_m - 1/2

        an_p_half = alpha_arr[i] + 1/2          # \alpha_n + 1/2
        am_p_half = alpha_arr[i-1] + 1/2        # \alpha_m + 1/2

        C = ((an_ym)**(an_m_half))/((am_ym)**(am_m_half)) # The coefficient in Eq. (3.18) and (3.19)

        f_1 = mpmath.besselj(-(an_m_half), an_ym) #[see Eq. (3.16)]
        f_2 = mpmath.besselj(-(am_m_half), am_ym) #[see Eq. (3.17)]
        f_3 = mpmath.besselj(-(an_p_half), an_ym) #[see Eq. (3.18)]
        f_4 = mpmath.besselj(-(am_p_half), am_ym) #[see Eq. (3.19)]

        g_1 = mpmath.besselj(an_m_half, an_ym) #[see Eq. (3.16)]
        g_2 = mpmath.besselj(am_m_half, am_ym) #[see Eq. (3.17)]
        g_3 = mpmath.besselj(an_p_half, an_ym) #[see Eq. (3.18)]
        g_4 = mpmath.besselj(am_p_half, am_ym) #[see Eq. (3.19)]
        
        
        Deno = f_1 * g_3 + g_1 * f_3 # Denominator in Eq. (3.20) and (3.21)

        K = C / Deno

        Num_A1 = g_2 * f_3 + g_4 * f_1
        Num_B1 = f_2 * f_3 - f_4 * f_1

        Num_A2 = g_2 * g_3 - g_4 * g_1
        Num_B2 = f_2 * g_3 + f_4 * g_1


        A_k_arr[i] = K * (A_k_arr[i-1] * Num_A1 + B_k_arr[i-1] * Num_B1) # [see Eq. (3.20)]
        B_k_arr[i] = K * (A_k_arr[i-1] * Num_A2 + B_k_arr[i-1] * Num_B2) # [see Eq. (3.21)]

    return A_k_arr[-1], B_k_arr[-1]

#.................................................................................................................................................................

# Relativistic correction factor at beginning of last RD epoch
G_R = (g_star_k(Temperature_list[-2])/ g_star_k(T_0)) * (g_s_k(T_0)/g_s_k(Temperature_list[-2]))**(4/3) # (g_{*, r*}/g_{*, 0}) * (g_{s, 0}/g_{s, r*})^{4/3} 
#[see Eq. (3.55)]

const_coeff = 1/(96*(np.pi)**3) * G_R * Omega_rad_0 * (H_inf/m_P)**2 # The coefficient in Eq. (3.55)

#Function for calculating the energy density spectrum at present time
def Omega_GW_0(f):
    """Function to calculate the present energy density spectrum of GWs for a given frequency f.
    [See Eq. (3.55)]"""

    y_eq = f/freq_list[-1] # y_eq = f/f_eq
    

    return const_coeff * y_eq**(-2) * (coeff(f)[0]**2 + coeff(f)[1]**2)

vec_Omega_GW_0 = np.vectorize(Omega_GW_0)   #vectorizing the function Omega_GW_0

#.................................................................................................................................................................

#Loading the data for the sensitive curves. These are power-law integrated sensitivity curves (PLIS) for various GW detectors (except Planck)
# obtained from https://zenodo.org/records/3689582 in relation to the paper https://arxiv.org/abs/2002.04615. 

# Column 1: Log10 of the gravitational-wave frequency f, redshifted to today, in units of Hz
# Column 2: Log10 of the gravitational-wave energy density power spectrum h^2\Omega_PLIS

DATA1 = np.loadtxt(Folder_path + "/PLIS/aLIGO.txt")
DATA2 = np.loadtxt(Folder_path + "/PLIS/DECIGO.txt")
DATA3 = np.loadtxt(Folder_path + "/PLIS/LISA.txt")
DATA4 = np.loadtxt(Folder_path + "/PLIS/IPTA.txt")
DATA5 = np.loadtxt(Folder_path + "/PLIS/BBO.txt")
DATA6 = np.loadtxt(Folder_path + "/PLIS/Planck.txt")
DATA7 = np.loadtxt(Folder_path + "/PLIS/LV.txt")
DATA8 = np.loadtxt(Folder_path + "/PLIS/SKA.txt")
DATA9 = np.loadtxt(Folder_path + "/PLIS/PPTA.txt")
DATA10 = np.loadtxt(Folder_path + "/PLIS/ET.txt")
DATA11 = np.loadtxt(Folder_path + "/PLIS/CE.txt")
DATA12 = np.loadtxt(Folder_path + "/PLIS/LVK.txt")
DATA13 = np.loadtxt(Folder_path + "/PLIS/LVO2.txt")
DATA14 = np.loadtxt(Folder_path + "/PLIS/EPTA.txt")
DATA15 = np.loadtxt(Folder_path + "/PLIS/NANOGrav.txt")

# Separating the data into frequency and energy density spectrum
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
# Plotting the GW energy density spectrum

f_inf = freq(Temp(E_inf)) #Hz # Present frequency corresponding to the energy scale during inflation

start_freq = np.log10(2*10**(-20))  #Hz      #starting frequency for the plot in Log10 scale
end_freq = np.log10(f_inf)      #Hz      #end frequency for the plot in Log10 scale
f = np.logspace(start_freq, end_freq, num_of_points, endpoint=True, base=10.0) #Hz #frequency range for the plot
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
plt.axhline(y=BBN_constraint, color='royalblue', linestyle=':', lw = 1)
plt.text(10e-7,12e-7, r"\textbf{BBN Constraint}", style='normal', fontsize=10, color='royalblue',
    verticalalignment='center', horizontalalignment='center', bbox={'facecolor': 'white', 'edgecolor': 'black',
                                                                     'pad': 0.3, 'boxstyle': 'round', 'linewidth': 0.5})

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
    plt.text(x = Center_arr[i], y = 5*10**(-20), s=r"$w_{"+ f'{i+1}' +"}$ = " + f'$\mathbf{{{EoS_list[i]}}}$', rotation=rot_new, fontsize=15
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
    plt.axvline(x=freq_list[i], color='purple', linestyle=':', lw = 0.5)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

plt.xlabel(r'$f [\textbf{Hz}]$')
plt.ylabel(r'$h^2 \, \Omega_{\rm{GW}}^{(0)}(f)$')

plt.ylim(10**(-20), 10**(-5))
plt.xlim(10**start_freq, 10**end_freq)

plt.tick_params(direction='in')
plt.savefig(Folder_path + "/FO_GWs_energy_density_spectrum.png", bbox_inches='tight', dpi = 300)
plt.show()

#.................................................................................................................................................................


# To check whether the GW energy density spectrum intersects the aLIGO sensitivity curve
aLIGO_curve = LineString(np.column_stack((X1_DATA, Y1_DATA))) # Creating a line string for the aLIGO sensitivity curve
Spectrum = LineString(np.column_stack((f, vec_Omega_GW_0(f)))) # Creating a line string for the GW energy density spectrum

# Creating a line string for the horizontal BBN constraint
BBN_constraint_line = LineString([(10**start_freq, BBN_constraint), (10**end_freq, BBN_constraint)])


# The weak BBN bound on Omega_GW from BBN observations
def BBN_bound(EoS_list):
    """Function to check if the BBN bound is satisfied. This function returns True if the BBN bound is satisfied, and False otherwise
    [see Eq. (3.59)]"""

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

print(f'The temperature at the end of reheating, T_Rh   = {T_Rh} GeV')

print('')

if intersects(Spectrum, aLIGO_curve):
    print('The GW energy density spectrum intersects the aLIGO sensitivity curve.')
else:
    print('The GW energy density spectrum does not intersect the aLIGO sensitivity curve.')

if BBN_method == 'intersection':
    if intersects(Spectrum, BBN_constraint_line):
        print('The GW energy density spectrum intersects the BBN constraint bound.')
    else:
        print('The GW energy density spectrum does not intersect the BBN constraint bound.')

elif BBN_method == 'weaker':
    if BBN_bound(EoS_list):
        print('The BBN constraint is satisfied.')
    else:
        print('The BBN constraint is not satisfied.')

print('')
print('--------------------------------------------------------------------------------------------------------------------------------------------------------')
