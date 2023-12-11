## file Config.py
## Configuación 
import os

# Número de procesadores
p = 2
# set number of CPUs to run on
# ncore = "1"

# # set env variables
# # have to set these before importing numpy
# os.environ["OMP_NUM_THREADS"] = ncore
# os.environ["OPENBLAS_NUM_THREADS"] = ncore
# os.environ["MKL_NUM_THREADS"] = ncore
# os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
# os.environ["NUMEXPR_NUM_THREADS"] = ncore

import numpy as np
import pandas as pd 

# Condiciones iniciales
U = 1                               # Velocidad del movimiento de la tapa 

# Condiciones de frontera Poisson
df_dx_l = 0                         # ψ'_x  Pared izquierda
df_dx_r = 0                         # ψ'_x  Pared derecha
df_dy_t = 0                         # ψ'_y  Pared superior
df_dy_b = 0                         # ψ'_y  Pared inferior

# Configuración 
config_ = 5                          # D2Q5

# Variables corriente
error = 10**(-4)
deltaW = 1
taoF = 1

# Especificación del número de nodos que se tienen
n_ = 20
n = n_**2

num_nodos = np.arange(0,n).reshape(n_, n_)
num_nodos_2 = np.arange(0,(n_-2)**2).reshape(n_-2, n_-2)
fun_corriente = np.zeros((n_, n_)) 
Fo = fun_corriente.copy()

g_k_t = np.zeros((n_, n_)) 
g_k_l = np.zeros((n_, n_)) 
g_k_r = np.zeros((n_, n_)) 
g_k_b = np.zeros((n_, n_)) 
g_k_c = np.zeros((n_, n_)) 

g_k_t_eq = np.zeros((n_, n_)) 
g_k_l_eq = np.zeros((n_, n_)) 
g_k_r_eq = np.zeros((n_, n_)) 
g_k_b_eq = np.zeros((n_, n_)) 
g_k_c_eq = np.zeros((n_, n_)) 

f_k_t = np.zeros(((n_, n_)))
f_k_l = np.zeros(((n_, n_)))
f_k_r = np.zeros(((n_, n_)))
f_k_b = np.zeros(((n_, n_)))
f_k_c = np.zeros(((n_, n_)))

f_k_t_ = np.zeros(((n_, n_)))
f_k_l_ = np.zeros(((n_, n_)))
f_k_r_ = np.zeros(((n_, n_)))
f_k_b_ = np.zeros(((n_, n_)))
f_k_c_ = np.zeros(((n_, n_)))

f_k_t_eq = np.zeros(((n_, n_)))
f_k_l_eq = np.zeros(((n_, n_)))
f_k_r_eq = np.zeros(((n_, n_)))
f_k_b_eq = np.zeros(((n_, n_)))
f_k_c_eq = np.zeros(((n_, n_)))

zita = np.append(np.array(([0.25]*4)), -1)

vorticidad = np.zeros((n_, n_))
vector = np.array([])   

# ========== Parámetros =========== # 
delta_x = 1                          # Espaciamiento entre cada punto (grid) 
delta_y = delta_x
delta_t = 1                       # Paso de tiempo
c = delta_x / delta_t                   # Velocidad de la partícula en el fluido
Re = 200

# =====================================================================================
# Asignación de las velocidades u y v para cada nodo
# =====================================================================================
vel_values = pd.DataFrame(np.zeros((n, 2)))
vel_values.columns = ['u','v']

from utils import iden_nodes, velocity_directions
# ====== Identificación de los nodos del conjunto total ====== #

n_l, n_r, n_t, n_b, n_s_r, n_s_l, n_i_r, n_i_l, n_i = iden_nodes(num_nodos)

# ====== Identificación de los nodos intermedios para ψ ====== #

n_l_p, n_r_p, n_t_p, n_b_p, n_s_r_p, n_s_l_p, n_i_r_p, n_i_l_p, n_i_p = iden_nodes(num_nodos_2)

# ====== Asignación de las velocidades u y v para cada nodo ====== #
vel_values = pd.DataFrame(np.zeros((n, 2)))
vel_values.columns = ['u','v']

# ====== Cálculo de la dirección de las velocidades ====== #

Vel_e_per_node = {}
directions = np.arange(1,5)
e = velocity_directions(directions) # type: ignore
Vel_e_per_node['R'] = e[:,0]
Vel_e_per_node['T'] = e[:,1]
Vel_e_per_node['L'] = e[:,2]
Vel_e_per_node['B'] = e[:,3]
Vel_e_per_node['C'] = np.array([0, 0], dtype=np.float64)

# Casos para las e_k 

# Fronteras
e_L = np.array([Vel_e_per_node['T'], Vel_e_per_node['R'],
                Vel_e_per_node['B'], Vel_e_per_node['C']], dtype=np.float64)
e_R = np.array([Vel_e_per_node['T'], Vel_e_per_node['L'],
                Vel_e_per_node['B'], Vel_e_per_node['C']], dtype=np.float64)
e_T = np.array([Vel_e_per_node['L'], Vel_e_per_node['R'], 
                Vel_e_per_node['B'], Vel_e_per_node['C']], dtype=np.float64)
e_B = np.array([Vel_e_per_node['T'], Vel_e_per_node['L'], 
                Vel_e_per_node['R'], Vel_e_per_node['C']], dtype=np.float64)

# Esquinas
e_SR = np.array([Vel_e_per_node['L'], Vel_e_per_node['B'],
                 Vel_e_per_node['C']], dtype=np.float64)
e_SL = np.array([Vel_e_per_node['R'], Vel_e_per_node['B'],
                 Vel_e_per_node['C']], dtype=np.float64)
e_IR = np.array([Vel_e_per_node['T'], Vel_e_per_node['L'],
                 Vel_e_per_node['C']], dtype=np.float64)
e_IL = np.array([Vel_e_per_node['T'], Vel_e_per_node['R'],
                 Vel_e_per_node['C']], dtype=np.float64)

# Nodos Internos
e_I = np.array([Vel_e_per_node['T'], Vel_e_per_node['L'],
                Vel_e_per_node['R'], Vel_e_per_node['B'],
                Vel_e_per_node['C']], dtype=np.float64)

# ====== Asignar la velocidad a los nodos superiores ====== #
vel_values.loc[n_t, 'u'] = U # type: ignore
vel_values.loc[n_s_l, 'u'] = U
vel_values.loc[n_s_r, 'u'] = U


# ====== Generar sistema de ecuaciones ====== #
sys_config = 2
dimension = n_ - 2 

config_1 = [n_s_l, n_t, n_s_r,
          n_l, n_i, n_r, 
          n_i_l, n_b, n_i_r]

config_2 = [n_s_l_p, n_t_p, n_s_r_p,
          n_l_p, n_i_p, n_r_p, 
          n_i_l_p, n_b_p, n_i_r_p]

