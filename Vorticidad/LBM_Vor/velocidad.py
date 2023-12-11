import numpy as np
from config import *
from utils import *

# =====================================================================================
# Actualizar velocidades u y v de cada uno de los nodos  
# =====================================================================================
def actualizar_vel(fun_corriente, vel_values):
    a_ = find_nearest(num_nodos,n_i)
    t = a_[:,0]
    l = a_[:,1]
    r = a_[:,2]
    b = a_[:,3]

    f_, _c = find_c_f(num_nodos, r)
    r_ = fun_corriente[f_, _c]

    f_, _c = find_c_f(num_nodos, b)
    b_ = fun_corriente[f_, _c]

    f_, _c = find_c_f(num_nodos, l)
    l_ = fun_corriente[f_, _c]

    f_, _c = find_c_f(num_nodos, t)
    t_ = fun_corriente[f_, _c]

    f_, _c = find_c_f(num_nodos, n_i)
    c_ = fun_corriente[f_, _c]

    u = (b_ - t_ )/(2*delta_y)
    v = -(r_ - l_)/(2*delta_x)
    

    vel_values.loc[n_i.flatten(), 'u'] = u
    vel_values.loc[n_i.flatten(), 'v'] = v
