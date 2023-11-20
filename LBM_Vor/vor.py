import numpy as np
import pandas as pd 

from config import *
from utils import *

# Modificar las esquinas 
vorticidad[0, -1] = 3*U/delta_x
vorticidad[0, 0] = 3*U/delta_x

# =====================================================================================
# Cálculos en la frontera de la vorticidad 
# =====================================================================================
def vorticity_streamfunction(pared: int, phi_0: float, phi_1: float, phi_2: float, delta_x: float, delta_y: float, u:list = [0,0,0,0]):
    """_summary_

    Args:
        pared (int): Ubicación de la pared a la que se hace referencia
        phi_0 (float): ψ_0
        phi_1 (float): ψ_1
        phi_2 (float): ψ_2
        delta_x (float): Espaciamiento en la componente horizontal
        delta_y (float): Espaciamiento en la componente vertical
        u (list, optional): Velocidades en cada una de las fronteras [L, R, T,B]. Defaults to [0,0,0,0].

    Returns:
        _type_: _description_
    """

    w = 0
    # 1. Pared derecha
    if (pared == 1):    
        w = (7*phi_0 - 8*phi_1 + phi_2)/(2*delta_x**2) - (3*u[1])/delta_x
    
    # 2. Pared izquierda
    elif (pared == 2):  
        w = (7*phi_0 - 8*phi_1 + phi_2)/(2*delta_x**2) + (3*u[0])/delta_x        

    # 3. Pared Superior
    elif (pared == 3):
        w = (7*phi_0 - 8*phi_1 + phi_2)/(2*delta_y**2) + (3*u[2])/delta_y

    # 4. Pared Inferior
    elif (pared == 4):
        w = (7*phi_0 - 8*phi_1 + phi_2)/(2*delta_y**2) - (3*u[3])/delta_y
    return w
   
# =====================================================================================
# Actualizar el valor de la vorticidad 
# =====================================================================================
def act_vorticidad(vorticidad, data, delta_x, delta_y, vel):
    search_fc_r = data[0]
    search_fc_l = data[1]
    search_fc_t = data[2]
    search_fc_b = data[3]

    w_r = vorticity_streamfunction(1, search_fc_r[:,2], search_fc_r[:,1], search_fc_r[:,0], delta_x, delta_y, vel)
    w_l = vorticity_streamfunction(2, search_fc_l[:,0], search_fc_l[:,1], search_fc_l[:,2], delta_x, delta_y, vel)
    w_t = vorticity_streamfunction(3, search_fc_t[0,:], search_fc_t[1,:], search_fc_t[2,:], delta_x, delta_y, vel)
    w_b = vorticity_streamfunction(4, search_fc_b[2,:], search_fc_b[1,:], search_fc_b[0,:], delta_x, delta_y, vel)
  
    vorticidad[1:-1,-1] = w_r
    vorticidad[1:-1,0] = w_l
    vorticidad[0, 1:-1] = w_t
    vorticidad[-1, 1:-1] = w_b

    return vorticidad