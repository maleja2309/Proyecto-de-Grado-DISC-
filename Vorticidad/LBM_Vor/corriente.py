# Adapted from Professor Jose Rafael Toro Gomez's code. 
# Author: María Alejandra Vargas Torres

import numpy as np
from config import *
from utils import *
from vor import *

# ==== Coeficientes para la solucion de la ecuación de Poisson ==== #

# Nodo central
conf_in = [1, 1, 1, 1, -4]           #[T,L,C,R,B]

# Nodo esquina superior derecha
conf_2 = [1, 1, -4]

# Nodo esquina superior izquierda 
conf_3 = [1, 1, -4]

# Nodo esquina inferior derecha
conf_4 = [1, 1, -4]

# Nodo esquina inferior izquierda
conf_5 =  [1, 1, -4]

# Nodo lateral izquierdo 
conf_6 = [1, 1, 1, -4]

# Nodo lateral derecho 
conf_7 =  [1, 1, 1, -4]

# Nodo superior
conf_8 =  [1, 1, 1, -4]

# Nodo inferior
conf_9 =  [1, 1, 1, -4]

conf_sol_1 = [conf_3, conf_8, conf_2,
              conf_6, conf_in, conf_7, 
              conf_5, conf_9, conf_4]

# ==== Coeficientes para la solucion de la ecuación de Poisson Modificados ==== #
#[T,L,C,R,B]
# Nodo central
conf_in_ = [1, 1, -4, 1, 1]           

# Nodo esquina superior derecha
conf_2_ = [1, -4, 1]

# Nodo esquina superior izquierda 
conf_3_ = [-4, 1, 1]

# Nodo esquina inferior derecha
conf_4_ = [1, 1, -4]

# Nodo esquina inferior izquierda
conf_5_ =  [1, -4, 1]

# Nodo lateral izquierdo 
conf_6_ = [1, -4, 1, 1]

# Nodo lateral derecho 
conf_7_ = [1, 1, -4, 1]

# Nodo superior
conf_8_ =  [1, -4, 1, 1]

# Nodo inferior
conf_9_ =  [1, 1, -4, 1]

conf_sol_2 = [conf_3_, conf_8_, conf_2_,
              conf_6_, conf_in_, conf_7_, 
              conf_5_, conf_9_, conf_4_]


# =====================================================================================
# Encontrar los valores de la función de corriente según la posición 
# =====================================================================================
def get_streamfunction(fun_corriente, num_nodos):
    
    if (len(num_nodos[0])> 2):
        phi_0_r = n_r 
        phi_1_r = n_r - 1
        phi_2_r = n_r - 2
        data_r = np.array([phi_0_r, phi_1_r, phi_2_r], dtype=np.float64)

        phi_0_l = n_l 
        phi_1_l = n_l + 1
        phi_2_l = n_l + 2
        data_l = np.array([phi_0_l, phi_1_l, phi_2_l], dtype=np.float64)

        phi_0_t = n_t 
        phi_1_t = n_t + n_
        phi_2_t = n_t + 2*n_ 
        data_t = np.array([phi_0_t, phi_1_t, phi_2_t], dtype=np.float64)

        phi_0_b = n_b 
        phi_1_b = n_b - n_
        phi_2_b = n_b - 2*n_ 
        data_b = np.array([phi_0_b, phi_1_b, phi_2_b], dtype=np.float64)

        fila_r, column_r = find_c_f(num_nodos, data_r)
        fila_l, column_l = find_c_f(num_nodos, data_l)
        fila_t, column_t = find_c_f(num_nodos, data_t)
        fila_b, column_b = find_c_f(num_nodos, data_b)      
    
    # Buscar en la matriz de la función de corriente 
    search_fc_r = fun_corriente[fila_r, column_r].reshape(n_-2,3) 
    search_fc_l = fun_corriente[fila_l, column_l].reshape(n_-2,3) 
    search_fc_t = fun_corriente[fila_t, column_t].reshape(3, n_-2)
    search_fc_b = fun_corriente[fila_b, column_b].reshape(3, n_-2)
    return [search_fc_r, search_fc_l, search_fc_t, search_fc_b]


def act_fron_t_c(_n, f_k_t, f_k_l, f_k_r, f_k_b, f_k_c, f_k_t_eq, f_k_l_eq, f_k_r_eq, f_k_b_eq, f_k_c_eq):

    # Encontrar la posición del nodo central 
    fila_i, columna_i = find_c_f(num_nodos, _n[0])

    # Encontrar vorticidades
    cor = fun_corriente[fila_i, columna_i]

    # Encontrar la g de equilibrio 
    g_e = cor*zita

    for i in range(1,len(_n)):

        # Encontrar la posición del nodo central 
        fila_i, columna_i = find_c_f(num_nodos, _n[i])

        # Encontrar vorticidades
        cor = fun_corriente[fila_i, columna_i]

        # Encontrar la g de equilibrio 
        g_e_c = cor*zita
        
        g_e = np.append(g_e, g_e_c)
        
    g_eq = g_e.reshape(len(_n), 5)
    
    fila_i, columna_i = find_c_f(num_nodos, _n)
    f_k_t[fila_i, columna_i] = g_eq[:,0]
    f_k_l[fila_i, columna_i] = g_eq[:,1]
    f_k_r[fila_i, columna_i] = g_eq[:,2]
    f_k_b[fila_i, columna_i] = g_eq[:,3]
    f_k_c[fila_i, columna_i] = g_eq[:,4]

    f_k_t_eq[fila_i, columna_i] = g_eq[:,0]
    f_k_l_eq[fila_i, columna_i] = g_eq[:,1]
    f_k_r_eq[fila_i, columna_i] = g_eq[:,2]
    f_k_b_eq[fila_i, columna_i] = g_eq[:,3]
    f_k_c_eq[fila_i, columna_i] = g_eq[:,4]
    
