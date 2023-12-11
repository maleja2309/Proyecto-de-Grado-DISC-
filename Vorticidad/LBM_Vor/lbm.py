from config import c, Re, n_i, n_t, n_l, n_r, n_b, num_nodos, e_I, vorticidad, vel_values, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_c_eq, g_k_l_eq, g_k_b_eq, g_k_r_eq, g_k_t_eq, config_, p
from utils import *
from main import vorticidad

# ===================================================================================================
# Cálculo de g_k_e en la frontera superior 
# ===================================================================================================
def int(_n, factor, i, vorticidad):

    # Encontrar la posición del nodo central 
    fila_i, columna_i = find_c_f(num_nodos, _n[i])

    # Encontrar vorticidades
    vor = vorticidad[fila_i, columna_i]
    print(num_nodos[fila_i, columna_i])

    # Encontrar velocidades de los nodos 
    mul_f_v = vel_values.iloc[np.zeros(5)+_n[i]]*factor

    # Encontrar la g de equilibrio 
    g_e_c = vor/5*(1 + 2.5*(mul_f_v.sum(axis=1))/c)

    return g_e_c

def act_fron(_n, factor):

    # Encontrar la posición del nodo central 
    fila_i, columna_i = find_c_f(num_nodos, _n[0])
    
    # Encontrar vorticidades
    vor = vorticidad[fila_i, columna_i]
   
    # Encontrar velocidades de los nodos 
    mul_f_v = vel_values.iloc[np.zeros(5)+_n[0]]*factor
    
    # Encontrar la g de equilibrio 
    g_e = vor/5*(1 + 2.5*(mul_f_v.sum(axis=1))/c)
  
    g_e = np.append(g_e, Parallel(n_jobs=p, backend="multiprocessing")(delayed(int)(_n, factor, x, vorticidad) for x in range(1,len(_n))))
  
    g_eq = g_e.reshape(len(_n), config_)

    fila_i, columna_i = find_c_f(num_nodos, _n)
    g_k_t[fila_i, columna_i] = g_eq[:,0]
    g_k_l[fila_i, columna_i] = g_eq[:,1]
    g_k_r[fila_i, columna_i] = g_eq[:,2]
    g_k_b[fila_i, columna_i] = g_eq[:,3]
    g_k_c[fila_i, columna_i] = g_eq[:,4]
 
    g_k_t_eq[fila_i, columna_i] = g_eq[:,0]
    g_k_l_eq[fila_i, columna_i] = g_eq[:,1]
    g_k_r_eq[fila_i, columna_i] = g_eq[:,2]
    g_k_b_eq[fila_i, columna_i] = g_eq[:,3]
    g_k_c_eq[fila_i, columna_i] = g_eq[:,4]

# ===================================================================================================
# Cálculo de g_k_e en las fronteras
# ===================================================================================================
def actualizar_fronteras():
    act_fron(n_t, e_I)
    #act_fron(n_l, e_I)
    #act_fron(n_r, e_I)
    #act_fron(n_b, e_I)

# ===================================================================================================
# Cálculo de g_k_e las esquinas
# ===================================================================================================
def esquinas():
    g_k_t[0,-1] = g_k_t[0, 1]
    g_k_t[0,0] = g_k_t[0, 1]

    g_k_t_eq[0,-1] = g_k_t[0, 1]
    g_k_t_eq[0,0] = g_k_t[0, 1]

    g_k_l[0,-1] = g_k_l[0,1]
    g_k_l[0,0] = g_k_l[0,1]

    g_k_l_eq[0,-1] = g_k_l[0,1]
    g_k_l_eq[0,0] = g_k_l[0,1]

    g_k_r[0,-1] = g_k_r[0,1]
    g_k_r[0,0] = g_k_r[0,1]

    g_k_r_eq[0,-1] = g_k_r[0,1]
    g_k_r_eq[0,0] = g_k_r[0,1]

    g_k_b[0,-1] = g_k_b[0,1]
    g_k_b[0,0] = g_k_b[0,1]

    g_k_b_eq[0,-1] = g_k_b[0,1]
    g_k_b_eq[0,0] = g_k_b[0,1]

    g_k_c[0,-1] = -vorticidad[0,-1]/5
    g_k_c[0,0] = -vorticidad[0,0]/5

    g_k_c_eq[0,-1] = -vorticidad[0,-1]/5
    g_k_c_eq[0,0] = -vorticidad[0,0]/5

# ===================================================================================================
# Cálculo de g_k_e en los nodos interiores 
# ===================================================================================================
def actualizar_g_k_e():
    _n = n_i.flatten()
    factor = e_I

    nearest = find_nearest(num_nodos, _n)
    nodes = np.append(nearest[0], _n[0])

    # Encontrar la posición del nodo central 
    fila_i, columna_i = find_c_f(num_nodos, _n[0])

    # Encontrar vorticidades
    vor = vorticidad[fila_i, columna_i]

    # Encontrar velocidades de los nodos 
    mul_f_v = vel_values.iloc[np.zeros(5)+_n[0]]*factor

    # Encontrar la g de equilibrio 
    g_e = vor/5*(1 + 2.5*(mul_f_v.sum(axis=1))/c)

    g_e = np.append(g_e, Parallel(n_jobs=p, backend="multiprocessing")(delayed(int)(_n, factor, x, vorticidad) for x in range(1,len(_n))))

    g_eq = g_e.reshape(len(_n),len(nodes))

    fila_i, columna_i = find_c_f(num_nodos, _n)
    g_k_t_eq[fila_i, columna_i] = g_eq[:,0]
    g_k_l_eq[fila_i, columna_i] = g_eq[:,2]
    g_k_r_eq[fila_i, columna_i] = g_eq[:,1]
    g_k_b_eq[fila_i, columna_i] = g_eq[:,3]
    g_k_c_eq[fila_i, columna_i] = g_eq[:,4]

# ===================================================================================================
# Cálculo de g_k en los nodos interiores
# ===================================================================================================
def calculo_g_k():
    _n = n_i 

    nearest = find_nearest(num_nodos, _n)

    f_t, c_t = find_c_f(num_nodos, nearest[:,0])
    f_l, c_l = find_c_f(num_nodos, nearest[:,1])
    f_r, c_r = find_c_f(num_nodos, nearest[:,2])
    f_b, c_b = find_c_f(num_nodos, nearest[:,3])
    f_i, c_i = find_c_f(num_nodos, _n)

    g_k_t[f_i, c_i] = g_k_t[f_t, c_t] - 1/tau_reynolds(Re,c)*(g_k_t[f_t, c_t] - g_k_t_eq[f_t, c_t])
    g_k_l[f_i, c_i] = g_k_l[f_l, c_l] - 1/tau_reynolds(Re,c)*(g_k_l[f_l, c_l] - g_k_l_eq[f_l, c_l])
    g_k_r[f_i, c_i] = g_k_r[f_r, c_r] - 1/tau_reynolds(Re,c)*(g_k_r[f_r, c_r] - g_k_r_eq[f_r, c_r])
    g_k_b[f_i, c_i] = g_k_b[f_b, c_b] - 1/tau_reynolds(Re,c)*(g_k_b[f_b, c_b] - g_k_b_eq[f_b, c_b])
    g_k_c[f_i, c_i] = g_k_c[f_i, c_i] - 1/tau_reynolds(Re,c)*(g_k_c[f_i, c_i] - g_k_c_eq[f_i, c_i])

