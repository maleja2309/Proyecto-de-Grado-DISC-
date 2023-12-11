import psutil
from utils import *
from config import *
from corriente import *
from velocidad import *
from vor import *
import time
from joblib import Parallel, delayed, parallel_config
import scipy.sparse.linalg as spsl
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_array, vstack

#import matplotlib.pyplot as plt
#from tqdm import tqdm

# ====== Actualizar la vorticidad en las esquinas ====== #
s = time.time() 

vorticidad[0,-1] = 3*U/delta_x
vorticidad[0,0] = 3*U/delta_x

stream_data = get_streamfunction(fun_corriente, num_nodos)
vorticidad = act_vorticidad(vorticidad, stream_data, delta_x, delta_y, [0,0,U,0])
Wo = vorticidad.copy()
hilos = 0

def int(factor, i, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq):
    # Encontrar la posición del nodo central 
    fila_i, columna_i = find_c_f(num_nodos, i)

    # Encontrar vorticidades
    vor = vorticidad[fila_i, columna_i]
    
    # Encontrar velocidades de los nodos 
    mul_f_v = vel_values.iloc[np.zeros(5)+i]*factor

    # Encontrar la g de equilibrio 
    g_e_c = vor/5*(1 + 2.5*(np.array(mul_f_v.sum(axis=1)))/c)

    g_k_t[fila_i, columna_i] = g_e_c[0]
    g_k_l[fila_i, columna_i] = g_e_c[1]
    g_k_r[fila_i, columna_i] = g_e_c[2]
    g_k_b[fila_i, columna_i] = g_e_c[3]
    g_k_c[fila_i, columna_i] = g_e_c[4]

    g_k_t_eq[fila_i, columna_i] = g_e_c[0]
    g_k_l_eq[fila_i, columna_i] = g_e_c[1]
    g_k_r_eq[fila_i, columna_i] = g_e_c[2]
    g_k_b_eq[fila_i, columna_i] = g_e_c[3]
    g_k_c_eq[fila_i, columna_i] = g_e_c[4]
    
def int_2(factor, i, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq):

    # Encontrar la posición del nodo central 
    fila_i, columna_i = find_c_f(num_nodos, i)

    # Encontrar vorticidades
    vor = vorticidad[fila_i, columna_i]

    # Encontrar velocidades de los nodos 
    mul_f_v = vel_values.iloc[np.zeros(5)+i]*factor

    # Encontrar la g de equilibrio 
    g_e_c = vor/5*(1 + 2.5*(np.array(mul_f_v.sum(axis=1)))/c)

    g_k_t_eq[fila_i, columna_i] = g_e_c[0]
    g_k_l_eq[fila_i, columna_i] = g_e_c[2]
    g_k_r_eq[fila_i, columna_i] = g_e_c[1]
    g_k_b_eq[fila_i, columna_i] = g_e_c[3]
    g_k_c_eq[fila_i, columna_i] = g_e_c[4]

def act_fron(_n, factor):
    """
    Función que actualiza las fronteras 

    Paremeters
    ----------
    _n (np.array): 
        Configuración de los nodos que se quiere estudiar

    factor (np.array): 
        Configuración de las direcciones de las velocidades
    """
    # Encontrar la posición del nodo central 
    fila_i, columna_i = find_c_f(num_nodos, _n[0])
    
    # Encontrar vorticidades
    vor = vorticidad[fila_i, columna_i]

    # Encontrar velocidades de los nodos 
    mul_f_v = vel_values.iloc[np.zeros(5)+_n[0]]*factor

    # Encontrar la g de equilibrio 
    g_e = vor/5*(1 + 2.5*(np.array(mul_f_v.sum(axis=1)))/c)
    g_k_t[fila_i, columna_i] = g_e[0]
    g_k_l[fila_i, columna_i] = g_e[1]
    g_k_r[fila_i, columna_i] = g_e[2]
    g_k_b[fila_i, columna_i] = g_e[3]
    g_k_c[fila_i, columna_i] = g_e[4]

    g_k_t_eq[fila_i, columna_i] = g_e[0]
    g_k_l_eq[fila_i, columna_i] = g_e[1]
    g_k_r_eq[fila_i, columna_i] = g_e[2]
    g_k_b_eq[fila_i, columna_i] = g_e[3]
    g_k_c_eq[fila_i, columna_i] = g_e[4]
 
    # Dividir las tareas en dos bloques
    con = np.array_split(_n, p)
    c_1 = con[0]
    c_2 = con[1]
    # c_3 = con[2]
    # c_4 = con[3]
    # c_5 = con[4]
    # c_6 = con[5]
    # c_7 = con[6]
    # c_8 = con[7]
    # c_9 = con[8]
    # c_10 = con[9]
    # c_11 = con[10]
    # c_12 = con[11]
    # c_13 = con[12]
    # c_14 = con[13]
    # c_15 = con[14]
    # c_16 = con[15]

    with parallel_config(backend='loky', inner_max_num_threads=hilos, prefer='threads'):
        Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_1)
        Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_2)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_3)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_4)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_5)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_6)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_7)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_8)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_9)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_10)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_11)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_12)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_13)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_14)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_15)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int)(factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_16)



def actualizar_fronteras():
    act_fron(n_t, e_I)
    act_fron(n_l, e_I)
    act_fron(n_r, e_I)
    act_fron(n_b, e_I)

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

    # Encontrar la posición del nodo central 
    fila_i, columna_i = find_c_f(num_nodos, _n[0])

    # Encontrar vorticidades
    vor = vorticidad[fila_i, columna_i]

    # Encontrar velocidades de los nodos 
    mul_f_v = vel_values.iloc[np.zeros(5)+_n[0]]*factor

    # Encontrar la g de equilibrio 
    g_e = vor/5*(1 + 2.5*(np.array(mul_f_v.sum(axis=1)))/c)
    g_k_t_eq[fila_i, columna_i] = g_e[0]
    g_k_l_eq[fila_i, columna_i] = g_e[2]
    g_k_r_eq[fila_i, columna_i] = g_e[1]
    g_k_b_eq[fila_i, columna_i] = g_e[3]
    g_k_c_eq[fila_i, columna_i] = g_e[4]

    # Dividir las tareas en dos bloques
    con = np.array_split(_n, p)
    c_1 = con[0]
    c_2 = con[1]
    # c_3 = con[2]
    # c_4 = con[3]
    # c_5 = con[4]
    # c_6 = con[5]
    # c_7 = con[6]
    # c_8 = con[7]
    # c_9 = con[8]
    # c_10 = con[9]
    # c_11 = con[10]
    # c_12 = con[11]
    # c_13 = con[12]
    # c_14 = con[13]
    # c_15 = con[14]
    # c_16 = con[15]


    with parallel_config(backend='loky', inner_max_num_threads=hilos, prefer='threads'):
        Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_1)
        Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_2)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_3)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_4)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_5)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_6)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_7)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_8)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_9)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_10)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_11)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_12)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_13)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_14)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_15)
        # Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in c_16)

# ===================================================================================================
# Cálculo de g_k en los nodos interiores
# ===================================================================================================
def calculo_g_k():
    global g_k_t, g_k_l, g_k_r, g_k_b, g_k_c
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

# Cálculo de g_k_e en los nodos interiores

def generate_eq(num_nodos, config, config_sol, n_):
    # Se agrega el identificador del nodo central 
    n_nodes = config
    o_matrix = num_nodos
    id = find_nearest(o_matrix, n_nodes)
    id = np.append(id, n_nodes.reshape(len(id),1),axis=1)

    col = id.flatten()
    row = (np.zeros((len(id), id[0].size)) + np.arange(len(id)).reshape(-1,1)).flatten()
    data = np.array(config_sol*len(id))
    return csr_array((data, (row, col)), shape=(len(id), (n_-2)**2))


def generate_sys_eq_sparse(num_nodos, config, conf_sol, n_):
    # ============================ Matríz ============================  #
    # Parte superior (E_L, T, E_R)
    lista_eq = generate_eq(num_nodos, config[0], conf_sol[0], n_)

    # Parte central (L_L, C, L_D)
    for i in range(1, 9):
        lista_eq = vstack((lista_eq, generate_eq(num_nodos, config[i], conf_sol[i], n_)))
    return lista_eq

# def actualizar_g_k_e_c(fun_corriente, f_k_t_eq, f_k_l_eq, f_k_r_eq, f_k_b_eq, f_k_c_eq):

#     _n = n_i.flatten()
    
#     # Encontrar la posición del nodo central 
#     fila_i, columna_i = find_c_f(num_nodos, _n[0])

#     # Encontrar vorticidades
#     cor = fun_corriente[fila_i, columna_i]
    
#     # Encontrar la g de equilibrio 
#     g_e = cor*zita

#     for i in range(1,len(_n)):

#         # Encontrar la posición del nodo central 
#         fila_i, columna_i = find_c_f(num_nodos, _n[i])

#         # Encontrar vorticidades
#         cor = fun_corriente[fila_i, columna_i]

#         # Encontrar la g de equilibrio 
#         g_e_c = cor*zita
    
#         g_e = np.append(g_e, g_e_c)

#     g_eq = g_e.reshape(len(_n),5)

#     fila_i, columna_i = find_c_f(num_nodos, _n)

#     f_k_t_eq[fila_i, columna_i] = g_eq[:,0]
#     f_k_l_eq[fila_i, columna_i] = g_eq[:,1]
#     f_k_r_eq[fila_i, columna_i] = g_eq[:,2]
#     f_k_b_eq[fila_i, columna_i] = g_eq[:,3]
#     f_k_c_eq[fila_i, columna_i] = g_eq[:,4]
    

# def resuelvepoissonAS(fun_corriente, vorticidad, Fo, Wo, deltaW, f_k_t, f_k_l, f_k_r, f_k_b, f_k_c, f_k_t_, f_k_l_, f_k_r_, f_k_b_, f_k_c_, f_k_t_eq, f_k_l_eq, f_k_r_eq, f_k_b_eq, f_k_c_eq):
#     _n = n_i 

#     nearest = find_nearest(num_nodos, _n)

#     f_t, c_t = find_c_f(num_nodos, nearest[:,0])
#     f_l, c_l = find_c_f(num_nodos, nearest[:,1])
#     f_r, c_r = find_c_f(num_nodos, nearest[:,2])
#     f_b, c_b = find_c_f(num_nodos, nearest[:,3])
#     f_i, c_i = find_c_f(num_nodos, _n)

#     while (deltaW > error):
#         omega = (delta_t*-1/4*vorticidad*c**2/2*(0.5-taoF)*taoF)[1:-1,1:-1]

#         # act_fron_t_c(n_t, e_I)
#         # act_fron_t_c(n_l, e_I)
#         # act_fron_t_c(n_r, e_I)
#         # act_fron_t_c(n_b, e_I)

#         actualizar_g_k_e_c(fun_corriente, f_k_t_eq, f_k_l_eq, f_k_r_eq, f_k_b_eq, f_k_c_eq)

#         f_k_t[f_i, c_i]  = f_k_t[f_t, c_t] - 1/taoF*(f_k_t[f_t, c_t] - f_k_t_eq[1:-1, 1:-1].flatten()) + omega.flatten()
#         f_k_l[f_i, c_i]  = f_k_l[f_l, c_l] - 1/taoF*(f_k_l[f_l, c_l] - f_k_l_eq[1:-1, 1:-1].flatten()) + omega.flatten()
#         f_k_r[f_i, c_i]  = f_k_r[f_r, c_r] - 1/taoF*(f_k_r[f_r, c_r] - f_k_r_eq[1:-1, 1:-1].flatten()) + omega.flatten()
#         f_k_b[f_i, c_i]  = f_k_b[f_b, c_b] - 1/taoF*(f_k_b[f_b, c_b] - f_k_b_eq[1:-1, 1:-1].flatten()) + omega.flatten()
#         f_k_c[f_i, c_i]  = f_k_c[f_i, c_i] - 1/taoF*(f_k_c[f_i, c_i] - f_k_c_eq[1:-1, 1:-1].flatten()) + 0
        
#         f_k_t_[f_i, c_i] = f_k_t[f_t, c_t] 
#         f_k_l_[f_i, c_i]  = f_k_l[f_l, c_l] 
#         f_k_r_[f_i, c_i]  = f_k_r[f_r, c_r] 
#         f_k_b_[f_i, c_i]  =  f_k_b[f_b, c_b] 
#         f_k_c_[f_i, c_i]  = f_k_c[f_i, c_i] 

#         fun_corriente = (f_k_l_ + f_k_r_ + f_k_b_ + f_k_t_)

#         stream_data = get_streamfunction(fun_corriente, num_nodos)
#         vorticidad = act_vorticidad(vorticidad, stream_data, delta_x, delta_y, [0,0,U,0])
    
#         dF = fun_corriente - Fo
#         dW1 = vorticidad[0,:] - Wo[0,:]
#         deltaF = np.sqrt((dF.flatten()@dF.flatten())/(fun_corriente.flatten()@fun_corriente.flatten())) 
#         deltaW = np.sqrt((dW1@dW1)/(vorticidad[0,:]@vorticidad[0,:]))
        
#         Fo = fun_corriente.copy()
#         Wo = vorticidad.copy()
       
#     return Fo, Wo

if __name__ == '__main__':
 
    actualizar_fronteras()
    
    esquinas()
    actualizar_g_k_e()
    calculo_g_k()

    # Actualizar matriz de vorticidad 
    vorticidad[1:-1,1:-1] = (g_k_t + g_k_l + g_k_r + g_k_b + g_k_c)[1:-1,1:-1]
    eq = generate_sys_eq_sparse(num_nodos_2, config_2, conf_sol_1, n_)

    I = np.eye(eq.shape[0])
    Ainv = spsolve(eq, I)

    num_iter = 500
    for i in range(num_iter):
        print(i)
        print(vorticidad[1][1])
        actualizar_vel(fun_corriente, vel_values)

        actualizar_g_k_e()
        actualizar_fronteras()
        calculo_g_k()

        # Actualizar matriz de vorticidad 
        vorticidad[1:-1,1:-1] = (g_k_t + g_k_l + g_k_r + g_k_b + g_k_c)[1:-1,1:-1]
        # Configuración parcial [ψ]
        vorticidad_2 = vorticidad[1:-1,1:-1]
        vector_2 = vorticidad_(vorticidad_2, vector, 2)*delta_x**2

        fun_corriente[1:-1,1:-1] = (Ainv @ vector_2).reshape(n_-2, n_-2)

        stream_data = get_streamfunction(fun_corriente, num_nodos)
        vorticidad = act_vorticidad(vorticidad, stream_data, delta_x, delta_y, [0,0,U,0])


    e = time.time()
    print(e-s)
    vel_values.to_csv('./velocidad.csv', index=False, index_label=False)
    pd.DataFrame(vorticidad).to_csv('./vorticidad.csv', index=False, index_label=False)
    pd.DataFrame(fun_corriente).to_csv('./fun_corriente.csv', index=False, index_label=False)

