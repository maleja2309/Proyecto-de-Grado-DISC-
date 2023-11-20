
from utils import *
from config import *
from corriente import *
from velocidad import *
from vor import *
import time
from joblib import Parallel, delayed

# ====== Actualizar la vorticidad en las esquinas ====== #
s = time.time() 
vorticidad[0,-1] = 3*U/delta_x
vorticidad[0,0] = 3*U/delta_x

stream_data = get_streamfunction(fun_corriente, num_nodos)
vorticidad = act_vorticidad(vorticidad, stream_data, delta_x, delta_y, [0,0,U,0])

def int(_n, factor, i, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq):
    # Encontrar la posición del nodo central 
    fila_i, columna_i = find_c_f(num_nodos, _n[i])

    # Encontrar vorticidades
    vor = vorticidad[fila_i, columna_i]
    
    # Encontrar velocidades de los nodos 
    mul_f_v = vel_values.iloc[np.zeros(5)+_n[i]]*factor

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

def int_2(_n, factor, i, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq):

    # Encontrar la posición del nodo central 
    fila_i, columna_i = find_c_f(num_nodos, _n[i])

    # Encontrar vorticidades
    vor = vorticidad[fila_i, columna_i]

    # Encontrar velocidades de los nodos 
    mul_f_v = vel_values.iloc[np.zeros(5)+_n[i]]*factor

    # Encontrar la g de equilibrio 
    g_e_c = vor/5*(1 + 2.5*(np.array(mul_f_v.sum(axis=1)))/c)

    g_k_t_eq[fila_i, columna_i] = g_e_c[0]
    g_k_l_eq[fila_i, columna_i] = g_e_c[2]
    g_k_r_eq[fila_i, columna_i] = g_e_c[1]
    g_k_b_eq[fila_i, columna_i] = g_e_c[3]
    g_k_c_eq[fila_i, columna_i] = g_e_c[4]

def act_fron(_n, factor):
    global g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq
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
    
    Parallel(n_jobs=16,  require='sharedmem')(delayed(int)(_n, factor, x, vorticidad, g_k_t, g_k_l, g_k_r, g_k_b, g_k_c, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in range(1,len(_n)))


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

    Parallel(n_jobs=p,  require='sharedmem')(delayed(int_2)(_n, factor, x, vorticidad, g_k_t_eq, g_k_l_eq, g_k_r_eq, g_k_b_eq, g_k_c_eq) for x in range(1,len(_n)))

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

if __name__ == '__main__':
    actualizar_fronteras()
    
    esquinas()
    actualizar_g_k_e()
    calculo_g_k()
    
    # Actualizar matriz de vorticidad 

    vorticidad[1:-1,1:-1] = (g_k_t + g_k_l + g_k_r + g_k_b + g_k_c)[1:-1,1:-1]

    equations = generate_sys_eq(num_nodos_2, dimension, config_2, conf_sol_2, sys_config)

    inv_A = np.linalg.inv(equations)

    num_iter = 100
    for i in range(num_iter):
        print(i)
        actualizar_vel()

        actualizar_g_k_e()
        actualizar_fronteras()
        calculo_g_k()

        # Actualizar matriz de vorticidad 
        vorticidad[1:-1,1:-1] = (g_k_t + g_k_l + g_k_r + g_k_b + g_k_c)[1:-1,1:-1]

        # Configuración parcial [ψ]
        vorticidad_2 = vorticidad[1:-1,1:-1]
        vector_2 = vorticidad_(vorticidad_2, vector, 2)*delta_x**2

        fun_corriente[1:-1,1:-1] = solve_system(inv_A, vector_2).reshape(n_-2, n_-2)

        stream_data = get_streamfunction(fun_corriente, num_nodos)
        vorticidad = act_vorticidad(vorticidad, stream_data, delta_x, delta_y, [0,0,U,0])
    e = time.time()
    print(e-s)
    graficar_f_corriente(vel_values, n_)

    pd.DataFrame(g_k_t).to_csv('./g_k_t.csv')
