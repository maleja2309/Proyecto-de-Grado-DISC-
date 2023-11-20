from utils import *
from config import *
from corriente import *
from velocidad import *
from vor import *
import time
from joblib import Parallel, delayed

s = time.time()   
# ====== Actualizar la vorticidad en las esquinas ====== #
vorticidad[0,-1] = 3*U/delta_x
vorticidad[0,0] = 3*U/delta_x

stream_data = get_streamfunction(fun_corriente, num_nodos)
vorticidad = act_vorticidad(vorticidad, stream_data, delta_x, delta_y, [0,0,U,0])

def act_fron_t():
    _n = n_t
    factor = e_I

    nearest = find_nearest(num_nodos, _n)
    nodes = np.append(0, nearest[0])
    nodes = np.append(nodes, _n[0])

    # Encontrar la posición del nodo central 
    fila_i, columna_i = find_c_f(num_nodos, _n[0])
    
    # Encontrar vorticidades
    vor = vorticidad[fila_i, columna_i]

    # Encontrar velocidades de los nodos 
    mul_f_v = vel_values.iloc[np.zeros(5)+_n[0]]*factor

    # Encontrar la g de equilibrio 
    g_e = vor/5*(1 + 2.5*(mul_f_v.sum(axis=1))/c)
        
    for i in range(1,len(_n)):
        nodes = np.append(0, nearest[i])
        nodes = np.append(nodes, _n[i])

        # Encontrar la posición del nodo central 
        fila_i, columna_i = find_c_f(num_nodos, _n[i])

        # Encontrar vorticidades
        vor = vorticidad[fila_i, columna_i]

        # Encontrar velocidades de los nodos 
        mul_f_v = vel_values.iloc[np.zeros(5)+_n[i]]*factor

        # Encontrar la g de equilibrio 
        g_e_c = vor/5*(1 + 2.5*(mul_f_v.sum(axis=1))/c)
 
        
        g_e = np.append(g_e, g_e_c)
  

    g_eq = g_e.reshape(len(_n),len(nodes))

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

def act_fron_l():
    _n = n_l
    factor = e_I

    nearest = find_nearest(num_nodos, _n)
    nodes = np.insert(nearest[0], 1, 0)
    nodes = np.append(nodes, _n[0])

    # Encontrar la posición del nodo central 
    fila_i, columna_i = find_c_f(num_nodos, _n[0])

    # Encontrar vorticidades
    vor = vorticidad[fila_i, columna_i]

    # Encontrar velocidades de los nodos 
    mul_f_v = vel_values.iloc[np.zeros(5)+_n[0]]*factor

    # Encontrar la g de equilibrio 
    g_e = vor/5*(1 + 2.5*(mul_f_v.sum(axis=1))/c)

    for i in range(1,len(_n)):
        nodes = np.insert(nearest[i], 1, 0)
        nodes = np.append(nodes, _n[i])

        # Encontrar la posición del nodo central 
        fila_i, columna_i = find_c_f(num_nodos, _n[i])

        # Encontrar vorticidades
        vor = vorticidad[fila_i, columna_i]

        # Encontrar velocidades de los nodos 
        mul_f_v = vel_values.iloc[np.zeros(5)+_n[i]]*factor

        # Encontrar la g de equilibrio 
        g_e_c = vor/5*(1 + 2.5*(mul_f_v.sum(axis=1))/c)
        
        g_e = np.append(g_e, g_e_c)

    g_eq = g_e.reshape(len(_n),len(nodes))

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

def act_fron_r():
    _n = n_r
    factor = e_I

    nearest = find_nearest(num_nodos, _n)
    nodes = np.insert(nearest[0], 2, 0)
    nodes = np.append(nodes, _n[0])

    # Encontrar la posición del nodo central 
    fila_i, columna_i = find_c_f(num_nodos, _n[0])

    # Encontrar vorticidades
    vor = vorticidad[fila_i, columna_i]

    # Encontrar velocidades de los nodos 
    mul_f_v = vel_values.iloc[np.zeros(5)+_n[0]]*factor

    # Encontrar la g de equilibrio 
    g_e = vor/5*(1 + 2.5*(mul_f_v.sum(axis=1))/c)

    for i in range(1,len(_n)):
        nodes = np.insert(nearest[i], 2, 0)
        nodes = np.append(nodes, _n[i])

        # Encontrar la posición del nodo central 
        fila_i, columna_i = find_c_f(num_nodos, _n[i])

        # Encontrar vorticidades
        vor = vorticidad[fila_i, columna_i]

        # Encontrar velocidades de los nodos 
        mul_f_v = vel_values.iloc[np.zeros(5)+_n[i]]*factor

        # Encontrar la g de equilibrio 
        g_e_c = vor/5*(1 + 2.5*(mul_f_v.sum(axis=1))/c)
        
        g_e = np.append(g_e, g_e_c)

    g_eq = g_e.reshape(len(_n),len(nodes))

    fila_i, columna_i = find_c_f(num_nodos, _n)
    g_k_t[fila_i, columna_i] = g_eq[:,0]
    g_k_l[fila_i, columna_i] = g_eq[:,1]
    g_k_r[fila_i, columna_i] = g_eq[:,2]
    g_k_b[fila_i, columna_i] = g_eq[:,3]
    g_k_c[fila_i, columna_i] = g_eq[:,4]

    g_k_t_eq[fila_i, columna_i] = g_eq[:,0]
    g_k_l_eq[fila_i, columna_i] = g_eq[:,2]
    g_k_r_eq[fila_i, columna_i] = g_eq[:,1]
    g_k_b_eq[fila_i, columna_i] = g_eq[:,3]
    g_k_c_eq[fila_i, columna_i] = g_eq[:,4]

def act_fron_b():
    _n = n_b
    factor = e_I

    nearest = find_nearest(num_nodos, _n)
    nodes = np.insert(nearest[0], 3, 0)
    nodes = np.append(nodes, _n[0])

    # Encontrar la posición del nodo central 
    fila_i, columna_i = find_c_f(num_nodos, _n[0])

    # Encontrar vorticidades
    vor = vorticidad[fila_i, columna_i]

    # Encontrar velocidades de los nodos 
    mul_f_v = vel_values.iloc[np.zeros(5)+_n[0]]*factor

    # Encontrar la g de equilibrio 
    g_e = vor/5*(1 + 2.5*(mul_f_v.sum(axis=1))/c)

    for i in range(1,len(_n)):
        nodes = np.insert(nearest[i], 3, 0)
        nodes = np.append(nodes, _n[i])

        # Encontrar la posición del nodo central 
        fila_i, columna_i = find_c_f(num_nodos, _n[i])

        # Encontrar vorticidades
        vor = vorticidad[fila_i, columna_i]

        # Encontrar velocidades de los nodos 
        mul_f_v = vel_values.iloc[np.zeros(5)+_n[i]]*factor

        # Encontrar la g de equilibrio 
        g_e_c = vor/5*(1 + 2.5*(mul_f_v.sum(axis=1))/c)
        
        g_e = np.append(g_e, g_e_c)

    g_eq = g_e.reshape(len(_n),len(nodes))

    fila_i, columna_i = find_c_f(num_nodos, _n)
    g_k_t[fila_i, columna_i] = g_eq[:,0]
    g_k_l[fila_i, columna_i] = g_eq[:,2]
    g_k_r[fila_i, columna_i] = g_eq[:,1]
    g_k_b[fila_i, columna_i] = g_eq[:,3]
    g_k_c[fila_i, columna_i] = g_eq[:,4]

    g_k_t_eq[fila_i, columna_i] = g_eq[:,0]
    g_k_l_eq[fila_i, columna_i] = g_eq[:,2]
    g_k_r_eq[fila_i, columna_i] = g_eq[:,1]
    g_k_b_eq[fila_i, columna_i] = g_eq[:,3]
    g_k_c_eq[fila_i, columna_i] = g_eq[:,4]

# Paso 2: Calculo de g_k^(eq) en la frontera superior
# Encontrar nodos vecinos 
def actualizar_fronteras():
    act_fron_t()
    act_fron_l()
    act_fron_r()
    act_fron_b()

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


# Cálculo de g_k_e en los nodos interiores

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

    for i in range(1,len(_n)):
        nodes = np.append(nearest[i], _n[i])
        
        # Encontrar la posición del nodo central 
        fila_i, columna_i = find_c_f(num_nodos, _n[i])

        # Encontrar vorticidades
        vor = vorticidad[fila_i, columna_i]
        
        # Encontrar velocidades de los nodos 
        mul_f_v = vel_values.iloc[np.zeros(5)+_n[i]]*factor

        # Encontrar la g de equilibrio 
        g_e_c = vor/5*(1 + 2.5*(mul_f_v.sum(axis=1))/c)
        
        g_e = np.append(g_e, g_e_c)

    g_eq = g_e.reshape(len(_n),len(nodes))

    fila_i, columna_i = find_c_f(num_nodos, _n)
    g_k_t_eq[fila_i, columna_i] = g_eq[:,0]
    g_k_l_eq[fila_i, columna_i] = g_eq[:,2]
    g_k_r_eq[fila_i, columna_i] = g_eq[:,1]
    g_k_b_eq[fila_i, columna_i] = g_eq[:,3]
    g_k_c_eq[fila_i, columna_i] = g_eq[:,4]
    
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

    print(e-s)


