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
Wo = vorticidad.copy()

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

# Cálculo de g_k_e en los nodos interiores

def actualizar_g_k_e_c(fun_corriente):
    global f_k_t_eq, f_k_l_eq, f_k_r_eq, f_k_b_eq, f_k_c_eq
    _n = n_i.flatten()
    
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

    g_eq = g_e.reshape(len(_n),5)

    fila_i, columna_i = find_c_f(num_nodos, _n)

    f_k_t_eq[fila_i, columna_i] = g_eq[:,0]
    f_k_l_eq[fila_i, columna_i] = g_eq[:,1]
    f_k_r_eq[fila_i, columna_i] = g_eq[:,2]
    f_k_b_eq[fila_i, columna_i] = g_eq[:,3]
    f_k_c_eq[fila_i, columna_i] = g_eq[:,4]
    

def resuelvepoissonAS(fun_corriente, Fo, Wo, deltaW, f_k_t, f_k_l, f_k_r, f_k_b, f_k_c, f_k_t_, f_k_l_, f_k_r_, f_k_b_, f_k_c_):
    global vorticidad
    _n = n_i 

    nearest = find_nearest(num_nodos, _n)

    f_t, c_t = find_c_f(num_nodos, nearest[:,0])
    f_l, c_l = find_c_f(num_nodos, nearest[:,1])
    f_r, c_r = find_c_f(num_nodos, nearest[:,2])
    f_b, c_b = find_c_f(num_nodos, nearest[:,3])
    f_i, c_i = find_c_f(num_nodos, _n)

    while (deltaW > error):
        omega = (delta_t*-1/4*vorticidad*c**2/2*(0.5-taoF)*taoF)[1:-1,1:-1]

        # act_fron_t_c(n_t, e_I)
        # act_fron_t_c(n_l, e_I)
        # act_fron_t_c(n_r, e_I)
        # act_fron_t_c(n_b, e_I)

        actualizar_g_k_e_c(fun_corriente)

        f_k_t[f_i, c_i]  = f_k_t[f_t, c_t] - 1/taoF*(f_k_t[f_t, c_t] - f_k_t_eq[1:-1, 1:-1].flatten()) + omega.flatten()
        f_k_l[f_i, c_i]  = f_k_l[f_l, c_l] - 1/taoF*(f_k_l[f_l, c_l] - f_k_l_eq[1:-1, 1:-1].flatten()) + omega.flatten()
        f_k_r[f_i, c_i]  = f_k_r[f_r, c_r] - 1/taoF*(f_k_r[f_r, c_r] - f_k_r_eq[1:-1, 1:-1].flatten()) + omega.flatten()
        f_k_b[f_i, c_i]  = f_k_b[f_b, c_b] - 1/taoF*(f_k_b[f_b, c_b] - f_k_b_eq[1:-1, 1:-1].flatten()) + omega.flatten()
        f_k_c[f_i, c_i]  = f_k_c[f_i, c_i] - 1/taoF*(f_k_c[f_i, c_i] - f_k_c_eq[1:-1, 1:-1].flatten()) + 0
        
        f_k_t_[f_i, c_i] = f_k_t[f_t, c_t] 
        f_k_l_[f_i, c_i]  = f_k_l[f_l, c_l] 
        f_k_r_[f_i, c_i]  = f_k_r[f_r, c_r] 
        f_k_b_[f_i, c_i]  =  f_k_b[f_b, c_b] 
        f_k_c_[f_i, c_i]  = f_k_c[f_i, c_i] 

        fun_corriente = (f_k_l_ + f_k_r_ + f_k_b_ + f_k_t_)

        stream_data = get_streamfunction(fun_corriente, num_nodos)
        vorticidad = act_vorticidad(vorticidad, stream_data, delta_x, delta_y, [0,0,U,0])
                
        dF = fun_corriente - Fo
        dW1 = vorticidad[0,:] - Wo[0,:]
        deltaF = np.sqrt((dF.flatten()@dF.flatten())/(fun_corriente.flatten()@fun_corriente.flatten())) 
        deltaW = np.sqrt((dW1@dW1)/(vorticidad[0,:]@vorticidad[0,:]))
        
        Fo = fun_corriente.copy()
        Wo = vorticidad.copy()

    return Fo, Wo

if __name__ == '__main__':
 
    actualizar_fronteras()
    
    esquinas()
    actualizar_g_k_e()
    calculo_g_k()

    # Actualizar matriz de vorticidad 
    vorticidad[1:-1,1:-1] = (g_k_t + g_k_l + g_k_r + g_k_b + g_k_c)[1:-1,1:-1]

    num_iter = 100
    for i in range(num_iter):
        print(i)
        actualizar_vel()

        actualizar_g_k_e()
        actualizar_fronteras()
        calculo_g_k()

        # Actualizar matriz de vorticidad 
        vorticidad[1:-1,1:-1] = (g_k_t + g_k_l + g_k_r + g_k_b + g_k_c)[1:-1,1:-1]

        fun_corriente, vorticidad = resuelvepoissonAS(fun_corriente, Fo, Wo, deltaW, f_k_t, f_k_l, f_k_r, f_k_b, f_k_c, f_k_t_, f_k_l_, f_k_r_, f_k_b_, f_k_c_)
        #print(vorticidad)

    e = time.time()
    print(e-s)
    vel_values.to_csv('./velocidad_secuencial.csv', index=False, index_label=False)
    pd.DataFrame(vorticidad).to_csv('./vorticidad.csv', index=False, index_label=False)
    pd.DataFrame(fun_corriente).to_csv('./fun_corriente.csv', index=False, index_label=False)
    #graficar_f_corriente(vel_values, n_)

# 2:27