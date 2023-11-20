import numpy as np

from config import *
from utils import *
from vor import *

# =====================================================================================
# Coeficientes para la solucion de la ecuación de Poisson 
# =====================================================================================
# ==== Coeficientes para la solucion de la ecuación de Poisson ==== #

# Nodo central
conf_in = [1, 1, -4, 1, 1]           #[T,L,C,R,B]

# Nodo esquina superior derecha
conf_2 = [2, -4, 2]

# Nodo esquina superior izquierda 
conf_3 = [-4, 2, 2]

# Nodo esquina inferior derecha
conf_4 = [2, 2, -4]

# Nodo esquina inferior izquierda
conf_5 =  [2, -4, 2]

# Nodo lateral izquierdo 
conf_6 = [1, -4, 2, 1]

# Nodo lateral derecho 
conf_7 = [1, 2, -4, 1]

# Nodo superior
conf_8 =  [1, -4, 1, 2]

# Nodo inferior
conf_9 =  [2, 1, -4, 1]

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
# Generar las ecuaciones para codo nodo
# =====================================================================================
def new_equations(o_matrix, id, num, n_, n_nodes):
    """_summary_

    Args:
        o_matrix (_type_): _description_
        id (_type_): _description_
        num (_type_): _description_
        n_ (_type_): _description_
        n_nodes (_type_): _description_

    Returns:
        _type_: _description_
    """

    lista_ = []
    # Matriz de ceros con las dimensiones de los nodos internos
    m = np.zeros((n_,n_))  
    
    # Se agrega el identificador del nodo central 
    id = np.append(id, n_nodes.reshape(len(id),1),axis=1)
    
    for i in range(len(id)):
        # De cada una de las filas se identifica en cuál posición está
        f, c = find_c_f(o_matrix, id[i])
        # Para cada nodo y sus vecinos se toma la configuración según las f,c y se reemplaza en m,
        # que identifica una de las ecuaciones del sistema. 
        m[f,c] = num[i]

        # Agregarla al sistema de ecuaciones
        lista_.append(m.reshape(-1,))

        # Inicializar una nueva fila que representa una ecuación del sistema
        m = np.zeros((n_,n_))
        
    return lista_

# =====================================================================================
# Cálculo vector independiente 
# =====================================================================================
def vorticidad_(w, vector, config):
    """
    Generación del vector de vorticidad para resolver Poisson

    Parameters:
        w (np.array): Arreglo con la vorticidad de cada uno de los nodos
        vector (np.aray): Arreglo de una dimensión en el que se agregan los resultdos 
        config (int): Especificación si se tiene en cuenta o no todos los nodos 

    Returns:
        _type_: _description_
    """

    if (config == 1):
        # Nodo esquina superior izquierda 
        # vector = np.append(vector, - w[0,0] + df_dx_l*2*delta_x - df_dy_t*2*delta_x)

        # Nodo superior
        vector = np.append(vector,- w[0,1:-1] - df_dy_t*2*delta_x)

        # Nodo esquina superior derecha
        # vector = np.append(vector, -w[0,-1] - df_dy_t*2*delta_x - df_dx_r*2*delta_x)

        # Nodo lateral izquierdo 
        vector = np.append(vector, - w[1:-1,0] + df_dx_l*2*delta_x)

        # Nodo central
        vector = np.append(vector,  -w[1:-1,1:-1])

        # Nodo lateral derecho 
        vector = np.append(vector, - w[1:-1,-1] - df_dx_l*2*delta_x)

        # Nodo esquina inferior izquierda
        # vector = np.append(vector,  - w[-1,0] + df_dx_l*2*delta_x + df_dy_b*2*delta_x)

        # Nodo inferior
        vector = np.append(vector,  - w[-1,1:-1] + df_dy_b*2*delta_x)

        # Nodo esquina inferior derecha
        # vector = np.append(vector, - w[-1,-1] - df_dx_r*2*delta_x - df_dy_b*2*delta_x)
    
    elif(config == 2):
        # Nodo esquina superior izquierda 
        vector = np.append(vector, - w[0,0] + df_dx_l*2*delta_x - df_dy_t*2*delta_x)

        # Nodo superior
        vector = np.append(vector,- w[0,1:-1] - df_dy_t*2*delta_x)

        # Nodo esquina superior derecha
        vector = np.append(vector, -w[0,-1] - df_dy_t*2*delta_x - df_dx_r*2*delta_x)

        # Nodo lateral izquierdo 
        vector = np.append(vector, - w[1:-1,0] + df_dx_l*2*delta_x)

        # Nodo central
        vector = np.append(vector,  -w[1:-1,1:-1])

        # Nodo lateral derecho 
        vector = np.append(vector, - w[1:-1,-1] - df_dx_l*2*delta_x)

        # Nodo esquina inferior izquierda
        vector = np.append(vector,  - w[-1,0] + df_dx_l*2*delta_x + df_dy_b*2*delta_x)

        # Nodo inferior
        vector = np.append(vector,  - w[-1,1:-1] + df_dy_b*2*delta_x)

        # Nodo esquina inferior derecha
        vector = np.append(vector, - w[-1,-1] - df_dx_r*2*delta_x - df_dy_b*2*delta_x)
    
    return vector

# Cantidad de elementos que hay en cada conjunto 
t_ = n_t.size
l_ = n_l.size
i_ = n_i.size
r_ = n_r.size
b_ = n_b.size
8
t_l = t_ + l_
t_l_i = t_l + i_
t_l_i_r = t_l_i + r_
t_l_i_r_b = t_l_i_r + b_

e_1 = 0
e_2 = t_ + 1
e_3 = e_2 + 1 + l_ + i_ + r_
e_4 = e_3 + 1 + b_
e_1, e_2, e_3, e_4

# =====================================================================================
# Encontrar ecuación según el nodo central
# =====================================================================================
def find_equations(num_nodos, config, conf_sol, n_):
    # Se encuentran los vecinos del nodo
    n = find_nearest(num_nodos, config)
    # Se generan n arrays de la misma dependiendo de los nodos que estén en la configuración
    values = np.tile(np.array(conf_sol, dtype=np.float64), config.size).reshape(config.size, len(conf_sol))
    return new_equations(num_nodos, n, values, n_, config)

# =====================================================================================
# Encontrar ecuaciones del sistema: Matríz de coeficientes 
# =====================================================================================
def generate_sys_eq(num_nodos, n_, config, conf_sol, sys_conf):
    if (sys_conf == 1):
        # ============================ Matríz ============================  #
        # Parte superior (T)
        lista_ecuaciones = find_equations(num_nodos, config[1], conf_sol[1], n_)

        # Parte central (L_L, C, L_D)
        lista_ecuaciones += find_equations(num_nodos, config[3], conf_sol[3], n_)
        lista_ecuaciones += find_equations(num_nodos, config[4], conf_sol[4], n_)
        lista_ecuaciones += find_equations(num_nodos, config[5], conf_sol[5], n_)   

        # Parte Inferior (B)
        lista_ecuaciones += find_equations(num_nodos, config[7], conf_sol[7], n_)

        lista_ecuaciones = np.array(lista_ecuaciones)
    
        lista_ecuaciones = np.delete(lista_ecuaciones, [e_1, e_2, e_3, e_4], axis=1)

    elif(sys_conf == 2):
  
        # ============================ Matríz ============================  #
        # Parte superior (E_L, T, E_R)
        lista_ecuaciones = find_equations(num_nodos, config[0], conf_sol[0], n_)
        lista_ecuaciones += find_equations(num_nodos, config[1], conf_sol[1], n_)
        lista_ecuaciones += find_equations(num_nodos, config[2], conf_sol[2], n_)

        # Parte central (L_L, C, L_D)
        lista_ecuaciones += find_equations(num_nodos, config[3], conf_sol[3], n_)
        lista_ecuaciones += find_equations(num_nodos, config[4], conf_sol[4], n_)
        lista_ecuaciones += find_equations(num_nodos, config[5], conf_sol[5], n_)   

        # Parte Inferior (E_I_L, B, E_I_D)
        lista_ecuaciones += find_equations(num_nodos, config[6], conf_sol[6], n_)
        lista_ecuaciones += find_equations(num_nodos, config[7], conf_sol[7], n_)
        lista_ecuaciones += find_equations(num_nodos, config[8], conf_sol[8], n_)

        lista_ecuaciones = np.array(lista_ecuaciones)
    
    return lista_ecuaciones

# =====================================================================================
# Solución de la función de corriente para cada nodo
# =====================================================================================
def solve_system(inv_A, vector_):
    x = inv_A @ vector_
    return x

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
