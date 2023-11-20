import numpy as np
import matplotlib.pyplot as plt

# =====================================================================================
# Formato para realizar las gráficas 
# =====================================================================================

def formato_grafica(ax, fig, titulo= "", x_label= "", y_label= "", 
                    leyenda=False, xlim=[None, None], ylim=[None, None]):
    """_summary_

    Args:
        ax (_type_): Axis para realizar la gráfica
        fig (_type_): Figura sobre la que se va a realizar la gráfica
        titulo (str, optional): Título de la gráfica. Defaults to "".
        x_label (str, optional): Etiqueta del eje x . Defaults to "".
        y_label (str, optional): Etiqueta del eje y. Defaults to "".
        leyenda (bool, optional): Leyenda de la gráfica. Defaults to False.
        xlim (list, optional): Límite en x. Defaults to [None, None].
        ylim (list, optional): Límite en y. Defaults to [None, None].
    """
    
    ax.set_title(titulo)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    ax.tick_params(direction='out', length=5, width=0.75, grid_alpha=0.3)
    
    # Utilizarlo si se necesita alguna rotación en los labels de los ejes
#     ax.set_xticklabels(ax.get_xticks(), rotation = 0)
#     ax.set_yticklabels(ax.get_yticks(), rotation = 0)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    # Grid de la gráfica
    ax.grid(True)
    ax.grid(visible=True, which='major', color='grey', linestyle='-')
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', color='lightgrey', linestyle='-', alpha=0.2)

    if leyenda == True:
        ax.legend(loc='best', fontsize= 7)

    fig.tight_layout()


def graficar_f_corriente(vel_values, n_):
    #  Gráfica de velocidad
    u_ = np.array(vel_values['u']).reshape(n_,n_)
    v_ = np.array(vel_values['v']).reshape(n_,n_)

    fig, ax = plt.subplots(figsize=(5, 5))
    x = np.linspace(0,1, num=n_)
    y = np.linspace(0,1, num=n_)

    X, Y = np.meshgrid(x, y)

    u_f = np.flipud(u_)
    v_f = -np.flipud(v_)

    ax.streamplot(X,Y, u_f, v_f, color='k', density=[2, 1.5], linewidth=0.5, arrowsize=0.7)  
    formato_grafica(ax, fig, titulo=r'$\psi$')
    plt.show()

# =====================================================================================
# Ecuación para cálcular 𝜏
# =====================================================================================
def tau_reynolds(Re: float, c: float):
    """
    Calculo del tiempo de relajación del sistema

    Args:
        Re (float): Número de reynolds
        c (float): Velocidad de la partícula del fluido 

    Returns:
        float: 𝜏
    """
    return 5/(2*c**2*Re) + 0.5

# =====================================================================================
# Cálculo de la dirección de las velocidades 
# =====================================================================================
def velocity_directions(i: int):
    """
    Dirección de las velocidades asociadas 
    Configuración D2Q5
    
    Parameters:
        i (int): Índice de cada nodo.

    Returns:
        e (np.array): Velocidades (u,v) para cada nodo.
    """

    e = np.array([np.round(np.cos((i-1)*np.pi/2),1), np.round(np.sin((i-1)*np.pi/2),1)], dtype=np.float64)

    return e

# =====================================================================================
# Identificación de los nodos dada una configuración 
# =====================================================================================
def iden_nodes(num_nodos):
    """
    Identificación de los nodos dada una configuración

    Args:
        num_nodos (np.array): Arreglo con la configuración que se quiere estudiar

    Returns:
        np.array: Componentes de la configuración 
    """

    # Fronteras
    n_l = num_nodos[1:-1,0]    # Nodos lateral izquierda
    n_r = num_nodos[1:-1,-1]   # Nodos lateral derecha
    n_t = num_nodos[0,1:-1]    # Nodos superiores
    n_b = num_nodos[-1,1:-1]   # Nodos inferiores

    # Esquinas
    n_s_r = num_nodos[0,-1]    # Nodo esquina superior derecha
    n_s_l = num_nodos[0,0]     # Nodo esquina superior izquierda
    n_i_r = num_nodos[-1,-1]   # Nodo esquina inferior derecha
    n_i_l = num_nodos[-1,0]    # Nodo esquina inferior izquierda

    # Nodos interiores
    n_i = num_nodos[1:-1,1:-1]

    return n_l, n_r, n_t, n_b, n_s_r, n_s_l, n_i_r, n_i_l, n_i

# =====================================================================================
# Encontrar fila y columna de un nodo específico
# =====================================================================================
def find_c_f(num_nodos, data):
    """
    Encontrar fila y columna de un nodo específico

    Parameters:
        num_nodos (np.array): Multidimensional array 2D que contiene la configuración de los nodos. 
        data (data): ID del nodo de interés. 

    Returns:
        int, int: Fila y columna del nodo correspondiente. 
    """
    elements = np.isin(num_nodos,data)
    fila, columna = np.where(elements)
    return fila, columna

# =====================================================================================
# Encontrar las posiciones de los nodos en una dirección especifica
# =====================================================================================
def find_nearest(num_nodos, i):
    """
    Encontrar los vecinos de un nodo 

    Parameters:
        num_nodos (np.array): Matriz de referencia
        i (nodo): Número del nodo del que se quieren encontrar los vecinos

    Returns:
        np.array: Array con los nodos vecinos [T,L,R,B]
    """

    fila, columna = find_c_f(num_nodos, i)
    lista = [False, False, False, False]
    lista_n = []
    
    # Derecha
    # Moverse entre filas o columnas
    try:
        R = num_nodos[fila, columna + 1]
        
        if (columna.all() == columna.all() + 1):
            lista[0] = True
    except IndexError: 
        lista[0] = True

    # Izquierda
    try:
        L = num_nodos[fila, columna - 1]
        if (columna.all() - 1 == -1):
            lista[1] = True
    except IndexError: 
        lista[1] = True
    # I = np.delete(I, [0], 1).reshape(-1,)

    # Superior
    try:
        T = num_nodos[fila - 1, columna]
        if (fila.all() - 1 == -1) :
            lista[2] = True
    except IndexError: 
        lista[2] = True
    # A = np.delete(A, 0, 0).reshape(-1,)
    
    # Inferior 
    try:
        B = num_nodos[fila + 1, columna]
        if (fila.all() == fila.all() + 1):
            lista[3] = True
    except IndexError: 
        lista[3] = True
    # Inf_ = np.delete(Inf_, len(Inf_)-1, 0).reshape(-1,)

    if (lista[2] == False):
        lista_n.append(T)

    if (lista[1] == False):
        lista_n.append(L)

    if (lista[0] == False):
        lista_n.append(R)

    if (lista[3] == False):
        lista_n.append(B)

    return np.array(lista_n, dtype=np.float64).transpose()
