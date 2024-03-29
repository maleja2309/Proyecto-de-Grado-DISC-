﻿# Lattice Boltzmann Codes for vorticity (D2Q5) and velocity (D2Q9)
----
This repository contains the source code for the Lattice Boltzmann Method for the Cavity Flow Problem. The implementation includes two approximations: Velocity (D2Q9) and Vorticity (D2Q5). Each of them has its respective code for both sequential and parallel execution, as shown below, for the parallelization was employed the Joblib library. 

Is important to mention:
- The second approximation (Vorticity) has some issues which may be attributed to the border conditions, the time step or the spacing between the points inside the lattice. Therefore, the implementation is currently under review.
- The velocity code was developed in Mathematica by Professor Jose Rafael Toro Gomez Ms.C. from Universidad de los Andes - Colombia, this is a translation in Python of his code. Also, the vorticity code was adapted from his code, working with the points inside the lattice for paralellization instead of the matrices.  

# Overview of the repository structure
----
```md
.
| -- Velocidad
|   |-- LMB_Vel.ipynb          # Sequential implementation step by step in a Jupyter Notebook 
|   |-- Velocidad.py           # Parallel implementation of velocity D215
|   |-- Velocidad_serial.py    # Serial implementation velocity D2Q5
| -- Vorticidad
|   |-- LBM_Vor
|     |-- config.py            # Configuration file for the conditions of the case 
|     |-- corriente.py         # Functions for the stream functions
|     |-- corriente_iter.py    # 
|     |-- lbm.py               # Implementation of LBM
|     |-- main.py              # 
|     |-- secuencial.py        # Sequential implementation of LBM with vorticity
|     |-- utils.py             # Requited functions for the program to function
|     |-- velocidad.py         # Calculate velocities based on the stream function. 
|     |-- vor.py               # Update vorticity values by their position inside the geometry
|     |-- __init__.py
| -- LICENSE
| -- README.md
```
