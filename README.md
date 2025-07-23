# Active Vibration Control of a Compound-Split Drivetrain Using a Fuzzy Model Predictive Control Approach

This repository contains MATLAB and Simulink files for simulating active vibration damping in a **compound-split (CS) drivetrain** using a **Fuzzy Model Predictive Control (FMPC)** strategy. The nonlinear drivetrain is locally linearized at multiple operating points. For each point, a dedicated linear MPC is designed. During runtime, the controller interpolates between these local controllers based on the current system state.

## Overview

The project implements:

- A **nonlinear compound-split drivetrain model** in Simulink
- Local **linearizations** around selected operating conditions
- Design of **MPCs for each linear model**
- A **fuzzy interpolation scheme** for online control blending
- Closed-loop simulation and performance evaluation
- A **Lyapunov-based stability analysis** of the overall FMPC control system


## Project Structure

```text
.
├── src/                         # Entry point and simulation script
│   └── FMPC_Mainfile.m
│
├── models/                      # Simulink models (nonlinear and linear)
│   ├── fmpc_simulation_3modelle.slx
│   ├── fmpc_simulation_3modelle_ohne_z.slx
│   ├── fmpc_simulation_4modelle.slx
│   ├── mpc1_simulation.slx
│   ├── mpc2_simulation.slx
│   ├── mpc3_simulation.slx
│   └── mpc4_simulation.slx
│
├── linearization/              # Linearization of drivetrain dynamics
│   ├── init_parameter_nl.m
│   ├── Linearisierung.m
│   ├── model_lin.m
│   └── nugap_metrik_distanz.m
│
├── mpc_setup/                  # MPC matrix generation and model extension
│   ├── mpc_matrizen.m
│   └── mpc_modellerweiterung.m
│
├── further_system_analysis/    # Plots for frequency response, spring stiffness, membership functions
│   ├── plot_bode.m
│   ├── plot_federsteifigkeit.m
│   └── Zugehoerigkeitsfunktionen.m
│
├── lyapunov/                   # Lyapunov-based stability verification
│   ├── lmi_2D_Q_variabel.m
│   ├── lmi_2D_Q_konst_lokal.m
│   └── lmi_2D_Q_konst_global.m
│
└── README.md
