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
