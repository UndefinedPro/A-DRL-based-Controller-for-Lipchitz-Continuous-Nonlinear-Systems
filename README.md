# Declaration (2025.5.27)

The related code has been uploaded. Other code of ploting will be added in the next few days. 

# Data Introduction 

All the data are organized by columns. 

**Spring_System_Data -- Data for Example 1**

- reward.xlsx -- The total value of of each trajectory during the training process. 
- step.xlsx     -- The steps of each trajectory. 
- theta1.xlsx  -- The simulation result of $\theta_1$.
- theta2.xlsx  -- The simulation result of $\theta_2$.
- u.xlsx         -- The control inputs of the system, the two columns represent the two inputs $\tau_1$ and $\tau_2$, respectively. 

**Manipulator_Data -- Data for Example 2**

- V (Simulation results with Lyapunov function being  $V(x)$ )
  - Lyp_reward.xlsx  -- The total value of of each trajectory during the training process. (The first column represents the trajectory counter)
  - Lyp_step.xlsx      -- The steps of each trajectory. (The first column represents the trajectory counter)
  - Lyp_x.xlsx          --  The simulation result of displacement $x$.

- expV (Simulation results with Lyapunov function being  $e^{2t}V(x)$ )
  - OneNet (Simulation results of a trained network)
    - Lyp_u.xlsx -- The simulation result of input $u$.
    - Lyp_v.xlsx -- The simulation result of velocity $v$.
    - Lyp_x.xlsx -- The simulation result of displacement $x$.
  - Lyp_reward.xlsx   -- The total value of of each trajectory during the training process. (The first column represents the trajectory counter)
  - Lyp_step.xlsx      -- The steps of each trajectory. (The first column represents the trajectory counter)
  - Lyp_x.xlsx          --  The simulation result of displacement $x$.

