# Gray-Scott Model
## what is the Gray-Scott application?
The Gray-Scott system is a **reaction–diffusion system**, meaning it models a process involving both chemical reactions and diffusion across space. In the case of the Gray-Scott model that reaction is a chemical reaction between two substances 
 u and v, both of which diffuse over time. During the reaction gets used up, while  is produced. The densities of the substances 
 and are represented in the simulation.

## what this model generate:
The Gray-Scott system models the chemical reaction

U + 2V ->  3V

This reaction consumes U and produces V. Therefore, the amount of both substances needs to be controlled to maintain the reaction. This is done by adding U at the "feed rate" F and removing V at the "kill rate" k. The removal of V can also be described by another chemical reaction:

V -> P

For this reaction P is an inert product, meaning it doesn't react and therefore does not contribute to our observations. In this case the Parameter k controls the rate of the second reaction.


## what is the input of gray-scott

The Gray-Scott system models two chemical species:

U: the feed chemical, continuously added to the system.

V: the activator chemical, which is produced during the reaction and also removed.




##  Key Input Parameters

| Parameter      | Description                                    | Typical Range or Example Values         |
|----------------|------------------------------------------------|----------------------------------------|
| **F**          | Feed rate of U (controls how quickly U is replenished in the system) | 0.01 – 0.08         |
| **k**          | Kill rate of V (controls how quickly V is removed from the system) | 0.03 – 0.07         |
| **Du**         | Diffusion coefficient for U                   | Typically ~2 × Dv        |
| **Dv**         | Diffusion coefficient for V                   | Lower than Du (e.g., half) |
| **Grid size(L)**  | Spatial resolution of the simulation grid     | 256×256, 512×512       |
| **Time step(Steps)**  | Time integration step size                   | 0.01 – 1.0            |
| **Initial condition** | Initial distribution of U and V         | U = 1, V = 0 with small localized perturbations (e.g., center patch with V = 1) |
| **Simulation speed** | Controls visual update or iteration speed | 1×, 2×, etc.          |
| **Color scheme** | Display mode for concentration visualization | Black & white or RGB sliders |
| **Noise(noise)** | add noise for the simulation | 0.01~0.1 |
| **I/O frequency(plotgap)** | the frequecey of I/O between simulation steps  | 1~10 |















