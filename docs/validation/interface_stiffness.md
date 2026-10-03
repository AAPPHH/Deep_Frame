# Interface stiffness (gap finder)

- load: F: force_n split equally over the selector nodes; M: minimum-norm nodal force field over the selector nodes with zero resultant force and resultant moment moment_nmm about the global axis through the node centroid (distributed linear traction couple)
- stiffness: work-conjugate: k = P^2 / sum_i f_i.u_i, i.e. load over the load-weighted mean displacement along F (= mean nodal displacement along F) or the load-weighted mean rotation about the moment axis; units N/mm and N*mm/rad
- support: fixture as in the frame evaluation (no inertia relief there): stack mount undersides (center_fixtures) fixed in all translations for motor pads, battery rails and camera; the four motor seat undersides (motor_fixtures) fixed for the stack
- point_masses: none (the battery rigid-body patch of the evaluation sits on the rail selector and would make the rail stiffness a rigid-patch artefact)

| block | Fx | Fy | Fz | Mx | My | Mz |
|---|---|---|---|---|---|---|
| tip | 1733 | 4.318 | 4.318 | 6002 | 9231 | 9231 |

F in N/mm, M in N*mm/rad.

- Fy/Fz: 1.0000
- My/Mz: 1.0000
- My/Fz over L^2/3: 1.0021
- Fx FE/beam: 0.9981
- Fy FE/beam: 0.9949
- Fz FE/beam: 0.9949
- My FE/beam: 0.9970
- Mz FE/beam: 0.9970
- Mx FE/beam: 0.9990

| ManaFly 3 BETA V4 | Fx | Fy | Fz | Mx | My | Mz |
|---|---|---|---|---|---|---|
| motor_front_left | 64.89 | 54.81 | 6.564 | 2917 | 4710 | 2.147e+04 |
| motor_front_right | 65.05 | 55.9 | 6.762 | 2928 | 4733 | 2.156e+04 |
| motor_rear_left | 31.8 | 14.72 | 6.295 | 5673 | 8700 | 1.387e+04 |
| motor_rear_right | 31.56 | 14.5 | 6.094 | 5612 | 8593 | 1.382e+04 |
| battery_rails | 49.73 | 88.2 | 406.9 | 2.192e+04 | 4.391e+04 | 1.409e+05 |
| camera | 27.62 | 60.16 | 42.01 | 2.529e+04 | 1.913e+04 | 5.587e+04 |
| stack | 433.7 | 683.5 | 94.32 | 3.434e+04 | 1.898e+04 | 4.051e+05 |

F in N/mm, M in N*mm/rad.

| BM Aether 4 (Ø9, body12) | Fx | Fy | Fz | Mx | My | Mz |
|---|---|---|---|---|---|---|
| motor_front_left | 88.82 | 62.9 | 12.43 | 2.004e+04 | 2.105e+04 | 1.267e+05 |
| motor_front_right | 88.73 | 62.99 | 12.52 | 2.021e+04 | 2.102e+04 | 1.234e+05 |
| motor_rear_left | 125.6 | 135 | 21.79 | 3.513e+04 | 2.915e+04 | 6.96e+04 |
| motor_rear_right | 125 | 135.4 | 21.55 | 3.5e+04 | 2.918e+04 | 6.987e+04 |
| battery_rails | 123.7 | 132.4 | 246.1 | 8273 | 7.455e+04 | 1.307e+05 |
| camera | 46.88 | 42.03 | 41.06 | 3.057e+04 | 5.811e+04 | 8.693e+04 |
| stack | 2606 | 1874 | 310.4 | 9.273e+04 | 1.131e+05 | 1.943e+06 |

F in N/mm, M in N*mm/rad.

| simp_mma_raw_1 | Fx | Fy | Fz | Mx | My | Mz |
|---|---|---|---|---|---|---|
| motor_front_left | 44.48 | 3.589 | 8.466 | 1503 | 2953 | 2307 |
| motor_front_right | 41.88 | 3.346 | 7.936 | 1448 | 2829 | 2215 |
| motor_rear_left | 32.2 | 10.33 | 7.431 | 2310 | 2920 | 5087 |
| motor_rear_right | 32.62 | 10.41 | 7.874 | 2403 | 2996 | 5249 |
| battery_rails | 50.14 | 89.83 | 229.7 | 2.149e+04 | 4.949e+04 | 4.048e+04 |
| camera | 88.83 | 113.4 | 92.69 | 2.931e+04 | 2.73e+04 | 4.232e+04 |
| stack | 308.7 | 208.4 | 113.3 | 2.137e+04 | 2.439e+04 | 1.041e+05 |

F in N/mm, M in N*mm/rad.

| simp_mma_recon_1 | Fx | Fy | Fz | Mx | My | Mz |
|---|---|---|---|---|---|---|
| motor_front_left | 46.07 | 3.03 | 8.094 | 1414 | 3198 | 1980 |
| motor_front_right | 43.71 | 2.854 | 7.826 | 1390 | 3306 | 1941 |
| motor_rear_left | 32.56 | 9.412 | 7.527 | 2305 | 2755 | 4454 |
| motor_rear_right | 34.45 | 9.846 | 7.324 | 2321 | 2770 | 4508 |
| battery_rails | 61.65 | 97.82 | 213.6 | 1.481e+04 | 4.366e+04 | 4.334e+04 |
| camera | 101.5 | 123.3 | 112.4 | 4.695e+04 | 3.144e+04 | 4.671e+04 |
| stack | 351.6 | 218.8 | 133.7 | 3.834e+04 | 3.385e+04 | 1.716e+05 |

F in N/mm, M in N*mm/rad.

Lücke: Kandidat < 70% beider Referenzen (roh). Grenzvorschlag = schwächere Referenz; skaliert = auf unsere Armlänge wie die Armspitze (gleiche Neigung F/(k*Arm), also k * Arm_ref/Arm_ours) für Kräfte, Momente unskaliert. Einheiten: F in N/mm, M in N*mm/rad.

### simp_mma_raw_1 (Arm 66.2 mm)

| Schnittstelle | Richtung | ManaFly 3 BETA V4 | BM Aether 4 (Ø9, body12) | simp_mma_raw_1 | /ManaFly 3 BETA V4 | /BM Aether 4 (Ø9, body12) | /ManaFly 3 BETA V4 skaliert | /BM Aether 4 (Ø9, body12) skaliert | Lücke |
|---|---|---|---|---|---|---|---|---|---|
| motor_front_left | Fx | 64.89 | 88.82 | 44.48 | 0.6854 | 0.5008 | 0.5673 | 0.348 | ja |
| motor_front_left | Fy | 54.81 | 62.9 | 3.589 | 0.06548 | 0.05706 | 0.05419 | 0.03965 | ja |
| motor_front_left | Fz | 6.564 | 12.43 | 8.466 | 1.29 | 0.6813 | 1.068 | 0.4734 |  |
| motor_front_left | Mx | 2917 | 2.004e+04 | 1503 | 0.5152 | 0.07498 | 0.5152 | 0.07498 | ja |
| motor_front_left | My | 4710 | 2.105e+04 | 2953 | 0.6269 | 0.1403 | 0.6269 | 0.1403 | ja |
| motor_front_left | Mz | 2.147e+04 | 1.267e+05 | 2307 | 0.1074 | 0.01821 | 0.1074 | 0.01821 | ja |
| motor_front_right | Fx | 65.05 | 88.73 | 41.88 | 0.6438 | 0.4719 | 0.5328 | 0.3279 | ja |
| motor_front_right | Fy | 55.9 | 62.99 | 3.346 | 0.05986 | 0.05313 | 0.04954 | 0.03692 | ja |
| motor_front_right | Fz | 6.762 | 12.52 | 7.936 | 1.174 | 0.6338 | 0.9713 | 0.4405 |  |
| motor_front_right | Mx | 2928 | 2.021e+04 | 1448 | 0.4947 | 0.07169 | 0.4947 | 0.07169 | ja |
| motor_front_right | My | 4733 | 2.102e+04 | 2829 | 0.5976 | 0.1346 | 0.5976 | 0.1346 | ja |
| motor_front_right | Mz | 2.156e+04 | 1.234e+05 | 2215 | 0.1028 | 0.01796 | 0.1028 | 0.01796 | ja |
| motor_rear_left | Fx | 31.8 | 125.6 | 32.2 | 1.012 | 0.2564 | 0.8378 | 0.1782 |  |
| motor_rear_left | Fy | 14.72 | 135 | 10.33 | 0.7016 | 0.07648 | 0.5807 | 0.05315 |  |
| motor_rear_left | Fz | 6.295 | 21.79 | 7.431 | 1.18 | 0.341 | 0.977 | 0.237 |  |
| motor_rear_left | Mx | 5673 | 3.513e+04 | 2310 | 0.4072 | 0.06576 | 0.4072 | 0.06576 | ja |
| motor_rear_left | My | 8700 | 2.915e+04 | 2920 | 0.3356 | 0.1002 | 0.3356 | 0.1002 | ja |
| motor_rear_left | Mz | 1.387e+04 | 6.96e+04 | 5087 | 0.3667 | 0.0731 | 0.3667 | 0.0731 | ja |
| motor_rear_right | Fx | 31.56 | 125 | 32.62 | 1.034 | 0.2611 | 0.8555 | 0.1814 |  |
| motor_rear_right | Fy | 14.5 | 135.4 | 10.41 | 0.7184 | 0.07691 | 0.5946 | 0.05345 |  |
| motor_rear_right | Fz | 6.094 | 21.55 | 7.874 | 1.292 | 0.3654 | 1.069 | 0.2539 |  |
| motor_rear_right | Mx | 5612 | 3.5e+04 | 2403 | 0.4282 | 0.06866 | 0.4282 | 0.06866 | ja |
| motor_rear_right | My | 8593 | 2.918e+04 | 2996 | 0.3486 | 0.1027 | 0.3486 | 0.1027 | ja |
| motor_rear_right | Mz | 1.382e+04 | 6.987e+04 | 5249 | 0.3799 | 0.07513 | 0.3799 | 0.07513 | ja |
| battery_rails | Fx | 49.73 | 123.7 | 50.14 | 1.008 | 0.4054 | 0.8346 | 0.2817 |  |
| battery_rails | Fy | 88.2 | 132.4 | 89.83 | 1.018 | 0.6787 | 0.843 | 0.4716 |  |
| battery_rails | Fz | 406.9 | 246.1 | 229.7 | 0.5645 | 0.9333 | 0.4672 | 0.6486 |  |
| battery_rails | Mx | 2.192e+04 | 8273 | 2.149e+04 | 0.9807 | 2.598 | 0.9807 | 2.598 |  |
| battery_rails | My | 4.391e+04 | 7.455e+04 | 4.949e+04 | 1.127 | 0.6639 | 1.127 | 0.6639 |  |
| battery_rails | Mz | 1.409e+05 | 1.307e+05 | 4.048e+04 | 0.2874 | 0.3098 | 0.2874 | 0.3098 | ja |
| camera | Fx | 27.62 | 46.88 | 88.83 | 3.216 | 1.895 | 2.662 | 1.317 |  |
| camera | Fy | 60.16 | 42.03 | 113.4 | 1.884 | 2.697 | 1.56 | 1.874 |  |
| camera | Fz | 42.01 | 41.06 | 92.69 | 2.206 | 2.258 | 1.826 | 1.569 |  |
| camera | Mx | 2.529e+04 | 3.057e+04 | 2.931e+04 | 1.159 | 0.9587 | 1.159 | 0.9587 |  |
| camera | My | 1.913e+04 | 5.811e+04 | 2.73e+04 | 1.427 | 0.4698 | 1.427 | 0.4698 |  |
| camera | Mz | 5.587e+04 | 8.693e+04 | 4.232e+04 | 0.7574 | 0.4868 | 0.7574 | 0.4868 |  |
| stack | Fx | 433.7 | 2606 | 308.7 | 0.7117 | 0.1184 | 0.5891 | 0.0823 |  |
| stack | Fy | 683.5 | 1874 | 208.4 | 0.3048 | 0.1112 | 0.2523 | 0.07725 | ja |
| stack | Fz | 94.32 | 310.4 | 113.3 | 1.201 | 0.365 | 0.9942 | 0.2536 |  |
| stack | Mx | 3.434e+04 | 9.273e+04 | 2.137e+04 | 0.6223 | 0.2305 | 0.6223 | 0.2305 | ja |
| stack | My | 1.898e+04 | 1.131e+05 | 2.439e+04 | 1.285 | 0.2157 | 1.285 | 0.2157 |  |
| stack | Mz | 4.051e+05 | 1.943e+06 | 1.041e+05 | 0.2571 | 0.05359 | 0.2571 | 0.05359 | ja |

### simp_mma_recon_1 (Arm 66.2 mm)

| Schnittstelle | Richtung | ManaFly 3 BETA V4 | BM Aether 4 (Ø9, body12) | simp_mma_recon_1 | /ManaFly 3 BETA V4 | /BM Aether 4 (Ø9, body12) | /ManaFly 3 BETA V4 skaliert | /BM Aether 4 (Ø9, body12) skaliert | Lücke |
|---|---|---|---|---|---|---|---|---|---|
| motor_front_left | Fx | 64.89 | 88.82 | 46.07 | 0.71 | 0.5187 | 0.5876 | 0.3604 |  |
| motor_front_left | Fy | 54.81 | 62.9 | 3.03 | 0.05527 | 0.04817 | 0.04575 | 0.03347 | ja |
| motor_front_left | Fz | 6.564 | 12.43 | 8.094 | 1.233 | 0.6513 | 1.021 | 0.4526 |  |
| motor_front_left | Mx | 2917 | 2.004e+04 | 1414 | 0.4849 | 0.07057 | 0.4849 | 0.07057 | ja |
| motor_front_left | My | 4710 | 2.105e+04 | 3198 | 0.6789 | 0.1519 | 0.6789 | 0.1519 | ja |
| motor_front_left | Mz | 2.147e+04 | 1.267e+05 | 1980 | 0.09219 | 0.01563 | 0.09219 | 0.01563 | ja |
| motor_front_right | Fx | 65.05 | 88.73 | 43.71 | 0.672 | 0.4927 | 0.5562 | 0.3423 | ja |
| motor_front_right | Fy | 55.9 | 62.99 | 2.854 | 0.05105 | 0.04531 | 0.04225 | 0.03149 | ja |
| motor_front_right | Fz | 6.762 | 12.52 | 7.826 | 1.157 | 0.6251 | 0.9579 | 0.4344 |  |
| motor_front_right | Mx | 2928 | 2.021e+04 | 1390 | 0.4746 | 0.06878 | 0.4746 | 0.06878 | ja |
| motor_front_right | My | 4733 | 2.102e+04 | 3306 | 0.6985 | 0.1573 | 0.6985 | 0.1573 | ja |
| motor_front_right | Mz | 2.156e+04 | 1.234e+05 | 1941 | 0.09006 | 0.01574 | 0.09006 | 0.01574 | ja |
| motor_rear_left | Fx | 31.8 | 125.6 | 32.56 | 1.024 | 0.2593 | 0.8473 | 0.1802 |  |
| motor_rear_left | Fy | 14.72 | 135 | 9.412 | 0.6395 | 0.06972 | 0.5293 | 0.04845 | ja |
| motor_rear_left | Fz | 6.295 | 21.79 | 7.527 | 1.196 | 0.3454 | 0.9896 | 0.24 |  |
| motor_rear_left | Mx | 5673 | 3.513e+04 | 2305 | 0.4064 | 0.06563 | 0.4064 | 0.06563 | ja |
| motor_rear_left | My | 8700 | 2.915e+04 | 2755 | 0.3166 | 0.0945 | 0.3166 | 0.0945 | ja |
| motor_rear_left | Mz | 1.387e+04 | 6.96e+04 | 4454 | 0.3211 | 0.064 | 0.3211 | 0.064 | ja |
| motor_rear_right | Fx | 31.56 | 125 | 34.45 | 1.091 | 0.2757 | 0.9034 | 0.1916 |  |
| motor_rear_right | Fy | 14.5 | 135.4 | 9.846 | 0.6792 | 0.07272 | 0.5622 | 0.05053 | ja |
| motor_rear_right | Fz | 6.094 | 21.55 | 7.324 | 1.202 | 0.3399 | 0.9948 | 0.2362 |  |
| motor_rear_right | Mx | 5612 | 3.5e+04 | 2321 | 0.4135 | 0.0663 | 0.4135 | 0.0663 | ja |
| motor_rear_right | My | 8593 | 2.918e+04 | 2770 | 0.3224 | 0.09494 | 0.3224 | 0.09494 | ja |
| motor_rear_right | Mz | 1.382e+04 | 6.987e+04 | 4508 | 0.3262 | 0.06452 | 0.3262 | 0.06452 | ja |
| battery_rails | Fx | 49.73 | 123.7 | 61.65 | 1.24 | 0.4984 | 1.026 | 0.3464 |  |
| battery_rails | Fy | 88.2 | 132.4 | 97.82 | 1.109 | 0.739 | 0.9179 | 0.5135 |  |
| battery_rails | Fz | 406.9 | 246.1 | 213.6 | 0.5248 | 0.8678 | 0.4344 | 0.603 |  |
| battery_rails | Mx | 2.192e+04 | 8273 | 1.481e+04 | 0.6758 | 1.79 | 0.6758 | 1.79 |  |
| battery_rails | My | 4.391e+04 | 7.455e+04 | 4.366e+04 | 0.9942 | 0.5857 | 0.9942 | 0.5857 |  |
| battery_rails | Mz | 1.409e+05 | 1.307e+05 | 4.334e+04 | 0.3077 | 0.3317 | 0.3077 | 0.3317 | ja |
| camera | Fx | 27.62 | 46.88 | 101.5 | 3.675 | 2.165 | 3.042 | 1.505 |  |
| camera | Fy | 60.16 | 42.03 | 123.3 | 2.049 | 2.933 | 1.696 | 2.038 |  |
| camera | Fz | 42.01 | 41.06 | 112.4 | 2.676 | 2.738 | 2.215 | 1.903 |  |
| camera | Mx | 2.529e+04 | 3.057e+04 | 4.695e+04 | 1.857 | 1.536 | 1.857 | 1.536 |  |
| camera | My | 1.913e+04 | 5.811e+04 | 3.144e+04 | 1.644 | 0.5411 | 1.644 | 0.5411 |  |
| camera | Mz | 5.587e+04 | 8.693e+04 | 4.671e+04 | 0.836 | 0.5373 | 0.836 | 0.5373 |  |
| stack | Fx | 433.7 | 2606 | 351.6 | 0.8106 | 0.1349 | 0.6709 | 0.09374 |  |
| stack | Fy | 683.5 | 1874 | 218.8 | 0.3201 | 0.1167 | 0.265 | 0.08112 | ja |
| stack | Fz | 94.32 | 310.4 | 133.7 | 1.417 | 0.4306 | 1.173 | 0.2992 |  |
| stack | Mx | 3.434e+04 | 9.273e+04 | 3.834e+04 | 1.117 | 0.4135 | 1.117 | 0.4135 |  |
| stack | My | 1.898e+04 | 1.131e+05 | 3.385e+04 | 1.783 | 0.2993 | 1.783 | 0.2993 |  |
| stack | Mz | 4.051e+05 | 1.943e+06 | 1.716e+05 | 0.4236 | 0.08831 | 0.4236 | 0.08831 | ja |

### Vorgeschlagene Bedingungen

| Kandidat | Schnittstelle | Richtung | k Kandidat | Grenze roh | Grenze skaliert | Quelle | auch skaliert < Schwelle |
|---|---|---|---|---|---|---|---|
| simp_mma_raw_1 | motor_front_left | Fx | 44.48 | 64.89 | 78.4 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | motor_front_left | Fy | 3.589 | 54.81 | 66.23 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | motor_front_left | Mx | 1503 | 2917 | 2917 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | motor_front_left | My | 2953 | 4710 | 4710 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | motor_front_left | Mz | 2307 | 2.147e+04 | 2.147e+04 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | motor_front_right | Fx | 41.88 | 65.05 | 78.59 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | motor_front_right | Fy | 3.346 | 55.9 | 67.54 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | motor_front_right | Mx | 1448 | 2928 | 2928 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | motor_front_right | My | 2829 | 4733 | 4733 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | motor_front_right | Mz | 2215 | 2.156e+04 | 2.156e+04 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | motor_rear_left | Mx | 2310 | 5673 | 5673 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | motor_rear_left | My | 2920 | 8700 | 8700 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | motor_rear_left | Mz | 5087 | 1.387e+04 | 1.387e+04 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | motor_rear_right | Mx | 2403 | 5612 | 5612 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | motor_rear_right | My | 2996 | 8593 | 8593 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | motor_rear_right | Mz | 5249 | 1.382e+04 | 1.382e+04 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | battery_rails | Mz | 4.048e+04 | 1.307e+05 | 1.307e+05 | BM Aether 4 (Ø9, body12) | ja |
| simp_mma_raw_1 | stack | Fy | 208.4 | 683.5 | 825.8 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | stack | Mx | 2.137e+04 | 3.434e+04 | 3.434e+04 | ManaFly 3 BETA V4 | ja |
| simp_mma_raw_1 | stack | Mz | 1.041e+05 | 4.051e+05 | 4.051e+05 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_front_left | Fy | 3.03 | 54.81 | 66.23 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_front_left | Mx | 1414 | 2917 | 2917 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_front_left | My | 3198 | 4710 | 4710 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_front_left | Mz | 1980 | 2.147e+04 | 2.147e+04 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_front_right | Fx | 43.71 | 65.05 | 78.59 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_front_right | Fy | 2.854 | 55.9 | 67.54 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_front_right | Mx | 1390 | 2928 | 2928 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_front_right | My | 3306 | 4733 | 4733 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_front_right | Mz | 1941 | 2.156e+04 | 2.156e+04 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_rear_left | Fy | 9.412 | 14.72 | 17.78 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_rear_left | Mx | 2305 | 5673 | 5673 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_rear_left | My | 2755 | 8700 | 8700 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_rear_left | Mz | 4454 | 1.387e+04 | 1.387e+04 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_rear_right | Fy | 9.846 | 14.5 | 17.51 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_rear_right | Mx | 2321 | 5612 | 5612 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_rear_right | My | 2770 | 8593 | 8593 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | motor_rear_right | Mz | 4508 | 1.382e+04 | 1.382e+04 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | battery_rails | Mz | 4.334e+04 | 1.307e+05 | 1.307e+05 | BM Aether 4 (Ø9, body12) | ja |
| simp_mma_recon_1 | stack | Fy | 218.8 | 683.5 | 825.8 | ManaFly 3 BETA V4 | ja |
| simp_mma_recon_1 | stack | Mz | 1.716e+05 | 4.051e+05 | 4.051e+05 | ManaFly 3 BETA V4 | ja |
