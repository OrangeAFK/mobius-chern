# S-matrix / anticrossing scale (Experiment 15)

Filled **after** the numerical sweep. This measures the two-loop junction scale
`|S_12| ~ Delta` on the Mobius strip; it is **not** the Experiment 17 local tube
flagship, and it does not claim a first observation of the wall (Huang-Lee /
Beugeling prior art).

## Gap vs smearing (`Nx=80`, Mobius antiperiodic)

Min spectral gap near `E=0` grows soft as `w` increases: the *local* gap still
tracks `~2|lambda(x)|`, but the globally soft (wall-like) region widens. Only
`w >~ Nx` is fully wall-like.

| w | gap |
|---|-----|
| 0.5 | 0.030448 |
| 1 | 0.013930 |
| 2 | 0.007419 |
| 5 | 0.024522 |
| 10 | 0.031159 |
| 20 | 0.035288 |
| 40 | 0.039428 |
| 80 | 0.050616 |

## Cylinder valley purity (periodic tanh, two walls)

Purity = majority-valley weight at `k_y = +/- pi/2` on wall columns (1 = valley-pure).

| w | n_modes | mean purity |
|---|---------|-------------|
| 0.5 | 4 | 0.852 |
| 2 | 4 | 0.989 |
| 5 | 4 | 0.992 |
| 20 | 4 | 0.871 |

## Mobius hole-flux anticrossing `Delta(w, Ny)` (`Nx=40`)

`Delta = min_Phi [E_1(Phi) - E_0(Phi)]` for the two levels closest to zero.
In the two-channel network picture this tracks `|S_12|`.

| w | Ny | Delta | avoided? |
|---|----|-------|----------|
| 0.5 | 10 | 3.6209e-03 | yes |
| 2 | 10 | 4.0445e-02 | yes |
| 5 | 10 | 6.1010e-02 | yes |
| 20 | 10 | 8.0497e-02 | yes |
| 0.5 | 16 | 5.2297e-02 | yes |
| 2 | 16 | 1.2974e-02 | yes |
| 5 | 16 | 4.3345e-02 | yes |
| 20 | 16 | 6.5566e-02 | yes |

Integer `x0 -> x0+1` glide (Mobius antiperiodic): max `|dE|` = 2.665e-15
(pass if `<~ 1e-12`; no spectrum-vs-`x0` figure).

## Interpretation

Cylinder wall modes stay relatively valley-pure even as `w` grows, while Mobius hole-flux still shows a finite avoided-crossing scale `Delta`. That points to **twist-induced mixing** at the junction (finite `|S_12|`), not mere smearing of the mass profile.
