# /// script
# requires-python = "==3.10.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "xdem",
#     "geoutils",
#     "rasterio",
#     "PyQt5",
# ]
# ///

# Usage: uv run --script cosine_hill.py --plot
import argparse
from numbers import Number

import matplotlib
matplotlib.use('Qt5Agg')

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator
from xdem import DEM

def write_flat_grid(
    filename: Path,
    dem: DEM,
    fmt: str = "%g",
    precision: int | None = 6,
) -> bool:
    o_fname = Path(filename)
    print(o_fname)
    dem.info()
    xs, ys = dem.coords(grid=False, force_offset="center")
    nx = xs.size
    ny = ys.size
    zs = dem.data[::-1, ::1].flatten(order="F")
    if precision is not None:
        zs = zs.round(precision)
    with o_fname.open("w") as f_out:
        f_out.write(f"{nx:d}\n")
        f_out.write(f"{ny:d}\n")
        np.savetxt(f_out, xs, fmt=fmt, delimiter=" ", newline="\n")
        np.savetxt(f_out, ys, fmt=fmt, delimiter=" ", newline="\n")
        np.savetxt(f_out, zs, fmt=fmt, delimiter=" ", newline="\n")
    return filename.is_file()


def main(plot=False, show=False):

    # Parameters from Table 5
    z0   = 0.000676  # Surface roughness length (m)
    uh   = 1.174     # Reference velocity (m/s)
    H    = 0.2       # Hill height (m)
    delta= 0.7       # Boundary layer thickness (m)
    T0   = 17        # Reference temperature (°C)
    u0   = 1.42      # Reference velocity (m/s)
    L    = float('inf')  # Monin-Obukhov length (m), ∞ for neutral
    ustar= 0.0845    # Friction velocity (m/s)
    kappa= 0.41      # von Karman constant
    kz   = 0.00507   # Roughness parameter (m)
    Rb = 0.42    # Hill base radius (m)

    scale = 1000


    upstream_distance = 2 # Distance upstream of the hill (m)
    downstream_distance = 6 # Distance downstream of the hill (m)
    resolution = 40 / 1000  # Grid resolution (m)
    width = 2.2
    height = 1.2

    resolution *= scale
    upstream_distance *= scale
    downstream_distance *= scale
    width *= scale
    height *= scale
    H *= scale
    Rb *= scale
    z0 *= scale
    ustar *= scale**(1/3.0)

    uref = ustar * (np.log(height / z0) / kappa)
    zref = height
    ustar = uref * (kappa / np.log(zref / z0))  # Friction velocity at reference height

    phi_e = 1.0  # Stability function for temperature
    phi_m = 1.0  # Stability function for momentum
    Cmu = 1/5.48/5.48

    dz = resolution / 10

    z_vals = np.arange(0.5*dz, height, dz)
    ux = ustar * (np.log(z_vals / z0) / kappa)
    uy = np.zeros_like(z_vals)
    uz = np.zeros_like(z_vals)
    epsilon = phi_e * (ustar**3) / (kappa * z_vals)  # Turbulent dissipation rate 
    tke = np.sqrt((kappa * ustar * z_vals * epsilon) / (Cmu * phi_m))   # Turbulent kinetic energy
    omega = epsilon / Cmu  / tke 

    with Path("./cosine_hill_profile.amrwind").open("w") as f:
        for i in range(len(z_vals)):
            f.write(f"{z_vals[i]} {ux[i]} {uy[i]} {uz[i]} {tke[i]} {omega[i]}\n")


    length = upstream_distance + downstream_distance

    nx = int(length / resolution) +1
    ny = int(width / resolution) +1
    nz = int(height / (8*resolution)) * 8 * 4
    print(f"{ int(length / (8*resolution)) * 8}")
    print(f"{ int(width / (8*resolution)) * 8}")
    print(f"{ int(height / (8*resolution)) * 8 * 4}")
    print(f"z0: {z0}")
    print(f"ux: {ux[-1]}")




    # Grid settings
    x = np.linspace(-upstream_distance, downstream_distance, nx, endpoint=True)
    y = np.linspace(0, width, ny, endpoint=True) -0.5*width

    print(f"Grid size: {nx} x {ny} (x: {x[0]} to {x[-1]}, y: {y[0]} to {y[-1]})")
    transform = rasterio.transform.from_bounds(
        x[0]-0.5*resolution, y[0]-0.5*resolution, x[-1]+0.5*resolution, y[-1]+0.5*resolution, nx, ny
    )
    raster = DEM.from_array(data=np.zeros((ny, nx), dtype=np.float32),
                 nodata=-1,
                 crs="EPSG:32633", transform=transform)
    X, Y = raster.coords(grid=True, force_offset="center")

    #X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Hill height function
    L = 2 * Rb 
    def h(r):
        return np.where(
            r < 0.5 * L,
            0.5 * H * (1 + np.cos(2 * np.pi * r / L)),
            0.0
        )


    Z = h(R)
    raster.data[:, :] = Z[::-1, ::1]  # Flip the data to match rasterio's convention

    if plot:        
        # Plot for visualization
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        raster.plot(ax=ax, cmap='gray_r', vmin=0, vmax=H)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Axisymmetric Cosine-Shaped Hill')
        ax.set_aspect('equal')
        plt.savefig("cosine_hill.png")
        if show:
            plt.show()

    write_flat_grid(Path("./cosine_hill.amrwind"), raster, fmt="%g", precision=6)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Show plot", default=True)
    parser.add_argument("--show", action="store_true", help="Show plot", default=False)
    args = parser.parse_args()
    main(plot=args.plot, show=args.show)