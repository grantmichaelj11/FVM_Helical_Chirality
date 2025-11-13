import MeshGenerator
from fipy import Gmsh3D, CellVariable, Viewer
from fipy import numerix
import numpy as np

#Define helical properties
c = 3e8                # speed of light
eps0 = 8.854e-12
mu0 = 1.0 / (eps0 * c**2)
E0 = 1.0               # electric field amplitude
radius = 1.5
pitch = 1.5
turns = 3
tube_r = 0.2
n_points = 45
n_circle_pts = 10
lc = 1.0
mesh_name = 'helix.msh'

#Define Electric Field properties
c = 3e8                # speed of light
eps0 = 8.854e-12
mu0 = 1.0 / (eps0 * c**2)
E0 = 1.0               # electric field amplitude
k = 2 * np.pi / pitch  # wavenumber along the helix axis
omega = c * k
t = 0.0

#Generate and load mesh
# MeshGenerator.createMesh(radius, pitch, turns, tube_r, n_points, n_circle_pts, lc, mesh_name, visualize=True)
mesh = Gmsh3D(mesh_name)


# -----------------------------
# 3. Compute local helical geometry
# -----------------------------
x, y, z = mesh.cellCenters

theta = np.arctan2(y, x)
s = (pitch / (2 * np.pi)) * theta    # approximate arc length
phi = k * s - omega * t              # phase

# Tangent direction
tx = -np.sin(theta)
ty = np.cos(theta)
tz = pitch / (2 * np.pi * radius)
norm_t = np.sqrt(tx**2 + ty**2 + tz**2)
tx, ty, tz = tx/norm_t, ty/norm_t, tz/norm_t

# Normal (radial inward)
nx = -np.cos(theta)
ny = -np.sin(theta)
nz = 0 * theta

# Binormal = tangent × normal
bx = ty * nz - tz * ny
by = tz * nx - tx * nz
bz = tx * ny - ty * nx
norm_b = np.sqrt(bx**2 + by**2 + bz**2)
bx, by, bz = bx/norm_b, by/norm_b, bz/norm_b

# -----------------------------
# 4. Define CPL initial fields
# -----------------------------
# Right-handed circular polarization
Ex = E0 * (numerix.cos(phi) * nx + numerix.sin(phi) * bx)
Ey = E0 * (numerix.cos(phi) * ny + numerix.sin(phi) * by)
Ez = E0 * (numerix.cos(phi) * nz + numerix.sin(phi) * bz)

Bx = (E0 / c) * (-numerix.sin(phi) * nx + numerix.cos(phi) * bx)
By = (E0 / c) * (-numerix.sin(phi) * ny + numerix.cos(phi) * by)
Bz = (E0 / c) * (-numerix.sin(phi) * nz + numerix.cos(phi) * bz)

E = CellVariable(mesh=mesh, rank=1, name="E", value=[Ex, Ey, Ez])
B = CellVariable(mesh=mesh, rank=1, name="B", value=[Bx, By, Bz])


# -----------------------------
# 5. Compute Curl Function
# -----------------------------
def curl(F):
    gradFx, gradFy, gradFz = F[0].grad, F[1].grad, F[2].grad
    cx = gradFz[1] - gradFy[2]
    cy = gradFx[2] - gradFz[0]
    cz = gradFy[0] - gradFx[1]
    return CellVariable(mesh=F.mesh, rank=1, value=[cx, cy, cz])

curlE = curl(E)
curlB = curl(B)

# -----------------------------
# 6. Magnitude for visualization
# -----------------------------
curlE_mag = CellVariable(mesh=mesh, name="|curlE|",
                         value=numerix.sqrt(curlE[0]**2 + curlE[1]**2 + curlE[2]**2))
curlB_mag = CellVariable(mesh=mesh, name="|curlB|",
                         value=numerix.sqrt(curlB[0]**2 + curlB[1]**2 + curlB[2]**2))

# -----------------------------
# 7. View fields or curls
# -----------------------------
# Uncomment whichever you want to see
# viewerE = Viewer(vars=(E,))
# viewerE.plot()

# viewerB = Viewer(vars=(B,))
# viewerB.plot()

viewerCurlE = Viewer(vars=(curlE_mag,))
viewerCurlE.plot()
