import gmsh
import numpy as np

gmsh.initialize()
gmsh.model.add("helical_pipe_occ")

# -----------------------
# Parameters
# -----------------------
radius = 1.5
pitch = 1.5
turns = 3
tube_r = 0.2
n_points = 50         # points along helix for tangent calculation
n_circle_pts = 12    # points per circle
lc = 0.1             # mesh size

points_coords = [(radius, 0, 0)]
tangents = []
normals = []
binormals = []
circle_coordinates = []

# -----------------------
# Generate helix points and tangents
# -----------------------
for i in range(1, n_points + 4):
    t = 2 * np.pi * turns * i / n_points
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = pitch/(2*np.pi) * t
    points_coords.append((x, y, z))

    dx = x - points_coords[i-1][0]
    dy = y - points_coords[i-1][1]
    dz = z - points_coords[i-1][2]

    tangent = np.array([dx, dy, dz])
    tangent /= np.linalg.norm(tangent)
    tangents.append(tangent)

for i in range(len(tangents)-1):
    normal = tangents[i+1] - tangents[i]
    normal /= np.linalg.norm(tangents[i+1] - tangents[i])
    normals.append(normal)

for i in range(len(normals)-1):
    binormal = np.cross(tangents[i], normals[i])
    binormals.append(binormal)

# -----------------------
# Generate circles along the helix using tangent/binormal
# -----------------------
for i in range(len(points_coords)-4):
    circ_radians = np.linspace(0, 2*np.pi, n_circle_pts, endpoint=False)
    circle_points = []
    for theta in circ_radians:
        x = points_coords[i][0] + tube_r*np.cos(theta)*normals[i][0] + tube_r*np.sin(theta)*binormals[i][0]
        y = points_coords[i][1] + tube_r*np.cos(theta)*normals[i][1] + tube_r*np.sin(theta)*binormals[i][1]
        z = points_coords[i][2] + tube_r*np.cos(theta)*normals[i][2] + tube_r*np.sin(theta)*binormals[i][2]
        circle_points.append([x, y, z])
    circle_coordinates.append(circle_points)

# -----------------------
# Add points in OCC
# -----------------------
point_tags = []
for circle in circle_coordinates:
    circle_point_tags = []
    for pt in circle:
        tag = gmsh.model.occ.addPoint(pt[0], pt[1], pt[2], lc)
        circle_point_tags.append(tag)
    point_tags.append(circle_point_tags)

n_pts = len(point_tags[0])
side_surfaces = []
front_lines = []
back_lines = []

# -----------------------
# Create side quads as OCC surfaces
# -----------------------
for i in range(len(point_tags)-1):
    c1 = point_tags[i]
    c2 = point_tags[i+1]
    for j in range(n_pts):
        p1 = c1[j]
        p2 = c1[(j+1)%n_pts]
        p3 = c2[(j+1)%n_pts]
        p4 = c2[j]

        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)

        if i == 0:
            front_lines.append(l1)
        if i == len(point_tags)-2:
            back_lines.append(l3)

        cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        surf = gmsh.model.occ.addSurfaceFilling(cl)
        side_surfaces.append(surf)

# -----------------------
# Create caps using OCC
# -----------------------
front_loop = gmsh.model.occ.addCurveLoop(front_lines)
cap_front = gmsh.model.occ.addPlaneSurface([front_loop])

back_loop = gmsh.model.occ.addCurveLoop(back_lines)
cap_back = gmsh.model.occ.addPlaneSurface([back_loop])

# -----------------------
# Fuse all surfaces into a single closed volume
# -----------------------
objects = [(2, s) for s in side_surfaces + [cap_front, cap_back]]
volumes, _ = gmsh.model.occ.fragment(objects, toolDimTags=[])

gmsh.model.occ.synchronize()

# -----------------------
# Mesh generation
# -----------------------
gmsh.model.mesh.generate(3)
gmsh.fltk.run()
gmsh.finalize()
