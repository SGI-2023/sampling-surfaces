import numpy as np

def generate_random_rectangle(n_points, seed_num_poly=None, seed_num_pos=None):
    rng_poly = np.random.default_rng(seed=seed_num_poly)
    rng_pos = np.random.default_rng(seed=seed_num_pos)

    posns = rng_pos.uniform(0,4,n_points)

    scale = rng_poly.uniform(high=1, size=(1,2))
    theta = rng_poly.uniform(high=np.pi)

    pts = np.zeros((n_points,2))
    normals = np.zeros((n_points,2))

    quadrant_1 = posns < 1
    quadrant_2 = (1 <= posns) & (posns < 2)
    quadrant_3 = (2 <= posns) & (posns < 3)
    quadrant_4 = 3 <= posns

    # Handle quadrant 1
    pts[quadrant_1, :] = np.column_stack([(posns[quadrant_1] - 0.5), -0.5 * np.ones_like(posns[quadrant_1])]) * scale
    normals[quadrant_1] = np.array([0,-1])

    # Handle quadrant 2
    pts[quadrant_2, :] = np.column_stack([0.5 * np.ones_like(posns[quadrant_2]), (posns[quadrant_2] - 1.5)]) * scale
    normals[quadrant_2] = np.array([1,0])

    # Handle quadrant 3
    pts[quadrant_3, :] = np.column_stack([(2.5 - posns[quadrant_3]), 0.5 * np.ones_like(posns[quadrant_3])]) * scale
    normals[quadrant_3] = np.array([0,1])

    # Handle quadrant 4
    pts[quadrant_4, :] = np.column_stack([-0.5 * np.ones_like(posns[quadrant_4]), (3.5 - posns[quadrant_4])]) * scale
    normals[quadrant_4] = np.array([-1,0])

    rotate = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    pts = np.dot(pts, rotate)
    normals = np.dot(normals, rotate)

    return pts, normals

def generate_pts_from_normals(pts, normals, n_new=1, n_points=None, dist_scale=0.1):
    num = pts.shape[0]
    rng = np.random.default_rng()

    if n_points is None:
        n_points = num

    # generate random indexes for existing points
    idx = np.random.choice(num, size=(n_points, n_new))

    # generate random distances for new points
    dists = rng.uniform(low=-1, high=1, size=(n_points, n_new, 1))

    # get selected points and normals
    selected_pts = pts[idx]
    selected_normals = normals[idx]

    # calculate new points
    new_pts = selected_pts + dists * dist_scale * selected_normals

    # reshape the points array into two columns
    new_pts = new_pts.reshape(-1, 2)

    # calculate new Y values
    new_Y = dists.ravel() * dist_scale

    return new_pts, new_Y
