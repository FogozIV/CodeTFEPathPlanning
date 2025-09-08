import numpy as np
from curve_library import CurveList, Position, ClothoidCurve, Angle
from matplotlib import pyplot as plt

from src.utils.path_following import curve_list_to_path, fit_clothoids, fit_clothoids_from_waypoints, \
    plot_path_on_field, plot_path


def dilate_mask(mask, robot_radius_m, resolution_m):
    # simple square dilation (replace with a disk structure if you want)
    r = int(np.ceil(robot_radius_m / resolution_m))
    if r <= 0:
        return mask.copy()
    from scipy.ndimage import maximum_filter
    return maximum_filter(mask.astype(np.uint8), size=2*r+1) > 0

def douglas_peucker(points, eps):
    """Curvature-aware tweak: weight perpendicular distance by local turn angle."""
    if len(points) <= 2:
        return points
    # find point with max weighted distance from the line p0->pN
    p0, pN = np.array(points[0]), np.array(points[-1])
    v = pN - p0
    lv = np.linalg.norm(v) + 1e-12
    v /= lv
    max_d, idx = -1.0, -1
    for i in range(1, len(points)-1):
        pi = np.array(points[i])
        # perpendicular distance
        d = np.linalg.norm(np.cross(np.r_[v,0], np.r_[pi-p0,0])) / 1.0
        # local turn weight
        a = np.array(points[i-1]); b = np.array(points[i]); c = np.array(points[i+1])
        t1 = np.arctan2(b[1]-a[1], b[0]-a[0])
        t2 = np.arctan2(c[1]-b[1], c[0]-b[0])
        turn = abs(np.unwrap([t1, t2])[1] - np.unwrap([t1, t2])[0])
        w = 1.0 + 0.5*turn  # upweight corners
        wd = w * d
        if wd > max_d:
            max_d, idx = wd, i
    if max_d > eps:
        left = douglas_peucker(points[:idx+1], eps)
        right = douglas_peucker(points[idx:], eps)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]

def polyline_tangents(points):
    tangents = []
    for i in range(len(points)):
        if i==0:
            v = np.array(points[1]) - np.array(points[0])
        elif i==len(points)-1:
            v = np.array(points[-1]) - np.array(points[-2])
        else:
            v = 0.5*(np.array(points[i+1])-np.array(points[i-1]))
        tangents.append(np.arctan2(v[1], v[0]))
    return np.unwrap(np.array(tangents)).tolist()

def discrete_curvature(points):
    """Estimate curvature; returns kappa_i for each point."""
    kappa = []
    pts = [np.array(p) for p in points]
    for i in range(len(points)):
        if i==0 or i==len(points)-1:
            kappa.append(0.0)
            continue
        a,b,c = pts[i-1], pts[i], pts[i+1]
        ab, bc, ca = np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(a-c)
        s = (ab+bc+ca)/2.0
        area = max(s*(s-ab)*(s-bc)*(s-ca), 0.0)
        if area==0 or ab*bc*ca==0:
            kappa.append(0.0)
        else:
            R = (ab*bc*ca)/(4*np.sqrt(area))
            kappa.append(1.0/max(R,1e-6))
    return kappa

def sample_clothoid_segment(seg, ds):
    """Assumed API: seg has length 'L' and a function eval(s)->(x,y,theta,kappa)."""
    s = seg.getMinValue()
    out = []
    while s < seg.getMaxValue():
        out.append(seg.get_position(s, 0.01))  # (x,y,theta,kappa)
        s += ds
    out.append(seg.getLastPosition())
    out = [(a.x, a.y, a.angle, a.curvature) for a in out]
    return out

def path_collides(curve_list, inflated_mask, resolution, ds=0.01):
    sampled_clothoid = sample_clothoid_segment(curve_list, ds)
    for (x,y,_,_) in sampled_clothoid:
        i = int(y / resolution); j = int(x / resolution)
        if i < 0 or j < 0 or i >= inflated_mask.shape[0] or j >= inflated_mask.shape[1]:
            return True
        if inflated_mask[i,j]:
            return True
    return False
def find_closest_collision_point(seg: CurveList, mask, resolution, ds=0.01):
    sampled_clothoid = sample_clothoid_segment(seg, ds)
    for (x,y,_,_) in sampled_clothoid:
        i = int(y / resolution); j = int(x / resolution)
        if i < 0 or j < 0 or i >= mask.shape[0] or j >= mask.shape[1]:
            return (x,y)
        if mask[i,j]:
            return (x,y)
    return None
def harmonic_guided_clothoids(field,
                              start_xy_theta_kappa,
                              goal_xy_theta_kappa,
                              robot_radius_m=0.0*1000,
                              simplify_eps=0.03*1000,
                              sample_ds=0.01*1000,
                              max_outer=120):
    """
    1) trace streamline from start
    2) simplify to sparse anchors
    3) attach boundary states
    4) fit G² clothoids (your solver)
    5) validate & relax if needed
    """
    # 1) trace streamline
    curve_hint = field.get_path(start_xy_theta_kappa[:2], step_size=0.05*1000, max_steps=20000)
    plot_path(np.array(curve_hint))
    if len(curve_hint) < 2:
        return None, None

    # 2) safety inflation
    inflated = dilate_mask(field.mask, robot_radius_m, field.resolution)

    # 3) simplify
    sparse = douglas_peucker(curve_hint, simplify_eps)

    # Make sure goal is in
    if np.linalg.norm(np.array(sparse[-1]) - np.array(goal_xy_theta_kappa[:2])) > 0.5*simplify_eps:
        sparse.append(goal_xy_theta_kappa[:2])

    # 4) headings & kappas on anchors
    thetas = polyline_tangents(sparse)
    kappas = discrete_curvature(sparse)

    # enforce start/goal states
    sparse[0] = start_xy_theta_kappa[:2]
    thetas[0] = start_xy_theta_kappa[2]
    kappas[0] = start_xy_theta_kappa[3]

    sparse[-1] = goal_xy_theta_kappa[:2]
    thetas[-1] = goal_xy_theta_kappa[2]
    kappas[-1] = goal_xy_theta_kappa[3]

    # 5) iterative fit/validate/relax
    for outer in range(max_outer):
        # Use your clothoid fitter between consecutive anchors, with endpoint (x,y,theta,kappa).
        # Sketch: build param_path = [(x,y,theta,kappa), ...] in meters and radians
        param_path = [(x, y, th, ka) for (x,y), th, ka in zip(sparse, thetas, kappas)]
        print(param_path)
        # If your fit function expects just xy + (theta,kappa) packed in param_path:
        curve_list = fit_clothoids_from_waypoints(param_path)
        print(curve_list.get_curve_count())
        if curve_list is None or curve_list.get_curve_count()==0:
            # fallback: insert a midpoint and retry
            mid_idx = np.argmax([np.linalg.norm(np.array(sparse[i+1])-np.array(sparse[i])) for i in range(len(sparse)-1)])
            mid = 0.5*(np.array(sparse[mid_idx])+np.array(sparse[mid_idx+1]))
            sparse.insert(mid_idx+1, mid.tolist())
            thetas = polyline_tangents(sparse)
            kappas = discrete_curvature(sparse)
            thetas[0] = start_xy_theta_kappa[2]; kappas[0]=start_xy_theta_kappa[3]
            thetas[-1]= goal_xy_theta_kappa[2]; kappas[-1]=goal_xy_theta_kappa[3]
            continue

        # 6) collision check
        if not path_collides(curve_list, inflated, field.resolution, ds=sample_ds):
            # success
            clothoid_path = curve_list_to_path(curve_list)
            return curve_list, clothoid_path
        print("Path colliding ?")
        # 7) relax: push anchors away from obstacles using potential/clearance gradient
        # Simple heuristic: move interior points a bit along +grad(phi) to increase potential (clearance)
        alpha_xy = 0.5 * field.resolution  # step
        for i in range(1, len(sparse)-1):
            x, y = sparse[i]
            i0, j0 = int(y/field.resolution), int(x/field.resolution)
            if i0<=0 or j0<=0 or i0>=field.ny-1 or j0>=field.nx-1:
                continue
            dphidx = (field.grid[i0, j0+1] - field.grid[i0, j0-1]) / (2*field.resolution)
            dphidy = (field.grid[i0+1, j0] - field.grid[i0-1, j0]) / (2*field.resolution)
            g = np.array([dphidx, dphidy])
            if np.linalg.norm(g) > 1e-12:
                g /= np.linalg.norm(g)
                sparse[i] = (x + alpha_xy*g[0], y + alpha_xy*g[1])

        # smooth headings slightly
        thetas = polyline_tangents(sparse)
        thetas[0] = start_xy_theta_kappa[2]; thetas[-1] = goal_xy_theta_kappa[2]
        # keep kappas moderate (bias to 0 for interiors)
        for i in range(1, len(kappas)-1):
            kappas[i] *= 0.5

    # If we reach here, we failed to find a valid clothoid path
    return None, None


def get_theta_kappa_from_spline(spline_points, query_xy):
    """
    spline_points: list of (x,y) along smoothed reference path
    query_xy: (x,y) to project
    Returns: (theta, kappa) at nearest spline point
    """
    pts = np.array(spline_points)
    q = np.array(query_xy)

    # nearest point index
    dists = np.linalg.norm(pts - q, axis=1)
    idx = np.argmin(dists)

    # tangent angle (theta)
    if idx == 0:
        v = pts[1] - pts[0]
    elif idx == len(pts)-1:
        v = pts[-1] - pts[-2]
    else:
        v = pts[idx+1] - pts[idx-1]
    theta = np.arctan2(v[1], v[0])

    # curvature (kappa) via discrete 3-point circle
    if 0 < idx < len(pts)-1:
        a, b, c = pts[idx-1], pts[idx], pts[idx+1]
        ab, bc, ca = np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(a-c)
        s = (ab+bc+ca)/2.0
        area = max(s*(s-ab)*(s-bc)*(s-ca), 0.0)
        if area > 1e-12:
            R = (ab*bc*ca)/(4*np.sqrt(area))
            kappa = 1.0/R
        else:
            kappa = 0.0
    else:
        kappa = 0.0

    return pts[idx][0], pts[idx][1], theta, kappa

def clothoid_with_obstacle_anchors(start, goal, spline, mask, resolution, max_depth=10):
    """
    start, goal: Position objects with (x,y,theta,kappa)
    spline: smoothed path (Douglas–Peucker result) to pull tangents/curvature from
    mask, resolution: obstacle map for collision checks
    """

    def recurse(p0, p1, depth):
        # Try direct clothoid
        seg = ClothoidCurve.get_from_positions(p0, p1)
        bad_pt = find_closest_collision_point(seg, mask, resolution)
        if bad_pt is None:
            return [seg]
        print("Wut ", bad_pt)
        if depth >= max_depth:
            raise RuntimeError("Failed to find collision-free clothoid path")

        # Project onto spline to get consistent tangent/curvature
        x,y, theta_k, kappa_k = get_theta_kappa_from_spline(spline, bad_pt)
        print("X : ", x, "Y : ", y)

        anchor = Position(x, y, Angle.from_radians(theta_k), kappa_k)

        # Recurse into two subproblems
        left = recurse(p0, anchor, depth+1)
        right = recurse(anchor, p1, depth+1)
        return left + right

    return recurse(Position(start[0], start[1], Angle.from_radians(start[2]), start[3]), Position(goal[0], goal[1], Angle.from_radians(goal[2]), goal[3]), 0)