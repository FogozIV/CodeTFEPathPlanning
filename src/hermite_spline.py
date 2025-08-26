import sympy as sp

# Time parameter
t = sp.Symbol('t', real=True)

# Define coefficients: x(t) = a0 + a1*t + ... + a5*t^5
a = sp.symbols('a0:6')
b = sp.symbols('b0:6')

# Quintic polynomials
x = sum(a[i] * t**i for i in range(6))
y = sum(b[i] * t**i for i in range(6))

# Derivatives
dx = sp.diff(x, t)
dy = sp.diff(y, t)
ddx = sp.diff(dx, t)
ddy = sp.diff(dy, t)

# Boundary symbols
x0, y0, x1, y1 = sp.symbols('x0 y0 x1 y1')
theta0, theta1 = sp.symbols('theta0 theta1')
k0, k1 = sp.symbols('k0 k1')
v = sp.Symbol('v', positive=True)

# Velocity vectors from heading
vx0 = v * sp.cos(theta0)
vy0 = v * sp.sin(theta0)
vx1 = v * sp.cos(theta1)
vy1 = v * sp.sin(theta1)

# Accelerations from curvature: a = v^2 * k * normal
ax0 = -v**2 * k0 * sp.sin(theta0)
ay0 =  v**2 * k0 * sp.cos(theta0)
ax1 = -v**2 * k1 * sp.sin(theta1)
ay1 =  v**2 * k1 * sp.cos(theta1)

# Constraints: 6 for x(t), 6 for y(t)
constraints = [
    sp.Eq(x.subs(t, 0), x0),
    sp.Eq(dx.subs(t, 0), vx0),
    sp.Eq(ddx.subs(t, 0), ax0),
    sp.Eq(x.subs(t, 1), x1),
    sp.Eq(dx.subs(t, 1), vx1),
    sp.Eq(ddx.subs(t, 1), ax1),

    sp.Eq(y.subs(t, 0), y0),
    sp.Eq(dy.subs(t, 0), vy0),
    sp.Eq(ddy.subs(t, 0), ay0),
    sp.Eq(y.subs(t, 1), y1),
    sp.Eq(dy.subs(t, 1), vy1),
    sp.Eq(ddy.subs(t, 1), ay1),
]

# Solve for all 12 unknowns
sol = sp.solve(constraints, a + b, dict=True)

# Pretty print solution
if sol:
    for coeff, expr in sol[0].items():
        print(f"{coeff} =")
        sp.pprint(sp.simplify(expr))
        print()
else:
    print("❌ No solution found — check constraints.")
