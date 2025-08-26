import casadi as ca
import numpy as np

m1_sqrt_pi = 0.564189583547756286948079451561

def compute_C_S_with_ca_if_else(x):
    eps = 1e-15

    # Absolute value and sign handling
    absx = ca.fabs(x)
    sign = ca.if_else(x >= 0, 1, -1)

    # region1: |x| < 1.0 (series expansion)
    def region1(absx):
        s = ca.pi/2 * absx**2
        t = -s**2

        # Cosine integral series
        sum_c = 1.0
        term = 0
        twofn = 0.0
        fact = 1.0
        denterm = 1.0
        numterm = 1.0

        for _ in range(20):  # fixed loop count for CasADi graph
            twofn += 2.0
            fact *= twofn * (twofn - 1.0)
            denterm += 4.0
            numterm *= t
            term = numterm / (fact * denterm)
            sum_c += term

        C = absx * sum_c

        # Sine integral series
        sum_s = 1.0/3.0
        twofn = 1.0
        fact = 1.0
        denterm = 3.0
        numterm = 1.0

        for _ in range(20):
            twofn += 2.0
            fact *= twofn * (twofn - 1.0)
            denterm += 4.0
            numterm *= t
            term = numterm / (fact * denterm)
            sum_s += term

        S = (ca.pi/2) * sum_s * (absx**3)

        return C, S

    # region2: 1 ≤ |x| < 6 (rational approximation)
    fn = [0.49999988085884732562, 1.3511177791210715095, 1.3175407836168659241, 1.1861149300293854992,
          0.7709627298888346769, 0.4173874338787963957, 0.19044202705272903923, 0.06655998896627697537,
          0.022789258616785717418, 0.0040116689358507943804, 0.0012192036851249883877]
    fd = [1.0, 2.7022305772400260215, 4.2059268151438492767, 4.5221882840107715516, 3.7240352281630359588,
          2.4589286254678152943, 1.3125491629443702962, 0.5997685720120932908, 0.20907680750378849485,
          0.07159621634657901433, 0.012602969513793714191, 0.0038302423512931250065]
    gn = [0.50000014392706344801, 0.032346434925349128728, 0.17619325157863254363, 0.038606273170706486252,
          0.023693692309257725361, 0.007092018516845033662, 0.0012492123212412087428,
          0.00044023040894778468486, -8.80266827476172521e-6,
          -1.4033554916580018648e-8, 2.3509221782155474353e-10]
    gd = [1.0, 2.0646987497019598937, 2.9109311766948031235, 2.6561936751333032911, 2.0195563983177268073,
          1.1167891129189363902, 0.57267874755973172715, 0.19408481169593070798, 0.07634808341431248904,
          0.011573247407207865977, 0.0044099273693067311209, -0.00009070958410429993314]

    def region2(absx):
        sumn, sumd = 0.0, fd[-1]
        for k in range(10, -1, -1):
            sumn = fn[k] + absx * sumn
            sumd = fd[k] + absx * sumd
        f = sumn / sumd

        sumn, sumd = 0.0, gd[-1]
        for k in range(10, -1, -1):
            sumn = gn[k] + absx * sumn
            sumd = gd[k] + absx * sumd
        g = sumn / sumd

        U = ca.pi/2 * absx**2
        SinU, CosU = ca.sin(U), ca.cos(U)
        C = 0.5 + f * SinU - g * CosU
        S = 0.5 - f * CosU - g * SinU
        return C, S

    # region3: |x| ≥ 6 (asymptotic expansion)
    def region3(absx):
        s = ca.pi * absx**2
        t = -1 / (s**2)

        numterm = -1.0
        term = 1.0
        sum_f = 1.0
        for _ in range(15):
            numterm += 4.0
            term *= numterm * (numterm - 2.0) * t
            sum_f += term
        f = sum_f / (ca.pi * absx)

        numterm = -1.0
        term = 1.0
        sum_g = 1.0
        for _ in range(15):
            numterm += 4.0
            term *= numterm * (numterm + 2.0) * t
            sum_g += term
        g = sum_g / (ca.pi * absx**3)

        U = ca.pi/2 * absx**2
        SinU, CosU = ca.sin(U), ca.cos(U)
        C = 0.5 + f * SinU - g * CosU
        S = 0.5 - f * CosU - g * SinU
        return C, S

    # combine regions with if_else
    C1, S1 = region1(absx)
    C2, S2 = region2(absx)
    C3, S3 = region3(absx)

    C = ca.if_else(absx < 1.0, C1,
                   ca.if_else(absx < 6.0, C2, C3))
    S = ca.if_else(absx < 1.0, S1,
                   ca.if_else(absx < 6.0, S2, S3))

    # apply sign correction for negative x
    C = sign * C
    S = sign * S
    return C, S


def fresnel_cs_symbolic(x, nk=1):
    C, S = compute_C_S_with_ca_if_else(x)
    if nk == 2:
        dC = ca.cos(0.5 * ca.pi * x ** 2)
        dS = ca.sin(0.5 * ca.pi * x ** 2)
        return [C, dC], [S, dS]
    if nk == 3:
        dC = ca.cos(0.5 * ca.pi * x ** 2)
        dS = ca.sin(0.5 * ca.pi * x ** 2)
        ddC = -ca.pi * x * ca.sin(0.5 * ca.pi * x ** 2)
        ddS = ca.pi * x * ca.cos(0.5 * ca.pi * x ** 2)
        return np.array([C, dC, ddC]), np.array([S, dS, ddS])
    return np.array([C]), np.array([S])

def do_dc_ds_1(ell, z, cg, sg, s, dC, dS):
    X = np.array([cg * dC[0] - s * sg * dS[0]])
    Y = np.array([cg * dC[1] - s * sg * dS[1]])
    return X, Y
def do_dc_ds_2(ell, z, cg, sg, s, dc, ds):
    X, Y = do_dc_ds_1(ell, z, cg, sg, s, dc, ds)
    cg/=z
    sg/=z
    DC = dc[1] - ell * ds[0]
    DS = ds[1] - ell * ds[1]
    X = np.append(X, cg * DC - s * sg * DS)
    Y = np.append(Y, cg * DC + s * sg * DS)
    return X, Y
def do_dc_ds_3(ell, z, cg, sg, s, dc, ds):
    X, Y = do_dc_ds_2(ell, z, cg, sg, s, dc, ds)
    cg/=z*z
    sg/=z*z
    DC = dc[2] + ell * (ell * dc[0] - 2 * dc[1])
    DS = ds[2] + ell * (ell * ds[0] - 2 * ds[1])
    X = np.append(X, cg * DC - s * sg * DS)
    Y = np.append(Y, cg * DC + s * sg * DS)
    return X, Y

def lommel_reduced(mu, nu, b):
    tmp = 1.0 / ((mu + nu + 1) * (mu - nu + 1))
    res = tmp

    for n in range(1, 101):
        # replicate: tmp *= (-b / (2n + mu - nu + 1)) * (b / (2n + mu + nu + 1))
        tmp = tmp * (-b / (2*n + mu - nu + 1)) * (b / (2*n + mu + nu + 1))
        res = res + tmp
    return res

def eval_xy_a_large(a,b, nk=1):
    s = ca.if_else(a > 0.0, 1, -1)
    abs_a = ca.fabs(a)
    z = m1_sqrt_pi * ca.sqrt(abs_a)
    ell = s * b * m1_sqrt_pi/ca.sqrt(abs_a)
    g = -0.5*s*ca.constpow(b, 2) /abs_a
    cg = ca.cos(g) /z
    sg = ca.sin(g) / z

    Cl, Sl = fresnel_cs_symbolic(ell, nk)
    Cz, Sz = fresnel_cs_symbolic(ell + z, nk)

    dC = Cz - Cl
    dS = Sz - Sl
    return ca.if_else(nk==1, do_dc_ds_1, ca.if_else(nk==2, do_dc_ds_2, do_dc_ds_3))(ell, z, cg, sg, s, dC, dS)



x = ca.MX.sym('x')
C, S = fresnel_cs_symbolic(x)
fresnel_fun = ca.Function('FresnelCS', [x], [C, S])