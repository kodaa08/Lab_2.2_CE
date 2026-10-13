# =====================================================================================
#  Modelo de la caldera pirotubular de 4 pasos - Lab 2 "Circuito de humos Planta de Vapor"
#  Centrales de Energía 541477, Grupo 12 (UdeC)
#  Base: Farías & Mellado (2022), hogar de una zona con Lobo-Evans + pasos de tubos (eps-NTU, pared a Tsat)
#  Casos: gas natural (90 % CH4 + 10 % C2H6) y mezcla 10 % H2 molar, ambos con 20 % de exceso de aire.
#  Requiere: numpy, scipy, cantera, CoolProp.   Ejecutar:  python caldera.py
# =====================================================================================
import numpy as np, cantera as ct
from scipy.optimize import brentq
from CoolProp.CoolProp import PropsSI

HOGAR = dict(D=0.381, L=2.310)
PASOS = {'2° paso': dict(D=0.040, L=2.310, N=25),
         '3° paso': dict(D=0.040, L=2.310, N=18),
         '4° paso': dict(D=0.040, L=2.310, N=15)}
EXTREMOS_REFRACTARIOS = 1
GAS_NATURAL = {'CH4': 0.90, 'C2H6': 0.10}
FRAC_H2 = 0.10
Q_VAPOR_KW = 169.0
P_MAN_BAR = 7.0
P_ATM_BAR = 1.01325
T_AGUA_ALIM_C = 80.0
PERDIDA_CARCASA = 0.015
T_AIRE_C = 32.08          # °C aire de combustión
CO_PPM = 5.0              # ppm de CO en humos secos (pérdida por inquemados, despreciable)
E_AIRE = 0.20             # exceso de aire de diseño
F_U = 1.302               # factor de corrección de la convección en los tubos (Hipótesis 9)
EPS_SUPERFICIE = 0.8

SIG = 5.670374e-8
GAS = ct.Solution('gri30.yaml', transport_model='mixture-averaged')
P0 = ct.one_atm
ATOMOS = {'CH4': (1, 4, 0), 'C2H6': (2, 6, 0), 'H2': (0, 2, 0)}
MW = {'CH4': 16.043, 'C2H6': 30.070, 'H2': 2.016}
MW_PROD = {'CO2': 44.01, 'H2O': 18.015, 'O2': 31.998, 'N2': 28.014}

def normalizar(f):
    s = sum(f.values()); return {k: v / s for k, v in f.items() if v > 0}
def construir_mezcla(base, xh):
    base = normalizar(base); mix = {s: x * (1 - xh) for s, x in base.items()}
    if xh > 0: mix['H2'] = xh
    return mix
def relacion_CH(f):
    C = sum(x * ATOMOS[s][0] for s, x in f.items()) * 12.011
    H = sum(x * ATOMOS[s][1] for s, x in f.items()) * 1.008
    return C / H
def estq(f):
    o2 = sum(x * (ATOMOS[s][0] + ATOMOS[s][1] / 4 - ATOMOS[s][2] / 2) for s, x in f.items())
    co2 = sum(x * ATOMOS[s][0] for s, x in f.items()); h2o = sum(x * ATOMOS[s][1] / 2 for s, x in f.items())
    return o2, co2, h2o
def productos(f, lam):
    o2s, co2, h2o = estq(f)
    return {'CO2': co2, 'H2O': h2o, 'O2': (lam - 1) * o2s, 'N2': 3.76 * lam * o2s}
def o2_seco(f, lam):
    p = productos(f, lam); return p['O2'] / (p['CO2'] + p['O2'] + p['N2'])
def lam_desde_O2(f, o2dry): return brentq(lambda l: o2_seco(f, l) - o2dry, 1.0001, 6)
def _h(s, T):
    GAS.TPX = T, P0, {s: 1}; return GAS.enthalpy_mole / 1e6
def hP(p, T):
    tot = sum(p.values()); GAS.TPX = T, P0, {k: v / tot for k, v in p.items()}
    return GAS.enthalpy_mole / 1e6 * tot
def hR(f, lam, T):
    o2s = estq(f)[0]
    return sum(x * _h(s, T) for s, x in f.items()) + lam * o2s * (_h('O2', T) + 3.76 * _h('N2', T))
def props_comb(f):
    M = sum(x * MW[s] for s, x in f.items())
    lhv = hR(f, 1.0, 298.15) - hP(productos(f, 1.0), 298.15)
    o2s = estq(f)[0]
    return dict(M=M, PCI_mol=lhv, PCI_Nm3=lhv / 22.414e-3, aire_est=o2s * 4.76, CH=relacion_CH(f))
def T_adiabatica(f, lam, Tair_K):
    p = productos(f, lam); h0 = hR(f, lam, Tair_K)
    return brentq(lambda T: hP(p, T) - h0, 500, 3200)
def props_gas(p, T):
    tot = sum(p.values()); GAS.TPX = T, P0, {k: v / tot for k, v in p.items()}
    return GAS.cp_mass, GAS.viscosity, GAS.thermal_conductivity
def nusselt(Re, Pr):
    if Re < 2300: return 3.66
    f = (0.79 * np.log(Re) - 1.64) ** -2
    Nu = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * np.sqrt(f / 8) * (Pr ** (2 / 3) - 1))
    if Re < 4000:
        w = (Re - 2300) / 1700; return (1 - w) * 3.66 + w * Nu
    return Nu
_H2O_C = [[-2.2118, -1.1987, 0.035596], [0.85667, 0.93048, -0.14391], [-0.10838, -0.17156, 0.045915]]
_CO2_C = [[-3.9893, 2.7669, -2.1081, 0.39163], [1.2710, -1.1090, 1.0195, -0.21897], [-0.23678, 0.19731, -0.19544, 0.044644]]
def _eps0(coefs, paL, T):
    t = T / 1000.0; x = np.log10(max(paL, 1e-6))
    return np.exp(sum(c * t**j * x**i for i, row in enumerate(coefs) for j, c in enumerate(row)))
def eps_h2o(pw, L, T, P):
    t = T / 1000.0; e0 = _eps0(_H2O_C, pw * L * 100, T); pE = P + 2.56 * pw / np.sqrt(t)
    a = 2.144 if t < 0.75 else 1.888 - 2.053 * np.log10(t); b = 1.10 / t**1.4; c = 0.5; rm = 13.2 * t**2
    return e0 * (1 - (a - 1) * (1 - pE) / (a + b - 1 + pE) * np.exp(-c * np.log10(rm / max(pw * L * 100, 1e-6))**2))
def eps_co2(pc, L, T, P):
    t = T / 1000.0; e0 = _eps0(_CO2_C, pc * L * 100, T); pE = P + 0.28 * pc
    a = 1 + 0.1 / t**1.45; b = 0.23; c = 1.47; rm = 0.054 / t**2 if t < 0.7 else 0.225 * t**2
    return e0 * (1 - (a - 1) * (1 - pE) / (a + b - 1 + pE) * np.exp(-c * np.log10(rm / max(pc * L * 100, 1e-6))**2))
def d_eps(pw, pc, L):
    z = pw / (pw + pc); s = np.log10((pw + pc) * L * 100)
    return max((z / (10.7 + 101 * z) - 0.0089 * z**10.4) * max(s, 0)**2.76, 0)
def radiacion_gases(p, Tg, Ts, L):
    nt = sum(p.values()); pc = p['CO2'] / nt * P_ATM_BAR; pw = p['H2O'] / nt * P_ATM_BAR
    ec, ew, de = eps_co2(pc, L, Tg, P_ATM_BAR), eps_h2o(pw, L, Tg, P_ATM_BAR), d_eps(pw, pc, L)
    ac = (Tg / Ts)**0.65 * eps_co2(pc * Ts / Tg, L, Ts, P_ATM_BAR)
    aw = (Tg / Ts)**0.45 * eps_h2o(pw * Ts / Tg, L, Ts, P_ATM_BAR)
    return dict(eps_g=ec + ew - de, alfa_g=ac + aw - de)
def fv_hollin(e, CH):
    a1 = 8.42486e-13 * CH**2 + 4.34741e-12 * CH - 2.06195e-11
    a2 = 2.5162e-11 * CH**3 - 2.21318e-10 * CH**2 + 8.51554e-10 * CH + 9.45919e-10
    a3 = 7.85413e-8 * CH**2 - 2.92181e-7 * CH + 2.67785e-7
    return max(a1 * e**2 + a2 * e + a3, 0.0)
def eps_hollin(fv, T, L): return 1 - (1 + 350 * fv * T * L) ** -4
def Tsat_C(P): return PropsSI('T', 'P', P * 1e5, 'Q', 1, 'Water') - 273.15

def hogar(f, p, lam, n_f, mg, Qin, Tair, Ts, Tad):
    D, L = HOGAR['D'], HOGAR['L']
    Lm = 1.8 * D * L / (D + 2 * L); A_pe = np.pi * D * L; A_r = EXTREMOS_REFRACTARIOS * np.pi * D**2 / 4; F_RC = 1.0
    fv = fv_hollin(100 * (lam - 1), relacion_CH(f))
    def terms(Tg):
        rg = radiacion_gases(p, Tg, Ts, Lm)
        em = rg['eps_g'] + eps_hollin(fv, Tg, Lm); am = rg['alfa_g'] + eps_hollin(fv, Ts, Lm)
        ell = (em * Tg**4 - am * Ts**4) / (Tg**4 - Ts**4)
        ef = ell * (1 + (A_r / A_pe) / (1 + (ell / (1 - ell)) * (1 / F_RC)))
        Fb = 1 / (1 / ef + 1 / EPS_SUPERFICIE - 1)
        Qrad = SIG * Fb * (Tg**4 - Ts**4) * A_pe / 1e3
        cp, mu, k = props_gas(p, Tg); Re = 4 * mg / (np.pi * D * mu); Pr = cp * mu / k
        Qconv = A_pe * nusselt(Re, Pr) * k / D * (Tg - Ts) / 1e3
        return dict(Q_rad=Qrad, Q_conv=Qconv, Q_gsc=n_f * (hP(p, Tg) - hP(p, Tair)))
    Tg = brentq(lambda T: (lambda t: t['Q_rad'] + t['Q_conv'] + t['Q_gsc'] - Qin)(terms(T)), Ts + 30, Tad)
    r = terms(Tg); r['T_gsc'] = Tg - 273.15; return r

def paso(p, mg, n_f, Tin, Ts, geo, fU):
    D, L, N = geo['D'], geo['L'], geo['N']
    A = N * np.pi * D * L; Lm = 0.9 * D; Tout = Tin - 50
    for _ in range(100):
        Tm = 0.5 * (Tin + Tout); cp, mu, k = props_gas(p, Tm)
        Re = 4 * (mg / N) / (np.pi * D * mu); Pr = cp * mu / k
        hc = fU * nusselt(Re, Pr) * k / D
        rg = radiacion_gases(p, Tm, Ts, Lm)
        ell = (rg['eps_g'] * Tm**4 - rg['alfa_g'] * Ts**4) / (Tm**4 - Ts**4)
        Fb = 1 / (1 / ell + 1 / EPS_SUPERFICIE - 1)
        hr = SIG * Fb * (Tm**4 - Ts**4) / (Tm - Ts)
        Tn = Ts + (Tin - Ts) * np.exp(-(hc + hr) * A / (mg * cp))
        if abs(Tn - Tout) < 1e-8: break
        Tout = Tn
    return Tout, n_f * (hP(p, Tin) - hP(p, Tout))

def calcular(f, Qin, Tair_C, Tsat, fU, o2dry=None, lam=None, carcasa=PERDIDA_CARCASA):
    fp = props_comb(f); lam = lam if lam else lam_desde_O2(f, o2dry)
    p = productos(f, lam); Tair, Ts = Tair_C + 273.15, Tsat + 273.15
    Tad = T_adiabatica(f, lam, Tair); n_f = Qin / fp['PCI_mol']
    mg = n_f * sum(p[s] * MW_PROD[s] for s in p) * 1e-3
    H = hogar(f, p, lam, n_f, mg, Qin, Tair, Ts, Tad)
    temps = [Tad - 273.15, H['T_gsc']]; Qs = [H['Q_rad'] + H['Q_conv']]
    T = H['T_gsc'] + 273.15
    for geo in PASOS.values():
        T, Q = paso(p, mg, n_f, T, Ts, geo, fU); temps.append(T - 273.15); Qs.append(Q)
    ndry = sum(p.values()) - p['H2O']
    q_CO = CO_PPM * 1e-6 * ndry * 283.0 / fp['PCI_mol']
    eta_pci = sum(Qs) / Qin
    return dict(lam=lam, temps=temps, Qs=Qs, eta_pci=eta_pci, eta_net=eta_pci - carcasa - q_CO,
                T_chim=T - 273.15, Qin=Qin, Tad=Tad - 273.15, Nm3h=n_f * 22.414e-3 * 3600)

def Q_para_vapor(f, Tsat, fU, Qv=Q_VAPOR_KW, **kw):
    q = Qv / 0.9
    for _ in range(30):
        qn = Qv / calcular(f, q, T_AIRE_C, Tsat, fU, **kw)['eta_net']
        if abs(qn - q) < 1e-5: break
        q = qn
    return qn

GN = normalizar(GAS_NATURAL); MEZ = construir_mezcla(GN, FRAC_H2)
TSAT = Tsat_C(P_MAN_BAR + P_ATM_BAR)

# ================================ EJECUCIÓN ================================
if __name__ == '__main__':
    lam = 1 + E_AIRE
    for nombre, f in (('Gas natural', GN), ('Mezcla 10 % H2', MEZ)):
        fp = props_comb(f)
        Qin = Q_para_vapor(f, TSAT, F_U, lam=lam)
        R = calcular(f, Qin, T_AIRE_C, TSAT, F_U, lam=lam)
        PCI_kg = fp['PCI_mol'] / fp['M'] * 1e3                     # kJ/kg
        mc = Qin / PCI_kg * 3600                                    # kg/h
        RAC = lam * fp['aire_est'] * 28.85 / fp['M']
        print(f"\n=== {nombre} (e = {100*E_AIRE:.0f} %) ===")
        print(f"PCI = {PCI_kg:,.0f} kJ/kg | M = {fp['M']:.2f} kg/kmol | RAC = {RAC:.2f} kg/kg | T_ad = {R['Tad']:.0f} °C")
        print(f"Q_comb = {Qin:.2f} kW | m_comb = {mc:.2f} kg/h | m_aire = {RAC*mc:.1f} kg/h")
        print("T humos [°C] llama / salida hogar / 2° / 3° / 4° paso:", " / ".join(f"{t:.1f}" for t in R['temps']))
        print("Q [kW] hogar / 2° / 3° / 4° paso:", " / ".join(f"{q:.1f}" for q in R['Qs']))
        print(f"eta_PCI = {100*R['eta_pci']:.2f} % | eta_neto = {100*R['eta_net']:.2f} % | T_chim = {R['T_chim']:.1f} °C")
