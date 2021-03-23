import math
import collections

K, CL, CA, B = 'K', 'CL', 'CA', 'B'

z = {K: 1, CL: -1, CA: 2, B: 1}
D = {K: 1.96e-9, CL: 2.08e-9, CA: 0.79e-9 } # m^2/s ; https://www.aqion.de/site/diffusion-coefficients
Cm = 2e-6*1e4 # F/cm^2*(100cm/m)^2 F = 96485 # sA/mol | C/mol R = 8.314 # J/Kmol
T = 293.3 # K
F = 96485 # sA/mol | C/mol
R = 8.314 # J/Kmol
RT = R * T

alpha = 1/80**2 * 1e-12 # per nM^2 * 1e-12 mM^2/nM^2 (Ca2+ pump)

class Extracel:
    def __init__(self, **kw):
        self.mmol = {K: 0, CL: 0, CA: 0}
        self.mem_pot = 0
        self.mmol.update(kw)
        self.prev = None
        self.next = None
        self.out = None

    def half_G(self, k):
        return 1e-10
        return float('inf') # or equivalently; length=0

    def setup_sympy(self, name):
        import sympy
        self.sym_name = name
        def s(k):
            return sympy.Symbol(f'{name}_{k}')
        def g(k):
            return sympy.Symbol(f'{k}')
        overwrite = dict(
                mmol = {K: s('n_K'), CL: s('n_CL'), CA: s('n_CA'), B: s('n_B')},
                log = sympy.log,
                fsum = sum,
                exp = sympy.exp
        )
        save = {k: getattr(self, k) for k in overwrite if hasattr(self, k)}
        self.__dict__.update(overwrite)
        global_overwrite = dict(
            D = {K: g('D_K'), CL: g('D_CL'), CA: g('D_CA') }, # m^2/s
            Cm = g('Cm'), # F/cm^2*(100cm/m)^2
            F = g('F'), # sA/mol | C/mol
            R = g('R'), # J/Kmol
            T = g('T'), # K
            RT = g('R')*g('T')
        )
        G = globals()
        global_save = {k: G[k] for k in global_overwrite}
        G.update(global_overwrite)
        def reset():
            self.__dict__.update(save)
            globals().update(global_save)
            for k in overwrite:
                if k not in save:
                    del self.__dict__[k]
        return reset

    def neuroml_state(self, name='out'):
        state = {}
        for k, v in self.mmol.items():
            state[f'{name}_n_{k}'] = v
        return state

class Compartment:
    def __init__(self, *, v=-61.e-3, diam_um=0, length_um=0, **kw):
        self.mmol = {K: 0, CL: 0, CA: 0}
        #self.mem_pot = -60e-3
        self.mem_pot = v
        self.diam = diam_um * 1e-6
        self.length = length_um * 1e-6
        self.mmol.update(kw)
        self.prev = None
        self.next = None
        self.I = {K: 0, CL: 0, CA: 0}
        self.grad_v = 0
        self.mmol_grad = {CL: 0, CA: 0}
        self.E = {}
        self.update_ions = [CL, CA]

        self.log = math.log
        self.exp = math.exp
        self.fsum = math.fsum
        self.pi = math.pi

    def const(self, name, value, unit_value):
        if hasattr(self, 'sym_name'):
            import sympy
            sym = sympy.Symbol(f'{self.sym_name}_{name}')
            self.sym_consts.append((sym, value))
            return sym
        return float(value)

    def setup_sympy(self, name):
        import sympy
        self.sym_name = name
        def s(k):
            return sympy.Symbol(f'{name}_{k}')
        def g(k):
            return sympy.Symbol(f'{k}')
        overwrite = dict(
                mmol = {K: s('n_K'), CL: s('n_CL'), CA: s('n_CA'), B: s('n_B'),
                    'X': s('X'), 'Y': s('Y'), 'ggaba': s('ggaba'), 'ca_presyn': s('ca_presyn')
                    },
                log = sympy.log,
                fsum = sum,
                exp = sympy.exp,
                I = {K: s('I_K'), CL: s('I_CL'), CA: s('I_CA')},
                diam = s('diam'),
                length = s('length'),
                mem_pot = s('V'),
                ggap = s('ggap'),
                pi = sympy.pi
        )
        save = {k: getattr(self, k) for k in overwrite if hasattr(self, k)}
        self.__dict__.update(overwrite)
        global_overwrite = dict(
            D = {K: g('D_K'), CL: g('D_CL'), CA: g('D_CA') }, # m^2/s
            Cm = g('Cm'), # F/cm^2*(100cm/m)^2
            F = g('F'), # sA/mol | C/mol
            R = g('R'), # J/Kmol
            T = g('T'), # K
            RT = g('R')*g('T')
        )
        G = globals()
        global_save = {k: G[k] for k in global_overwrite}
        G.update(global_overwrite)
        self.sym_consts = []
        def reset():
            self.__dict__.update(save)
            G.update(global_save)
            for k in overwrite:
                if k not in save:
                    del self.__dict__[k]
            del self.__dict__['sym_name']
        return reset


    def generate_neuroml(self):
        name = getattr(self, 'sym_name', None)
        if name is None:
            raise Exception('call setup_sympy first')
        self.calc_grads()
        grads = {
                f'{name}_V': self.grad_v,
        }
        for ion in self.update_ions:
            grads[f'{name}_n_{ion}'] = self.mmol_grad[ion]
        return grads

    @property
    def vol(self):
        return self.length * self.diam**2 * self.pi / 4

    @property
    def ex_area(self):
        A = self.diam * self.pi * self.length
        #if self.prev == None: A += circle_area
        #if self.next == None: A += circle_area
        return A

    @property
    def circle_area(self):
        return self.diam**2 * self.pi / 4

    def half_G(self, k):
        inv_rho = F**2*z[k]**2/RT*D[k]*self.mmol[k]
        #print(self.mmol[k])
        #print('-->', 1e-9/(F**2/RT*D[k]*z[k]**2*15*0.01e-6))
        # if area increases; G increases
        # if length increases; G decreases
        return inv_rho / (0.5*self.length) * self.circle_area
        #print(a, b)

    def E_k(self, other, k):
        a = {K:-89e-3, CL:-89e-3, CA:138e-3}[k]
        b = RT/(F*z[k]) * self.log(other.mmol[k]/self.mmol[k])
        self.E[k] = b
        #if other == self.out: print('E', k, b)
        return b

    def current_i_k(self, other, k):
        G_l = self.half_G(k)
        G_r = other.half_G(k)
        Gtot = 1/(1/G_l + 1/G_r)
        E = self.E_k(other, k)
        #print(f'{k}: {G_l:g} {G_r:g} {Gtot:g} {E}')
        # inward current
        # if self > other; outward
        # if self < other; inward
        return -Gtot*(E + self.mem_pot - other.mem_pot)

    def mech_I_ca_pump(self, k):
        # 50 nM = 50e-6 mM
        # 80 nM = 80e-6 mM # XXX maybe /nM?
        if k != CA:
            return 0
        return -F*z[k]*(self.mmol[k]-50e-6)*80e-6*alpha

    def mech_I_long_diff(self, k):
        if not (k == CA or k == CL or k == K):
            return 0
        Iprev = self.current_i_k(self.prev, k)
        Inext = self.current_i_k(self.next, k)
        return (Iprev + Inext)

    def mech_I_leak(self, k):
        if k != CL:
            return 0
        # Inward leak
        # E_k ~ -70mV
        # If self.mem_pot < E_k inward, else outward
        return self.const('Gleak', 1e-8, 'missing')*(self.E_k(self.out, k) - self.mem_pot)
        #return 1e-8*(-self.E_k(self.out, k) + self.mem_pot)

    def current_k(self, k):
        'Inward current for species <k>'
        I = 0
        for method_name in dir(self):
            if method_name.startswith('mech_I_'):
                f = getattr(self, method_name)
                i = f(k)
                #print(method_name, i)
                I += i
        return I

    def calc_grads(self):
        for method_name in dir(self):
            if method_name.startswith('mech_grad_calc_'):
                f = getattr(self, method_name)
                f()
        Iks = []
        for k in self.update_ions:
            Ik = self.I[k] = self.current_k(k)
            #print('Current', k, Ik)
            Iks.append(Ik)
            self.mmol_grad[k] = 1/self.vol * Ik / z[k] / F
        I_tot = self.fsum(Iks)
        self.grad_v = I_tot  / (Cm * self.ex_area)

    def update_grads(self, dt):
        self.mem_pot += self.grad_v * dt
        for k in self.update_ions:
            self.mmol[k] += self.mmol_grad[k] * dt
        for method_name in dir(self):
            if method_name.startswith('mech_grad_update_'):
                f = getattr(self, method_name)
                f(dt)

    def Rtot(self):
        Gtot = 0
        for k in [K, CL, CA]:
            Gtot += 0.5*self.half_G(k)
            print(f'RR {k} {1e-6*2/self.half_G(k):.1f} MegaOhm')
        return 1/Gtot

    def neuroml_state(self, name):
        assert not hasattr(self, 'sym_name')
        state = {
                f'{name}_V': self.mem_pot,
                f'{name}_diam': self.diam,
                f'{name}_length': self.diam,
        }
        for k, v in self.mmol.items():
            state[f'{name}_n_{k}'] = v
        return state

class SympyDend(Compartment):
    def half_G(self, k):
        import sympy
        return self.const(f'half_G_{k}', 100, 'missing')

class SpineHead(Compartment):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.update_ions.append(B)
        self.is_spike = 0 if False else 1
        self.nmda_t = 0
        self.ggap = 15e-12 # half of 30 pS baseline

    def mech_I_kcc2(self, k):
        'Lewin 2012'
        if k != CL:
            return 0
        I_max = self.const('I_max', 30, 'A/m^2 = 3.0 mA/cm^2')
        Ko, CLo, Ki, CLi, = self.out.mmol[K], self.out.mmol[CL], self.mmol[K], self.mmol[CL]
        kK = self.const('kK', 9, 'mM')
        kCL = self.const('kCL', 1, 'mM')

        # Ikcc2 has same units as I_max ; A/m^2
        Ikcc2 =  -I_max*(Ko*CLo-Ki*CLi)/kK*kCL/(
                                    (1+(Ko*CLo)/(kK*kCL)) * (1+Ki/kK) * (1+CLi/kCL)+
                                    (1+(Ki*CLi)/(kK*kCL)) * (1+Ko/kK) * (1+CLo/kCL))

        return Ikcc2 * self.ex_area

    def mech_I_NMDA(self, k):
        'Jahr and Steven 1990'
        if k != 'CA':
            return 0
        Vhead = self.mem_pot
        n_Mg_out = self.const('n_Mg_out', 2, 'mM (?)')
        gbar = self.const('gbar_nmda', 0.2e-9, '0.2 nS')
        tau_decay_nmda = self.const('tau_decay_nmda', 89e-3, '89 ms')
        tau_rise_nmda = self.const('tau_rise_nmda', 1.2e-3, '1.2 ms')
        E_K_NA = self.const('E_K_NA', 10e-3, '10 mV')
        E_head_ca = self.const('E_head_ca', 10e-3, '10 mV')
        # units of 0.063 are a bit unclear, but Vhead is mV originally
        g_nmda = gbar*(self.exp(-self.nmda_t/tau_decay_nmda)-self.exp(-self.nmda_t/tau_rise_nmda))/(
                1.0 + self.exp(-0.063e3*Vhead)*n_Mg_out/3.57);
        I_nmda = 0.9*g_nmda*(Vhead-E_K_NA) + 0.1*g_nmda*(Vhead-E_head_ca);
        return I_nmda

    def mech_grad_calc_ggaba(self):
        # X, Y, ggaba:
        tau_d = self.const('tau_d', 255e-3, '255 ms')
        tau_r = self.const('tau_r', 5e-3, '5 ms')
        gbar = self.const('gbar_gaba', 1e-9, '1 nS')
        # tau gaba:
        tau_gaba0 = self.const('tau_gaba0', 30e-3, '30ms')
        eta_gaba = self.const('eta_gaba', 34e-3, '34 ms')
        theta_gaba = self.const('theta_gaba', 1.2e-3, '1.2 uM')
        sigma_gaba = self.const('sigma_gaba', 60e-6, '60 nM')
        # ca_presyn:
        beta = self.const('beta', 0.08, '80 nM/ms')
        Kp = self.const('Kp', 0.2e-3, '0.2 uM')
        Ip = self.const('Ip', 0.47e-3, '0.47 nM/ms')
        gamma = self.const('gamma', 0, 'missing')

        Y = self.mmol.get('Y', 0)
        X = self.mmol.get('X', 0)
        ggaba = self.mmol.get('ggaba', 0)
        ca_presyn = self.mmol.get('ca_presyn', 1e-9)

        tau_gaba = tau_gaba0 + eta_gaba / (1.0+self.exp(-(ca_presyn-theta_gaba)/sigma_gaba));
        self.grad_Y = -Y/tau_d
        self.grad_X = (1-X-Y)/tau_r
        self.grad_ggaba = -ggaba/tau_gaba + gbar * Y
        self.grad_ca_presyn = (
                -beta*ca_presyn**2/(Kp**2 + ca_presyn**2) +
                gamma*self.log(2 / ca_presyn)*self.is_spike + Ip)

    def mech_grad_update_ggaba(self, dt):
        U = 0.3 # unit is reciprocal X's ???
        Y = self.mmol.get('Y', 0)
        X = self.mmol.get('X', 0)
        ggaba = self.mmol.get('ggaba', 0)
        ca_presyn = self.mmol.get('ca_presyn', 1e-9)
        self.Y = Y + self.grad_Y*dt + self.is_spike*X*U
        self.X = X + self.grad_X*dt - self.is_spike*X*U
        self.ggaba = ggaba + self.grad_ggaba*dt
        self.ca_presyn = ca_presyn + self.grad_ca_presyn*dt

    def mech_I_buffer(self, k):
        'Donnel et al.'
        if k != CA and k != B:
            return 0
        nB = self.mmol[B]
        nCA = self.mmol[CA]
        nTB = self.const('nTB', 100e-3, '100 uM')
        kf = self.const('kf', 1e6, '10e8 per Ms cubicm/mols')
        kb = self.const('kb', 500, 'per s')
        J = kf*(nB*nCA) - kb*(nTB-nB)
        I = -F*z[CA]*self.vol*J
        if k == CA:
            return -I
        elif k == B:
            return I

    def current_i_k(self, other, k):
        if not isinstance(other, SpineHead):
            return super().current_i_k(other, k)
        Gtot = 1/(1/self.ggap + 1/other.ggap + 1/self.half_G(k) + 1/other.half_G(k))
        return Gtot*(other.mem_pot - self.mem_pot)

    def generate_neuroml(self):
        grads = super().generate_neuroml()
        self.mech_grad_calc_ggaba()
        grads[f'{self.sym_name}_Y'] = self.grad_Y
        grads[f'{self.sym_name}_X'] = self.grad_X
        grads[f'{self.sym_name}_ggaba'] = self.grad_ggaba
        grads[f'{self.sym_name}_ca_presyn'] = self.grad_ca_presyn
        return grads

    def neuroml_state(self, name):
        state = super().neuroml_state(name)
        state[f'{name}_ggaba'] = getattr(self, 'ggaba', 1e-10)
        state[f'{name}_ggap'] = self.ggap
        state[f'{name}_Y'] = getattr(self, 'Y', 0)
        state[f'{name}_X'] = getattr(self, 'X', 0)
        state[f'{name}_ggaba'] = getattr(self, 'ggaba', 0)
        state[f'{name}_ca_presyn'] = getattr(self, 'ca_presyn', 0)
        return state

def simulate_one_side():
    out = Extracel(K=3, CL=134, CA=2)
    dend = Compartment(K=85, CA=15, CL=15, length_um=2, diam_um=1)
    neck = Compartment(K=85, CA=15, CL=15, length_um=1, diam_um=0.1)
    head = SpineHead(K=85, CA=15, CL=15, length_um=0.69, diam_um=0.2, B=0)

    dend.mem_pot = -55e-3

    morph = [out, out, dend, neck, head, out, out]
    for i, j in zip(morph[:-1], morph[1:]):
        i.next = j
        j.prev = i
        i.out = out

    print(f'Rtot neck {neck.Rtot()*1e-6:g} MegaOhm')
    print(f'Rtot head {dend.Rtot()*1e-6:g} MegaOhm')

    dt = .1e-7
    t = 0

    for _i in range(1000000):
        t = _i * dt
        head.nmda_t = t*10
        #print('neck')
        neck.calc_grads()
        #print('head')
        head.calc_grads()
        neck.update_grads(dt)
        head.update_grads(dt)
        dend.mem_pot = -55e-3 + 15e-3*self.sin(_i/2000)
        if _i % 50 == 0:
            #print(f'# {dend.mem_pot*1e3:.2f} {neck.mem_pot*1e3:.2f}mV {head.mem_pot*1e3:.2f}mV')
            #print(f'E_Cl: {neck.E[CL]*1e3:.3g}mV {head.E[CL]*1e3:.2g}mV')
            #print(f'Ca2+: {head.mmol[CA]:.3g}mM {neck.mmol[CA]:.2g}mM')
            #print(f'{head.mmol_grad} {neck.mmol_grad}')
            #print(f'{head.ggaba}')
            yield dict(dend=dend.mem_pot*1e3,
                       neck=neck.mem_pot*1e3,
                       head=head.mem_pot*1e3)

def simulate_glomerulus():
    out = Extracel(K=3, CL=134, CA=2)
    # left
    dend1 = Compartment(K=85, CA=15, CL=15, length_um=2, diam_um=1)
    neck1 = Compartment(K=85, CA=10, CL=5, length_um=1, diam_um=0.1)
    head1 = SpineHead(K=85, CA=10, CL=3.5, length_um=0.69, diam_um=0.1, B=0)
    # right
    dend2 = Compartment(K=85, CA=15, CL=15, length_um=2, diam_um=1)
    neck2 = Compartment(K=85, CA=10, CL=5, length_um=1, diam_um=0.1)
    head2 = SpineHead(K=85, CA=10, CL=3.5, length_um=0.69, diam_um=0.1, B=0)

    dend1.mem_pot = dend2.mem_pot = -55e-3
    head1.mem_pot = head2.mem_pot = -91.2e-3
    neck1.mem_pot = neck2.mem_pot = -81.7e-3

    morph = [out, out, dend1, neck1, head1, head2, neck2, dend2, out, out]
    for i, j in zip(morph[:-1], morph[1:]):
        i.next = j
        j.prev = i
        i.out = out

    Rtot = neck1.Rtot() + head1.Rtot() + neck2.Rtot() + head2.Rtot() + 1/head1.ggap + 1/head2.ggap
    print(f'Rtot glomerulus {Rtot*1e-6:g} MegaOhm')

    dt = .1e-7
    t = 0

    for _i in range(1000000):
        t = _i * dt
        head1.nmda_t = t
        head2.nmda_t = t
        # calculate gradients
        neck1.calc_grads()
        neck2.calc_grads()
        head1.calc_grads()
        head2.calc_grads()
        dend2.calc_grads()
        # apply gradients
        neck1.update_grads(dt)
        neck2.update_grads(dt)
        head1.update_grads(dt)
        head2.update_grads(dt)
        dend2.update_grads(dt)
        dend1.mem_pot = -55e-3 + 15e-3*math.sin(_i/2000)
        dend2.mem_pot = -55e-3 #- 15e-3*math.sin(_i/2000)
        if _i % 50 == 0:
            #print(f'# {dend1.mem_pot*1e3:.2f} {neck1.mem_pot*1e3:.2f}mV {head1.mem_pot*1e3:.2f}mV')
            #print(f'E_Cl: {neck.E[CL]*1e3:.3g}mV {head.E[CL]*1e3:.2g}mV')
            #print(head1.mmol, neck1.mmol)
            #print(f'Ca2+: {head.mmol[CA]:.3g}mM {neck.mmol[CA]:.2g}mM')
            #print(f'{head.mmol_grad} {neck.mmol_grad}')
            #print(f'{head.ggaba}')
            yield dict(dend1=dend1.mem_pot*1e3,
                       neck1=neck1.mem_pot*1e3,
                       head1=head1.mem_pot*1e3,
                       dend2=dend2.mem_pot*1e3,
                       neck2=neck2.mem_pot*1e3,
                       head2=head2.mem_pot*1e3)

def runsim():
    import cv2 as cv
    import numpy as np
    import matplotlib.pyplot as plt

    #SIMULATION = simulate_one_side()
    SIMULATION = simulate_glomerulus()

    cv.startWindowThread()
    cv.namedWindow('preview')
    lines = collections.defaultdict(list)
    fig = plt.figure()

    for i, data in enumerate(SIMULATION):
        data['i'] = i
        for k, v in data.items():
            if len(lines[k]) > 500:
                lines[k].pop(0)
            lines[k].append(v)
        if i % 10 == 0:
            #d = np.array(lines['d']).std()
            #h = np.array(lines['h']).std()
            #print('Dendrite-head coupling:', h/d)
            d = np.array(lines['dend1']).std()
            h = np.array(lines['dend2']).std()
            print('Dendrite-Dendrite coupling:', h/d)
            plt.cla()
            for k, v in lines.items():
                if k == 'i': continue
                plt.plot(lines['i'], v, label=k)
            plt.legend()
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            cv.imshow('preview', img)
            key = cv.waitKey(1)
            if key == ord('q'):
                break

    cv.destroyAllWindows()

def codegen():
    import sympy
    from sympy.codegen.ast import Assignment
    out = Extracel(K=3, CL=134, CA=2)
    # left
    dend1 = SympyDend(K=85, CA=15, CL=15, length_um=2, diam_um=1)
    neck1 = Compartment(K=85, CA=10, CL=5, length_um=1, diam_um=0.1)
    head1 = SpineHead(K=85, CA=10, CL=3.5, length_um=0.69, diam_um=0.1, B=0)
    # right
    dend2 = SympyDend(K=85, CA=15, CL=15, length_um=2, diam_um=1)
    neck2 = Compartment(K=85, CA=10, CL=5, length_um=1, diam_um=0.1)
    head2 = SpineHead(K=85, CA=10, CL=3.5, length_um=0.69, diam_um=0.1, B=0)

    morph = [out, out, dend1, neck1, head1, head2, neck2, dend2, out, out]
    for i, j in zip(morph[:-1], morph[1:]):
        i.next = j
        j.prev = i
        i.out = out

    state = {}
    state.update(out.neuroml_state('out'))
    state.update(neck1.neuroml_state('neck1'))
    state.update(head1.neuroml_state('head1'))
    state.update(neck2.neuroml_state('neck2'))
    state.update(head2.neuroml_state('head2'))
    state.update(dend1.neuroml_state('dend1'))
    state.update(dend2.neuroml_state('dend2'))

    reset = []
    reset.append(neck1.setup_sympy('neck1'))
    reset.append(head1.setup_sympy('head1'))
    reset.append(neck2.setup_sympy('neck2'))
    reset.append(head2.setup_sympy('head2'))
    reset.append(out.setup_sympy('out'))
    reset.append(dend1.setup_sympy('dend1'))
    reset.append(dend2.setup_sympy('dend2'))

    grads = {}
    grads.update(neck1.generate_neuroml())
    grads.update(head1.generate_neuroml())
    grads.update(neck2.generate_neuroml())
    grads.update(head2.generate_neuroml())

    consts = sum([dend1.sym_consts, dend2.sym_consts, neck1.sym_consts, neck2.sym_consts, head1.sym_consts, head2.sym_consts], start=[])

    final_grads = []

    for k, v in grads.items():
        final_grads.append(Assignment(sympy.Symbol(f'grad_{k}'), v))

    seen = set()
    final_consts = []
    for k, v in consts:
        if k not in seen:
            final_consts.append(Assignment(k, v))
            seen.add(k)

    final_state = []

    for k, v in state.items():
        final_state.append(Assignment(sympy.Symbol(k), v))

    for f in reset:
        f()

    return dict(state=final_state, const=final_consts, grads=final_grads)

def print_ccode():
    import sympy
    from sympy.printing import ccode
    from sympy.codegen.ast import CodeBlock

    code = codegen()

    print('''#include <math.h>\n#include <stdio.h>\n''')
    print('''double Cm = 2e-6*1e4;
double D_CA = 0.79e-9;
double D_CL = 2.08e-9;
double F = 96485;
double R = 8.314;
double T = 293;
''')

    for eq in code['const']:
        print('const double', ccode(eq, standard='c99'))

    for eq in code['state']:
        print('double', ccode(eq, standard='c99'))

    print('void timestep(double dt) {')
    for eq in CodeBlock(*code['grads']).cse():
        print('    double', ccode(eq, standard='c99'))

    for eq in code['grads']:
        k = eq.lhs
        print(f'    {str(k).replace("grad_", "")} += dt * {k};');
    print('}')

    print('''int main() {
    for (int j = 0; j < 10; j++) {
        for (long i = 0; i < 10*1000*1000; i++) {
            timestep(.1e-7);
        }
        printf("%f %f %f %f\\n", neck1_V, head1_V, head2_V, neck1_V);
    }
}
    ''')

def print_neuroml():
    import sympy
    from sympy.printing import ccode
    from sympy.codegen.ast import CodeBlock

    def lexp(e):
        'Sympy to LEMS expression'
        return str(e).replace('**', '^')

    code = codegen()

    print('''<Lems>
    <Include file="NeuroMLCoreCompTypes.xml"/>
    <Include file="Simulation.xml" />
    <ComponentType name="ggj" description="de Gruijl 2016">
        <Constant name="D_CA" dimension="none" value="0.79e-9"/>
        <Constant name="D_CL" dimension="none" value="2.08e-9"/>
        <Constant name="F" dimension="none" value="96485"/>
        <Constant name="R" dimension="none" value="8.314"/>
        <Constant name="T" dimension="none" value="293"/>
        <Constant name="pi" dimension="none" value="3.14159265"/>
        <Constant name="Cm" dimension="none" value="2e-2"/>
        <Constant name="qqqqq" dimension="time" value="1s"/>
''')

    for eq in code['const']:
        print(8*' ' + f'<Constant name="{eq.lhs}" dimension="none" value="{eq.rhs}"/>')
    for eq in code['state']:
        if f'grad_{eq.lhs}' not in [str(g.lhs) for g in code['grads']]:
            print(8*' ' + f'<Constant name="{eq.lhs}" dimension="none" value="{eq.rhs}"/>')

    print(8*' ' + '<Dynamics>')
    for eq in code['state']:
        if f'grad_{eq.lhs}' in [str(g.lhs) for g in code['grads']]:
            print(12*' ' + f'<StateVariable name="{eq.lhs}" dimension="none" exposure="{eq.lhs}"/>')
        # DerivedVariable name=i dimension=current exposure=i value=g*E-v

    for eq in code['grads']:
        k = str(eq.lhs).replace('grad_', '')
        print(12*' ' + f'<TimeDerivative variable="{k}" value="({lexp(eq.rhs)})/qqqqq"/>')

    print(12*' ' + f'<OnStart>')
    for eq in code['state']:
        if f'grad_{eq.lhs}' in [str(g.lhs) for g in code['grads']]:
            print(16*' ' + f'<StateAssignment variable="{eq.lhs}" value="{eq.rhs}"/>')
    print(12*' ' + f'</OnStart>')

    print(8*' ' + '</Dynamics>')

    #for eq in code['grads']: k = eq.lhs; print(f'    {str(k).replace("grad_", "")} += dt * {k};');
    for eq in code['state']:
        if f'grad_{eq.lhs}' in [str(g.lhs) for g in code['grads']]:
            print(8*' ' + f'<Exposure name="{eq.lhs}" dimension="none"/>')

    print('''
    </ComponentType>
    <ggj id="ggj1"/>
    <Simulation id="sim1" length="0.001s" step="0.00000001s" target="ggj1">
        <Display id="d0" title="Plot title" timeScale="1ms" xmin="0"  xmax="1" ymin="-100" ymax="100">
            <Line id="test" quantity="head1_V" scale="1" color="#FF0000" timeScale="1ms" />
        </Display>
    </Simulation>
    <Target component="sim1" />
</Lems>
    ''')


print_neuroml()
#runsim()
