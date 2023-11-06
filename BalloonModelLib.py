#%%
import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.signal import convolve

# from numba import jit, njit, cfunc
from random import choices as rch

def neural_response(stimulus: np.ndarray, dt:np.float64= 0.01, N_0:bool= False, scale:bool= True, params=None):
    '''
    neural_response
    
    Solves the inhibition equation and returns the neural response according to the equation system eq.14.
    
    **Inputs:**

        - ``stimulus``: list, a list of zeros and ones called stimulus, where each digit corresponds to one second.
        - ``dt``: float, equivalent to the integration step, which defaults to one hundredth of a second.
        - ``N_0``: bool, default is False, when True, the function's output has a basal condition different from zero.
        - ``scale``: bool, determines if the output is scaled, so the output would have a maximum of 1, independent of N_0 and param.
        - ``param``: list, containing the parameters ``[k=1, tau_I=2]`` from equation 14 according to buxton2004. 

    These values can be found in ``buxton2004modeling``.
    
    **Outputs:**
    
        - ``response``: array, neural response 
        - ``time``: array, the time over which the response takes place
    
    '''
    k = 1 # "Inhibitory gain factor" )
    tau_i =  2 # "Inhibitory time constant" )
    n_0   =  0.316 # "Basal Neural Activity")
    
    if params!=None:
        if "k" in params:
            k = params["k"]
        if "tau_i" in params:
            tau_i= params["tau_i"]
        
    stimulus_array= np.asarray(stimulus, dtype=np.float64)
    ext_stim= array_extend(stimulus_array, dt=dt)

    # @njit
    def didt(t,i,s):
    #retornamos lo que sale de ((k*(s-i)-i)/taui)
        return ((k*s-(k+1)*i)/tau_i)

    time=time_segment(ext_stim, dt=dt)
    
    def solver(t,stim):
        I=np.zeros(1,dtype=np.float64) #preallocate the output
        y=np.zeros(1, dtype=np.float64)
        for ts in zip(t,stim):
            y=odeint(didt, y0=I[-1], t=ts[0], args=(ts[1],), tfirst=True)
            I=np.append(I, [y.T[0][1]])
        return I 

    I=solver(time, ext_stim)
    
    if N_0 == False:
        response = ext_stim*(1-I) #Im using ext_stim like  indicator function otherwise we would havew a rebound
    else:
        response = n_0.random() + ext_stim-I
        response[np.where(response<=0.0)]= 0.0

    if scale:
        response = (response-np.min(response))/(np.max(response)-np.min(response))

    return response, np.append(time[:,0], time[-1,1])

def NeurovascularCoupling(stimulus: np.ndarray, version:str= 'differential' ,params=None, dt:np.float64=0.01, mode='full', method='direct', y0=(1,0), AmpI:np.float64=0.2):
    '''
    NeurovascularCoupling
    
    Solves the neurovascular coupling equations diferencital or convlolutional
    
    **Input:**
    
        - ``stimulus``= np.ndarray A list of zeros and ones called 'stimulus', where each digit corresponds to one second.
        - ``version`` = str 'differential' for Maith 2022 or 'convolution' for Buxton 2004.
        - ``dt`` = np.float6 equivalent to the integration time step, which by default is one-hundredth of a second.
        - ``params``: list, 3 parameters in the following order: [tau_h=4, delta t_f=1, f_1=1.009].
    
    - if version == 'differential'
    
            This function calculates the neurovascular coupling according to Stephen 2007 (therefore a modification of Friston 2000) where s is some ﬂow inducing signal deﬁned, operationally, in units corresponding to the rate of change of normalised ﬂow 
        (i.e., s^{-1}) and the stimulus*0.2 corresponds with the Impulse I_{CBF}, and  stimulus*0.05 with I_{CMRO2} 
    
    **Inputs:**
    
        - ``stimulus``= np.ndarray A list of zeros and ones called "stimulus", where each digit corresponds to one second.
        - ``params`` = kappa and gamma acoording to Maith 2022, tau_s and tau_f according to Friston 2000 
        - ``dt`` = np.float64=0.01 equivalent to the integration time step, which by default is one-hundredth of a second.
        - ``y0`` =(1,0) Initual contiditions for f and s
        - ``AmpI``= np.float64, Scales Stimulus to be used for f_{in}(t) [app = 0.2] or m(t)[app = 0.05]

    **Outputs:**
    
        - ``fm``= np.ndarray this could be either m or f depending on AmpI.
        - ``s``= np.ndarray  some ﬂow inducing signal deﬁned, operationally.
    
    - elif version == 'convolution'
    
            Solves the neurovascular coupling equations described for the normalized baseline cerebral blood flow (CBF) 'f_in'. 
        We then assume that both CBF and CMRO2 are linear convolutions of an impulse response function h(t) with the appropriate
        measurement of neuronal activity N(t). According to Buxton 2004. The 'h(t)' shape is then scaled to provide the desired amplitude and duration of the impulse response. For this shape and a 
        desired FWHM of tau_f, the time constant in Eq. (12) is given by the empirical expression tau_h = 0.242 * tau_f.

    **Input:**
    
        - ``params``: list, 3 parameters in the following order: [tau_h=4, delta t_f=1, f_1=1.009].
    
    The values for tau and delta were extracted from Buxton 2004, while the value for f_1=1.009 was obtained heuristically, such 
    that max(f_in(t)) approx 1.5 (note that if f_1=1.00 then f_in(t)=0).
    
    mode and method are parameters inherited from the scipy.signal.convolve function, used for the convolution between N(t) and h(t),
    with options including:
    
        - ``mode``: str {'full', 'valid', 'same'}, optional. A string indicating the size of the output:

            * 'full': The output is the full discrete linear convolution of the inputs. (Default)
            * 'valid': The output consists only of those elements that do not depend on the padding with zeros.
            * 'same': The output is the same size as the first signal, in our case N(t), centered with respect to the 'full' output.
            
        - ``method``: str {'auto', 'direct', 'fft'}, optional. A string indicating which method to use for convolution.

            * 'direct': Convolution is determined directly from sums, the convolution definition.
            * 'fft': Fourier transform is used to perform convolution by calling fftconvolved.
            * 'auto': Automatically chooses between direct or Fourier method based on an estimation of which is faster (default).
    
    **Outputs:**
    
        - ``h(t)``: A plausible form for h(t) is a gamma-variant function with a full-width at half-maximum (FWHM) of approximately 4 s, corresponding to equation 12.
        - ``f_in(t)``: corresponds to equation 13.
        - ``m(t)``: corresponds to equation 13.
    '''
    if version=='convolution':
        
        tau_f = 4 # "Width of CBF impulse response" 
        delta_tf = 1 # "the delay after the start of the stimulus before the CBF response begins"
        scale =  True # "represents the normalized flow increase on the plateau of the CBF response to a sustained neural activity with unit amplitude"
        
        if params!=None:
            if "tau_f" in params:
                tau_f    = params["tau_f"] # "Width of CBF impulse response" 
            if "delta_tf" in params:
                delta_tf = params["delta_tf"] # "the delay after the start of the stimulus before the CBF response begins"
            if "scale" in params:
                scale    = params["scale"] # "represents the normalized flow increase on the plateau of the CBF response to a sustained neural activity with unit amplitude"
            if 'f1' in params:
                f1      = params['f1']   
        
        TOLERANCE = 1.0e-04
         
        tau,delta,amplitude = tau_f, delta_tf, scale
        tau_h = 0.242*tau
        # ``Nt``: array, obtained neuronal response.
        # ``time``: array,internally used to implement a gamma function 
        # using time (plus a delay) as its domain.the time over which this unfolds. 
        Nt, time = neural_response(stimulus, dt, N_0 = False, scale= True, params=None)
        
        def gamma(tau_h,t):    
            k = 3 # "Nameless constant, revisit Buxton 2004 eq 12" )    
            return (1/(tau_h*np.math.factorial(k)))*((t/tau_h)**k) * np.exp(-(t/tau_h))
                
        #h corresponds to h(t) the impulse function gamma
        h = gamma(tau_h,(time-delta))
        #fix for convergence
        h[np.where(h<=TOLERANCE)]= 0.0
        
        NVC = 1+ (f1-1)*convolve(Nt,h, mode=mode, method=method)
        
        return NVC,h

    elif version == 'differential':
        
        k = 1/1.54  ## Maith 2022
        # kappa = 0.8 ## Friston 2000
        g = 1/2.46  ## Maith 2022
        # gamma = 0.4 ## Friston 2000 
        
        if params!=None:
            if "kappa" in params:
                k = params["kappa"]
            if "gamma" in params: 
                g = params["gamma"]
                    
        Nt= array_extend(np.asanyarray(stimulus), dt=dt)*AmpI
        
        def dNC_dt(t, NC, Nt): #NC stands for Neurovascular coupling
            
            s,fm= NC
            
            return [Nt-k*s-g*(fm-1), s]
        
        time=time_segment(Nt, dt=dt)
        s=np.ones(1,dtype=np.float64)*y0[1]
        fm=np.ones(1,dtype=np.float64)*y0[0]
        
        for tfm in zip(time,Nt):

            sol= odeint(dNC_dt, y0=(s[-1], fm[-1]), t=tfm[0],args=(tfm[1],), tfirst=True)
            s=np.append(s,[sol.T[0,1]])
            fm=np.append(fm,[sol.T[1,1]])
        
        return  fm,s


def array_extend(arr: np.ndarray ,dt:np.float64):
    
    new_arr=np.empty(0,np.float64)
    
    for a in arr:
        tmp=np.ones(int(np.ceil(1/dt)), dtype=np.float64)*a
        new_arr=np.append(new_arr,tmp)
    
    return new_arr

def scale_fun(arr:np.ndarray, factor:np.float32):
    '''
    Escala funciones usando la forma [a,b]->[a,d]
    '''
    return np.min(arr)+(factor/(np.max(arr)-np.min(arr)))*(arr-np.min(arr))

# @njit
def Efun(f_in: np.ndarray, E0:float=0.32) -> float:
    '''
    Efun
        
    Resuelve la ecuación para la proporción de oxígeno extraído de la sangre 'E'. 'Efun', definido aquí, proviene de (Friston et al., 2000: Nonlinear Responses in fMRI:...), que a su vez cita a (Buxton et al., 1998)
    
    **Inputs:**
    
        -``f_in``: array, corresponde a la ecuación 13 i.e. el flujo de ingreso.
        -``E0``: float, oxígeno extraído de la sangre, 0.32 por defecto. Valor en Buxton 2004
    
    **Output:**

        -``E``: array, la proporción de oxígeno extraído de la sangre
    
    '''
    E_0 = 0.32 # "baseline value of oxygen extraction fraction"

    if E0!=None:
        E_0 = E0

    try:
        E=1 - (1 - E_0)**(1/f_in);

    except ZeroDivisionError:
        E=1;

    return E;

# @njit
def m_t_E(f_in: np.ndarray, E0:float=0.32):
    # '''
    # m_t_E
    # 
        
    # En estado estable, el CBF y el CMRO_2 están relacionados entre sí por la concentración arterial de oxígeno y la fracción neta de extracción de oxígeno E.
    # La forma de CMRO_2 que se presenta a continuación, corresponde su forma normalizada contra su estado basal; proviene de la ecuación (2) de Buxton 2004, sumando E de Friston 2000.

    # **Inputs:**
    
    #     - ``f_in`` : array, corresponde a la ecuación 13 i.e. el flujo de ingreso.
    #     - ``E0 `` : float, oxígeno extraído de la sangre, 0.32 por defecto. Valor en Buxton 2004
    
    # **Outputs:**

    #     - ``mE`` : array, CMRO_2 normalizada contra su estado basal, en función de la proporción de oxígeno extraído de la sangre 'E'.
        
    # '''

    E_0 = 0.32 # "baseline value of oxygen extraction fraction"

    if E0!=None:
        E_0 = E0

    
    mE= f_in*(Efun(f_in, E_0)/E_0)

    return mE

# @njit
def vol_func(f_in: np.ndarray, params: np.ndarray, vol0:float=1, dt:float= 0.01, viscoelastic:bool=False):
    '''
    vol_func
    
    vol_fun entrega la   solución de la ecuación diferencial para el volumen, según una combinación las ecuaciones 10 y 11 del articulo de Buxton 2004. 
    
    **Inputs:**
    
        - ``f_in`` :  array, corresponde a la ecuación 13 i.e. el flujo de ingreso.
        - ``params`` : list, incluyen las constantes de las ecuaciones 10 y 11, estas son son [tau_MTT = 3,0, alpha = 0,4, tau_m ∈ [0, 30]].
        - ``dt`` : float, dt refiere al paso de integración.
        - ``viscoelastic`` :bool  determina si la salida contempla o no el efecto viscoelástico, i.e. tau = 0  ́o 0 < tau ≤ 30.

    **Outputs:**
     
        - ``v(t)`` : array, serie de tiempo del volumen
        - ``time`` : array, serrie con el tiempo en el cual transcurre v(t).
    
    '''

    tau_MTT = 3.0 # venous time constant
    alpha   = 0.4 # Grubb's exponent (stiffness)
    tau_m   = 10  # Viscoelastic time constant (deflation)
    tau_p   = 15  # Viscoelastic time constant (inflation)

    if params!=None:
        if "tau_MTT" in params:
            tau_MTT = params["tau_MTT"]
        if "alpha" in params:
            alpha   = params["alpha"]
        if "tau_m" in params:    
            tau_m   = params["tau_m"]   
    
    if viscoelastic:
        taum = tau_m
    else:
        taum= 0

    tauMTT,a=tau_MTT, alpha

    def dvdt(t ,v ,f ) -> float:
        return ((f - v**(1/a))/(tauMTT+taum))

    v=np.empty(0, dtype=np.float64) #preallocate the output
    time=time_segment(f_in, dt=dt)

    for ts in zip(time,f_in):
        if 0 in ts[0]:
            tmp = vol0
        else: tmp = v[-1]

        y=odeint(dvdt, y0=tmp, t=ts[0], args=(ts[1],), tfirst= True)
        v=np.append(v, [y.T[0][1]])

    time=time[:,0]

    return [v, time]

# @njit
def f_out(vol: np.ndarray, f_in: np.ndarray, viscoelastic:bool=False, params=None):
    '''
    f_out

    Fundamental para el modelo balloon, el flujo de salida se obtiene a partir de nuestra función 'f_out', resolviendo la tasa de cambio del volumen en la ecuación de flujo de salida, es decir, en la ecuación 11 del artículo de Buxton 2004.

    **Inputs:**
        - ``vol`` : array, serie de tiempo del volumen.
        - ``f_in`` :  array, corresponde a la ecuación 13 i.e. el flujo de ingreso.
        - ``viscoelastic`` : bool, determina si la salida contempla o no el efecto viscoelástico, es decir, tau = 0 o 0 < tau ≤ 30.
        - ``params`` : list, incluye las constantes de las ecuaciones 10 y 11, que son [tau_MTT = 3.0, alpha = 0.4, tau_m ∈ [0, 30]].

    **Outputs:**
        - ``fout``: array, serie de tiempo del flujo de salida.

    '''
    
    tau_MTT = 3.0   # "venous time constant"
    alpha   = 0.4   # "Grubb's exponent (stiffness)"
    tau_m   = 10    # "Viscoelastic time constant (deflation)"
    tau_p   = 15    # "Viscoelastic time constant (inflation)"

    if params!=None:
        if "tau_MTT" in params:
            tau_MTT = params["tau_MTT"]
        if "alpha" in params:
            alpha   = params["alpha"]
        if "tau_m" in params:    
            tau_m   = params["tau_m"]   

    if viscoelastic:
        taum = tau_m
    else:
        taum= 0

    tauMTT,a=tau_MTT, alpha

    fout= ((tauMTT*vol**(1/a)) + taum * f_in)/(tauMTT+ taum)
    fout[fout<0.0]=0

    return fout

# @njit
def time_segment(time: np.ndarray, dt:np.float64=0.01):
    '''
    time_segment
    
    Esta función produce intervalos de tiempo de longitud dt que segmentan un rango de tiempo similar a 'time'.
    
    **Inputs:**
    
        - ``time``: array o lista, cuya longitud se usa para ser duplicada.
        - ``dt``: float, paso de integración usado como longitud de los intervalos.
    
    **Output:**
    
        - ``new_time``: array, tal que 'array.shape = (-1,2)' con -1 intervalos de tiempo de longitud dt.

    '''

    newt=np.arange(start=0,stop=len(time)*dt, step=dt, dtype=np.float64)

    #segments the nwet[a,b] in n intervals [a_i,b_i]
    #whith a_0 = 0 and b_n = newt[-1]
    if len(newt)%2==0: #if len(newt) is even, no problem
        tmp = [[newt[i],newt[i+1]] for i in range(len(newt)-1)]
    else: #if len(newt) is odd, just add an item
        newt= np.append(newt,newt[-1]+dt)
        tmp = [[newt[i],newt[i+1]] for i in range(len(newt)-1)]

    #turns tmp into an array cause <3 (heart) numpy
    new_time=np.array(tmp)

    return new_time


# @njit
def q_func(vol: np.ndarray, mt: np.ndarray, f_out: np.ndarray, params, dt:float= 0.01):
    '''
    q_fun
    ========

    q_fun entrega la solución de la ecuación diferencial para el contenido de deoxyhemoglobina, según una combinación las ecuaciones 10 y 11 del articulo de Buxton 2004. 
    
    **Inputs:**
    
        - ``vol`` : array, serie de tiempo del volumen de acuerdo a la ecuación 10 de buxton 2004
        - ``mt`` : array, serie de tiempo de la tasa metabólica cerebral de oxígeno (CMRO_2) normalizado a su valor en reposo.
        - ``f_out`` : array, serie de tiempo con el flujo de salida, de acuerdo con la función 11 de buxton 2004.
        - ``params`` : list, incluyen las constantes de las ecuaciones 10 y 11, estas son son [tau_MTT = 3,0, alpha = 0,4, tau_m ∈ [0, 30]].
        - ``dt`` : float, dt refiere al paso de integración.
        - ``viscoelastic`` : bool  determina si la salida contempla o no el efecto viscoelástico, i.e. tau = 0  ́o 0 < tau ≤ 30.

    **Outputs:** 
    
        - ``q`` : array, serie de tiempo con el contenido de deoxyhemoblobina
        - ``time`` : array, serrie con el tiempo en el cual transcurre q(t).

    '''
    
    tau_MTT =     3.0 # "venous time constant")

    if params!=None:
        if "tau_MTT" in params:
            tau_MTT = params["tau_MTT"]

    tauMTT=tau_MTT

    def dqdt(t ,q , V= vol, CMRO=mt, fout= f_out) -> float:
        return ((CMRO - (q/V)*fout)/tauMTT)

    q=np.ones(1, dtype=np.float64) #preallocate the output
    time=time_segment(vol, dt=dt)

    for ts in zip(time,vol, mt, f_out):
        y=odeint(dqdt, y0=q[-1], t=ts[0], args=(ts[1],ts[2],ts[3],), tfirst= True)
        q=np.append(q, [y.T[0][1]])

    time=np.append(time[:,0], time[-1,1])
    
    return q,time


def Balloon_odeint(f_in: np.ndarray, mt: np.ndarray, params=None, dt:float=0.01, y0=(1,1), viscoelastic=False):
    '''
    Balloon_odeint
    
    Balloon_odeint resuelve el sistema de ecuaciones para el modelo balloon (ecuaciones 10 y 11 del articulo de Buxton 2004.)
    
    **Inputs:**
        - ``f_in`` :  array, corresponde a la ecuación 13 i.e. el flujo de ingreso.
        - ``mt`` : array, serie de tiempo de la tasa metabólica cerebral de oxígeno (CMRO_2) normalizado a su valor en reposo.
        - ``params`` : list, incluyen las constantes de las ecuaciones 10 y 11, estas son son [tau_MTT = 3,0, alpha = 0,4, tau_m ∈ [0, 30]].
        - ``dt`` : float, dt refiere al paso de integración.
        - ``y0`` : tuple, coordenanas iniciales, tanto de vt, como de qt.
        - ``viscoelastic`` : bool  determina si la salida contempla o no el efecto viscoelástico, i.e. tau = 0  ́o 0 < tau ≤ 30.
        
    **Outputs:**
        - ``v```:array, serie de tiempo con el volumen sanguineo 
        - ``q``: array, serie de tiempo con el contenido de deoxyhemoblobina
    
    '''  
    tau_MTT =    3.0 # "venous time constant"
    alpha =      0.4 # "Grubb's exponent (stiffness)"
    tau_m =      10  # "Viscoelastic time constant (deflation)"
    
    if params!=None:
        if "tau_MTT" in params:
            tau_MTT = params["tau_MTT"]
        if "alpha" in params:
            alpha   = params["alpha"]
        if "tau_m" in params:    
            tau_m   = params["tau_m"]   

    if viscoelastic:
        taum = tau_m
    else:
        taum= 0
    
    tauMTT,a=tau_MTT, alpha
    # @njit
    def dB_dt(t,B,f, m):

        v,q= B

        foutF= lambda v,f:((tauMTT*v**(1/a))+(taum*f))/(tauMTT+taum)

        return [(f - v**(1/a))/(tauMTT+taum),
                ((m - (q/v)*foutF(v,f))/tauMTT)]

    time=time_segment(f_in, dt=dt)
    v=np.ones(1,dtype=np.float64)*y0[0]
    q=np.ones(1,dtype=np.float64)*y0[1]
    
    for tfm in zip(time,f_in,mt):
        sol=odeint(dB_dt, y0=(v[-1], q[-1]), t=tfm[0], args=(tfm[1],tfm[2],) , tfirst=True)
        v=np.append(v,sol[1][0])
        q=np.append(q,sol[1][1])
   
    return v,q
    
def Balloon_ivp(f: np.ndarray, m: np.ndarray, params=None, y0=(1,1), viscoelastic:bool=False, method:str="DOP853"):
    '''
    Balloon_ivp
    
    Balloon_ivp resuelve el sistema de ecuaciones para el modelo balloon (ecuaciones 10 y 11 del articulo de Buxton 2004.)
    
    **Inputs:**
        - ``f_in`` :  array, corresponde a la ecuación 13 i.e. el flujo de ingreso.
        - ``mt`` : array, serie de tiempo de la tasa metabólica cerebral de oxígeno (CMRO_2) normalizado a su valor en reposo.
        - ``params`` : list, incluyen las constantes de las ecuaciones 10 y 11, estas son son [tau_MTT = 3,0, alpha = 0,4, tau_m ∈ [0, 30]].
        - ``dt`` : float, dt refiere al paso de integración.
        - ``y0`` : tuple, coordenanas iniciales, tanto de vt, como de qt.
        - ``viscoelastic`` : bool  determina si la salida contempla o no el efecto viscoelástico, i.e. tau = 0  ́o 0 < tau ≤ 30.
        
    **Outputs:**
        - ``v```:array, serie de tiempo con el volumen sanguineo 
        - ``q``: array, serie de tiempo con el contenido de deoxyhemoblobina
    
    '''  
      
    if params!=None:
        if "tau_MTT" in params:
            tau_MTT = params["tau_MTT"]
        if "alpha" in params:
            alpha   = params["alpha"]
        if "tau_m" in params:    
            tau_m   = params["tau_m"]
    
    if viscoelastic:
        taum = tau_m
    else:
        taum= 0

    tauMTT,a=tau_MTT, alpha
    
    time=time_segment(f, dt=0.01)
    
    # @njit
    def dB_dt(t,B,f, m):
        v,q= B

        foutF= lambda v,f:((tauMTT*v**(1/a))+(taum*f))/(tauMTT+taum)

        return [(f - v**(1/a))/(tauMTT+taum),
                ((m - (q/v)*foutF(v,f))/tauMTT)]
    
    
    def solver(t,f,m,y0):
        v=np.ones(1,dtype=np.float64)*y0[0]
        q=np.ones(1,dtype=np.float64)*y0[1]
        
        for tfm in zip(t,f,m):

            sol_ivp= solve_ivp(dB_dt, t_span=tfm[0], y0=(v[-1], q[-1]), method=method, t_eval=[tfm[0][1]],
                            dense_output=False, vectorized=False, rtol=1e-9, atol=1e-9, args=(tfm[1],tfm[2],))
            v=np.append(v,sol_ivp.y[0])
            q=np.append(q,sol_ivp.y[1])
        
        return v,q
    
    v,q = solver(time,f,m,y0) 
    return(v,q)


# @njit
def cartesian(arrays, out=None):
    '''
    Cartesian
        
    Generea el producto cartesiano entre los arrays ingresados.
    
    **Inputs:**
        - ``arrays`` : tupla de array-like, 1-D arrays entre los cuales hacer el producto cartesiano
    

    **Output:**
        - ``out`` : ndarray, 2-D array con formato/forma (M, len(arrays)) contenedor del producto cartesiano formado a partir de los inputs.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
        [1, 4, 7],
        [1, 5, 6],
        [1, 5, 7],
        [2, 4, 6],
        [2, 4, 7],
        [2, 5, 6],
        [2, 5, 7],
        [3, 4, 6],
        [3, 4, 7],
        [3, 5, 6],
        [3, 5, 7]])

    '''
    mesh = np.meshgrid(*arrays)
    # Concatenar los resultados en una matriz de 
    result = np.concatenate([x.reshape(-1,1) for x in mesh], axis=1)
    return result

def BOLD_func(vt: np.ndarray,qt: np.ndarray, params=None, BM:str='classic'):
    '''
    BOLD_func
    
    Calcula la senal dependiente de oxigeno en la sangre (BOLD)
    usando valores de volumen (vt) y deoxihemoglobina (qt) de acuerdo con 
    las estimaciones de Obata(2004) y Buxton(2000) expuestas en Stephen(2007)

    **Inputs:**
    
        - ``vt`` : np.ndarray, a 1D array of volume (in arbitrary units) over time
        - ``qt`` : np.ndarray, a 1D array of cerebral deoxyhemoglibine (in arbitrary units) over time
        - ``params`` : dict, o None, a list model parameters (optional, defaults to None) E_0, V_0, v_0, TE, epsilon and r_0
        - ``BM`` : str, kind of Balloon Model (optional, defaults is "classic", while the alternative is "revised") accordin Stephan 2007
    
    **Outputs:**
    
        - ``bold`` : np.ndarray, a 1D array of simulated BOLD signals over time
            
    '''
    
    E_0 = 0.32  # "Baseline value of oxygen extraction fraction")
    V_0 = 0.03  # "Baseline blood volume")
    TE  = 0.04  # "Echo time in miliseconds")
    eps = 1.43  # "Ratio of intra- to extravascular BOLD signal at rest")
    r_0 = 25    # "The slope of the relation between the intravascular relaxation rate and oxygen saturation. For a field strength of 1.5[T], r0=25 s^{-1}")
    O_o = 40.3  # "The frequency offset at the outer surface of the magnetized vessel for fully deoxygenated blood. For a field strength of 1.5[T], v0=40.3 s^{-1}")

    
    if params!=None:
        if "E_0" in params:
            E_0 = params['E_0']
        if "V_0" in params: 
            V_0   = params['V_0']
        if "TE" in params: 
            TE   = params['TE'] 
        if "O_0" in params: 
            O_o   = params['O_0'] 
        if "r_0" in params: 
            r_0   = params['r_0']
        if "epsilon" in params: 
            eps   = params['epsilon']  

    if BM == "classic":
        # Classic coefficients
        k_1 = (1-V_0)*4.3*O_o*E_0*TE
        k_2 = 2*E_0
    elif BM == "revised":
        # Revised coefficients
        k_1 = 4.3*O_o*E_0*TE
        k_2 = eps*r_0*E_0*TE
    else: print("So far we only have known 2 kinds of Balloons, as sujested by Stephan 2007")
    
    k_3 = 1.0 - eps

    return V_0 * (k_1*(1.0-qt) + k_2*(1.0 - qt/vt) + k_3*(1.0 - vt))

# @njit
def BOLD_Davis(f: np.ndarray, m: np.ndarray, author:str='Davis198'):
    if author=='Davis1998':
        A       = 0.075 #   "Amplitud constant"
        alpha   = 0.4   #   "TODO"
        beta    = 1.5   #   "TODO"

    elif author=='Maith2022':
        A       = 140.9 #   "Amplitud constant"
        alpha   = 0.14  #   "TODO"
        beta    = 0.91  #   "TODO"

    else: print('So far we only have known 2 authors, Davis1998, and Maith2022')

    A,a,b=A, alpha, beta
    bold = A*(1-f**(a-b)*m**b)   
    
    return bold

# %%
