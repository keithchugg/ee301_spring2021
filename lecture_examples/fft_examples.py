import numpy as np
import matplotlib.pyplot as plt

def direct_dtft(x):
    ## range of normalized frequency -- nu  in [0, 0.5]
    nu = np.arange(0, 0.5, 0.001)
    N_nu_points = len(nu)

    ### the DTFT of x
    X = np.zeros(N_nu_points, dtype=np.complex)

    ## the integer time n
    n = np.arange(len(x))

    for i in range(N_nu_points):
        dtft_exp_nu = np.exp( -2.j * np.pi * nu[i] * n)
        X[i] = np.dot(x, dtft_exp_nu)
    
    return nu, X

def plot_DFT_DTFT(x, use_db=False, use_direct_dtft=False):
    N = len(x)
    X = np.fft.fft(x)
    Xr = np.fft.rfft(x)

    if use_direct_dtft:
        nu, X_dtft = direct_dtft(x)
    else:
        N_fft = 2 ** 16
        X_dtft = np.fft.rfft(x, N_fft)
        nu = np.arange(len(X_dtft)) / N_fft
    nu_k = np.arange(len(Xr)) / N

    fig, ax = plt.subplots(1, 3, figsize=(20, 4))
    ax[0].stem(x, label='x[n]')
    ax[0].set_xlabel('n')
    ax[0].legend()

    ax[1].stem(np.abs(X), label=f'DFT N = {N}')
    ax[1].set_xlabel('n')
    ax[1].legend()

    mag_Xr = np.abs(Xr)
    mag_X_dtft = np.abs(X_dtft)
    if use_db:
        db_tag = 'dB'
        mag_Xr = 20 * np.log10(mag_Xr)
        mag_X_dtft = 20 * np.log10(mag_X_dtft)
    else:
        db_tag = ''
    
    ax[2].scatter(nu_k, mag_Xr, color='k', label=f'DFT N = {N}')
    ax[2].plot(nu, mag_X_dtft, color='g', label='DTFT')
    ax[2].legend()
    ax[2].set_xlabel(r'$\nu$')
    ax[2].set_ylabel(fr'$|X|$')

def plot_DFT_DTFT_AR1(alpha, N):
    n = np.arange(N)
    x = alpha ** n
    X = np.fft.fft(x)
    Xr = np.fft.rfft(x)

    fig, ax = plt.subplots(1, 3, figsize=(20, 4))
    ax[0].stem(x, label='x[n]')
    ax[0].set_xlabel('n')
    ax[0].legend()

    ax[1].stem(np.abs(X), label=f'DFT N = {N}')
    ax[1].set_xlabel('n')
    ax[1].legend()

    mag_Xr = np.abs(Xr)
    nu = np.arange(0, 0.5, 0.001)
    X_dtft = 1 / (1 - alpha * np.exp(-2j * np.pi * nu))
    mag_X_dtft = np.abs(X_dtft)
    nu_k = np.arange(len(Xr)) / N

    ax[2].scatter(nu_k, mag_Xr, color='k', label=f'DFT N = {N}')
    ax[2].plot(nu, mag_X_dtft, color='g', label=fr'DTFT; $\alpha = $ {alpha}')
    ax[2].legend()
    ax[2].set_xlabel(r'$\nu$')
    ax[2].set_ylabel(fr'$|X|$')



def cosine_example(N, nu_0):
    n = np.arange(N)
    x = np.cos(2 * np.pi * nu_0 * n)
    plot_DFT_DTFT(x)

def drect_example(N):
    n = np.arange(N)
    x = (n <= N/2).astype(np.float)
    plot_DFT_DTFT(x)

def hanning_example(N, use_db=True):
    n = np.arange(N)
    x = np.hanning(N)
    plot_DFT_DTFT(x, use_db)

def AR1_example(N, alpha):
    n = np.arange(N)
    x = alpha ** n
    plot_DFT_DTFT(x)


################################
#####  Simple Example of Cosine
################################

## small N, cos with freq = k/N
cosine_example(4, 1/4)

## small N, cos with freq != k/N
cosine_example(4, 1/3)

## larger N, cos with freq = k/N
cosine_example(128, 1/4)

## larger N, cos with freq != k/N
cosine_example(128, 1/3)

################################
#####  Drect example
################################

for N in [4, 8, 128]:
    drect_example(N)

################################
#####  Hanning Window example
################################

for N in [4, 8, 128]:
    hanning_example(N)

################################
#####  truncated AR1 response
################################
for N in [4, 8, 16]:
    plot_DFT_DTFT_AR1(0.5, N)

for N in [4, 8, 16]:
    plot_DFT_DTFT_AR1(0.95, N)


##############################################
#####  DTFT from diric-interpolted DFT example
##############################################

## this is explained in my (advanced) Transform Theory Notes
## it comes from windowning the periodic signal IDFT{X[k]} by a rect
## this means convolution in frequency, which yields the diric (dinc) interoloation

def diric(v, L):
    return np.sinc(L * v)/np.sinc(v)

N = 8
nu_0 = 1/3
cosine_example(N, nu_0)

### show how we can interpolate DTFT from the DFT
x = np.cos(2 * np.pi * nu_0 * np.arange(N))
nu = np.arange(0, 1, 0.001)
X = np.fft.fft(x)
nu_samples = np.arange(N) / N

X_dtft_hat = np.zeros(len(nu))
diric_terms = np.zeros((N, len(nu)))
for k in range(N):
    diric_terms[k] = ( X[k] * diric(nu - k/N, N) ).real
    X_dtft_hat += diric_terms[k]

plt.figure(figsize=(12, 8))
for k in range(N):
    plt.plot(nu, diric_terms[k].real, linestyle='--')
plt.stem(nu_samples, X.real)
plt.plot(nu, X_dtft_hat.real, color='k')
plt.xlabel(r'$\nu$')
plt.ylabel('diric interpolation terms')