import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal



def plot_harmonics(K):

    t = np.arange(0,1,0.001)
    N_plots = 2
    fig, ax = plt.subplots(N_plots, sharex=True, figsize=(12,8))

    for k in range(1, K):
        ax[0].plot(t, np.cos( 2 * np.pi * k * t), label=f'$cos( 2 \pi ({k}) t)$')
        ax[1].plot(t, np.sin( 2 * np.pi * k * t), label=f'$sin( 2 \pi ({k}) t)$')
    ax[0].legend()
    ax[1].legend()
    ax[0].grid(linestyle=':')
    ax[1].grid(linestyle=':')
    plt.savefig(f'figs/harmonics.pdf' )


def fourier_series_real(X, t):
    K = len(X)
    x_hat = X[0] * np.ones(len(t))
    for k in range(1, K):
        # cosine_k = np.cos( 2 * np.pi * k * t)
        # x_hat += 2 * X[k] * cosine_k
        exp_k = np.exp( 2j * np.pi * k * t)
        x_hat += 2 * (X[k] * exp_k).real
    return x_hat

def plot_FS_approximations(t, X, K, case_name, x_true=None):

    N_plots = len(K)
    fig, ax = plt.subplots(N_plots, sharex=True, figsize=(12,8))

    for m, k  in enumerate(K):
        x_hat = fourier_series_real(X[:k+1], t)
        # print(m, k, X[:m+1])
        if x_true is not None:
            ax[m].plot(t, x_true, c='k', linestyle='--', label=f'signal')
        ax[m].plot(t, x_hat, c = 'b', label=f'FS with K = {k}')
        ax[m].legend()
        ax[m].grid(linestyle=':')
    plt.savefig(f'FS_approx_{case_name}.pdf' )


#### square wave
t = np.arange(-2, 2, 0.0001)

x_true = 0.5 * np.sign(np.cos(2 * np.pi * t)) + 0.5
K = [1, 3, 5, 7, 9]
### X_k = 0.5 sinc(k/2)
X = np.asarray( 0.5 * np.sinc(  np.arange(20) / 2 ))
plot_FS_approximations(t, X, K, 'lifted_square_wave', x_true)
