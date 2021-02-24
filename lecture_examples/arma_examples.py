import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pandas as pd
import os
import soundfile as sf


#####################################################################################
### exploring COVID data and filtering -- ROUTINES
#####################################################################################
def filter_and_plot(b, a, x):
	y = signal.lfilter(b, a, x)
	plt.figure()
	plt.plot(x, color = 'tab:gray', label='input') # linewidth=1,
	plt.plot(y, color = 'r', label='output')
	plt.legend()
	plt.grid(linestyle=':')
	plt.xlabel("time (days)")
	plt.ylabel("signals")
	# plt.xlim([0, 100])
	#plt.ylim([-20, 2])
	plt.show()
	#plt.savefig('toy.png', dpi=256)

	return y

def read_covid_data(refresh=False):
    fname = 'data/91-DIVOC-countries.csv'
    data_url = 'http://91-divoc.com/pages/covid-visualization/?chart=countries&highlight=United%20States&show=25&y=both&scale=linear&data=cases-daily&data-source=jhu&xaxis=right#countries'
    file_downloaded = os.path.isfile(fname)
    if not file_downloaded or refresh:
        ## this does not work because of indirect linking of the CSV data on the page
        data = pd.read_csv(data_url, index_col=0, header=None).fillna(value = 0).T
    else:
        data = pd.read_csv(fname, index_col=0, header=None).fillna(value = 0).T
    
    us_data = data['United States']

    return np.asarray(us_data, dtype=np.int)

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


def plot_freq_respone(b, a, x_low=0, x_high=0.5, y_low=-60, y_high=0, freq_tag='', freq_scale=1):
    ## this computes the arma filter's frequency response from the diff. eq. coefficents
    w, H = signal.freqz(b, a, worN=2**16)
    
    plt.figure()
    nu =  ( w / ( 2 * np.pi ) ) 
    plt.plot(nu * freq_scale, 20 * np.log10(abs(H)), label='|H| (dB)')
    plt.legend()
    plt.grid(linestyle=':')
    xlims = np.asarray([x_low, x_high]) * freq_scale
    plt.xlim()
    plt.ylim([y_low, y_high])
    plt.xlabel(f'frequency {freq_tag}')
    plt.ylabel('filter gain (dB)')


#####################################################################################
### exploring COVID data and filtering -- EXAMPLES
#####################################################################################
x_cov = read_covid_data()

## plot the DTFT of the data - this is the signal's frequency content
nu, X_cov = direct_dtft(x_cov)
X_cov_mag_dB = 20 * np.log10(X_cov)
plt.figure()
plt.plot(nu, X_cov_mag_dB, color = 'b', label='covid data (freq)') # linewidth=1,
plt.axvline(x=1/7, c='r')
plt.legend()
plt.grid(linestyle=':')
plt.xlabel("frequency (cycles/day)")
plt.ylabel('signal freq. content (dB)')


### seven day averaging
b7 = np.ones(7) / 7
a7 = np.asarray([1])

y7 = filter_and_plot(b7, a7, x_cov)

plot_freq_respone(b7, a7, freq_tag='(cycles/day')


### butterworth filter
nu_0 = 0.5 * (1 / 7)
filter_order = 4
# b_butter, a_butter = signal.butter(2, 2 * nu_0, btype='lowpass')
b_butter, a_butter = signal.butter(filter_order, 2 * nu_0)

y_butter = filter_and_plot(b_butter, a_butter, x_cov)

plot_freq_respone(b_butter, a_butter, freq_tag='(cycles/day')

## compare the 2....

plt.figure()
plt.plot(y7, color = 'b', label='7 day average') # linewidth=1,
plt.plot(y_butter, color = 'r', label='butterworth')
plt.legend()
plt.grid(linestyle=':')
# plt.xlim([250, 330])
plt.xlabel("time (days)")
plt.ylabel("signals")


#####################################################################################
### toy filtering example
#####################################################################################
### some toy sythetic data (noisy sine wave)
nu0 = 0.025
n = np.arange(0, int(3/nu0))
x = np.sin(2 * np.pi * nu0 * n) + np.random.normal(0, 0.5, len(n))
b, a = signal.butter(3, 2 * (1.5 * nu0) )
y = filter_and_plot(b, a, x)


#####################################################################################
### exploring DT frequency via plots
#####################################################################################
n = np.arange(20)
t = np.arange(0, 20, 0.001)
nu = [ 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.9 ] #np.arange(0, 2, 0.2)
for nu0 in nu:
    plt.figure()
    plt.stem(n, np.cos(2 * np.pi * nu0 * n), label=f'DT: nu = {nu0}')
    plt.plot(t, np.cos(2 * np.pi * nu0 * t), c='g', linewidth=0.75, label='cont. time')
    plt.legend()

#####################################################################################
### audio example
#####################################################################################
def filter_and_plot_audio(f_cut_hz, filter_order, fname):
    ## read the wav file
    x, f_sample = sf.read(fname)

    nu_0 = f_cut_hz / f_sample
    b, a = signal.butter(filter_order, 2 * nu_0)
    plot_freq_respone(b, a, freq_tag='(kHz)', freq_scale=f_sample/1000)

    y = signal.lfilter(b, a, x)
    N_samples = len(x)
    t = np.arange(N_samples) / f_sample
    plt.figure()
    plt.plot(t, x, color = 'tab:gray', label='input') # linewidth=1,
    plt.plot(t, y, color = 'r', label='output')
    plt.legend()
    plt.grid(linestyle=':')
    plt.xlabel("time (sec)")
    plt.ylabel("signals")
    # plt.xlim([0, 100])
    plt.ylim([-1, 1])
    plt.show()
    #plt.savefig('toy.png', dpi=256)
    
    return y, f_sample


f_cut = 4000
y_chirp, f_sample = filter_and_plot_audio(f_cut, 4, 'data/chirp.wav')
sf.write(f'data/chirp_{f_cut}.wav', y_chirp, f_sample)

f_cut = 2000
y_chirp, f_sample = filter_and_plot_audio(f_cut, 4, 'data/chugg_welcome.wav')
sf.write(f'data/chugg_welcome_{f_cut}.wav', y_chirp, f_sample)