import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift, rfft, rfftfreq
from scipy import fftpack
import matplotlib.patches as patches


def D(h, f, f_p):
    return (4 * 299792458 * 5 * f_p) / (abs(h) * 96 * np.power(np.pi,2) * np.power(f, 3))


Cdata = pd.read_csv("/Users/James/Downloads/C_spectrum.csv")
Hdata = pd.read_csv("/Users/James/Downloads/gw_data.txt", sep= " ")
wl = 5896
c = 2.998e8

wavel = Cdata['#Wavelength']
flux = Cdata['Flux']




#Simple loop that goes through values to find the min peak next to the actual value for Sodium D abs
obs_wl = 7000
count = 0
y = 8000
i = 0
while i < len(wavel):
    if wavel[i] > wl and wavel[i] < 6200:
        if flux[i] < y:
            y = flux[i]
            obs_wl = wavel[i]
            count = i

    i+=1



'''plt.plot(wavel,flux)
plt.axvline(x = 5896, color = 'r', linestyle = ':')
plt.text(5896,2000,'Sodium D absortion ',rotation = -90, fontsize = 8)
plt.scatter(obs_wl, flux[count] , s=40, facecolors='none', edgecolors='r')
plt.text(obs_wl+100, flux[count]-1000,'Observed Wavelength ',rotation = -90, fontsize = 8)
plt.title("Spectum of Rest Frame Spectral Wavelength")
plt.xlabel("Wavelength (Angstrom)")
plt.ylabel(" Flux (Counts) ")
plt.show()
'''

#obs_wl = 5988.8

wl_std = wavel.sem()

v = c*(obs_wl - wl)/wl
print(obs_wl, '+/-', wl_std)

time = Hdata['#t'].to_numpy()
h = Hdata['h'].to_numpy()


#Graph of Quasi Nature
'''fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Quasi-Periodic Nature of the Graph h(t)')
ax1.plot(time, h)
ax2.plot(time,h)
ax2.set_xlim(-10,-9.8)
ax2.set_ylim(-1e-22,1e-22)
ax2.set(xlabel = 'time (s)')
ax2.set(ylabel = 'h')
ax1.set(ylabel = 'h')

ax1.add_patch(
    patches.Rectangle(
        (-10, -1e-22),
        1,
        2e-22,
        edgecolor = 'blue',
        facecolor = 'red',
        fill=False
    ) )

plt.show()
'''

'''fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Quasi-Periodic Nature of the Graph h(t)')
ax1.plot(time, h)
ax2.plot(time,h)
ax1.set_xlim(-30,-29.9)
ax2.set_xlim(-10,-9.9)
plt.show()
'''

#find interval of 1s
'''i = 0
count = 0
while i < len(time):
    if time[i] <= time[0]+1:
        count += 1
    i+=1
'''







Sample_Rate = 4096
Duration = .07
print(len(time))

N = int(Sample_Rate*Duration)

f = np.zeros(len(time))

FreqTime = np.empty((0,2), float)

'''FreqTime = np.append(FreqTime, np.array([[time[5000],h[5000]]]), axis = 0 )
FreqTime = np.append(FreqTime, np.array([[time[6000],h[6000]]]), axis = 0 )'''



#Loop that calcs rfft of h, then
'''j = 0
spot = 0
while spot < len(time):
    u = int(spot - N/2)
    l = int(spot + N/2)
    if u > 0 and l < len(time):
        yf = rfft(h[u:l])
        xf = rfftfreq(N, 1 / Sample_Rate)
        max = np.where(np.abs(yf) == np.max(np.abs(yf)))
        if xf[max]  not in  f:
            f[spot] = xf[max]
            FreqTime = np.append( FreqTime, np.array([[time[spot],f[spot]]]), axis = 0 )
            print(spot)
        else:
            f[spot] = xf[max]
    else:
        f[spot] = 0
    spot+=1
#np.savetxt("frequencies.csv", f, delimiter = ",")
print(FreqTime)
np.savetxt("Freq_time", FreqTime, delimiter = ",")'''


TimeFreq = pd.read_csv("/Users/James/PycharmProjects/HubbleConst_Re/Freq_time")
t = TimeFreq['t']
f = TimeFreq['f']

var_t = 2.5e-4
sum = 0
i = 0
while i < N-1:
    sum = sum + var_t**2
    i+=1
print('sum',sum)
'''
plt.scatter(t,f)
plt.title("Scatter plot of f(t)")
plt.xlabel('Time (s)')
plt.ylabel('Freq (1/s)')
plt.show()
'''


t = TimeFreq['t'].to_numpy()
f = TimeFreq['f'].to_numpy()



d = np.zeros(len(t))


h_abs = np.zeros(len(t))

result = np.where(time == t[2])

i = 0
while i < len(t):
    result = np.where(time == t[i])[0].tolist()
    result = [int(x) for x in result]
    result = int(result[0])
    h0 = h[result-100:result+100]
    h_abs[i] = np.max(h0)- np.min(h0)
    i+=1




i = 0
distance = np.zeros(len(t))
while i < len(t):
    fp = 0
    if i+1 < len(t):
        fp = (f[i+1]-f[i])/(t[i+1]-t[i])
        distance[i] = D(h_abs[i],f[i],fp)
    i+=1


distance = np.delete(distance, 25,axis = 0)

print(len(distance))


d_avg = distance.mean()
d_stdm = np.std(distance)
hubble = v/d_avg*3.086e19
print('H:', hubble)

print('vel:', v)
print('distance:', d_avg, '+/-', d_stdm/np.sqrt(25) )

'''#local derivative
freq_derivatives = np.zeros(len(time))

i = 0
while i < len(time):
    if i+1 == len(time):
        break
    der = (freq[i+1] - freq[i]) / (time[i+1] - time[i])
    freq_derivatives[i] = der
    i+=1
'''

'''Distance = np.zeros(len(time))
i=0
while i < len(time):
    if freq[i] == 0:
        Distance[i] = 0
    if freq_derivatives[i] == 0:
        Distance[i] = 0
    else:
        Distance[i] = D(h[i],freq[i],freq_derivatives[i])
        print(i)
    i+=1
'''


















#A try of Fourier
'''time = time[1:].to_numpy()
signal = Hdata['h'].to_numpy()
fourier = fft(signal)
N = 50000
T = 1.0 / 800.0
xf = fftfreq(N,T)[:N//2]
print(xf)
plt.plot(xf, 2.0/N * np.abs(fourier[0:N//2]))
plt.grid()
plt.show()
'''