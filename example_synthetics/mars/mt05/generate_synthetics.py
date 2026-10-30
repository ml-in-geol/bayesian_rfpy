import instaseis
import obspy
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.taup import TauPyModel
import pyproj

def add_noise(st,snr,gcarc,freqmin,freqmax):
    '''
    adds band limited white noise to synthetics

    args---
    st: obspy Stream with clean synthetic traces
    snr: desired signal to noise ratio (signal level is based on P-wave amplitude)
    gcarc: epicentral distance (needed to find phase amplitudes)
    freqmin: minimum frequency of noise
    freqmax: maximum frequency of noise

    returns:
    st_noisy: obspy Stream with noise added
    '''
    stc = st.copy()
    mod = TauPyModel('insight_kks21_gp')
    arrs = mod.get_travel_times(source_depth_in_km = 20., distance_in_degree = gcarc, phase_list = ['P'])
    Ptime = arrs[0].time

    start=stc[0].stats.starttime
    trz = stc.copy().trim(starttime=start+Ptime-15,endtime=start+Ptime+15.).select(channel='*Z')[0]
    sig_amp = np.max(np.abs(trz.data))
    print('P wave amplitude = {}'.format(sig_amp))

    trz_noise = obspy.Trace()
    trn_noise = obspy.Trace()
    tre_noise = obspy.Trace()

    trz_noise.stats.npts = stc[0].stats.npts
    trn_noise.stats.npts = stc[0].stats.npts
    tre_noise.stats.npts = stc[0].stats.npts

    print(trz_noise)

    trz_noise.stats.sampling_rate = stc[0].stats.sampling_rate
    trn_noise.stats.sampling_rate = stc[0].stats.sampling_rate
    tre_noise.stats.sampling_rate = stc[0].stats.sampling_rate

    mean = 0
    std = 1
    noise_z = np.random.normal(mean,std,size=trz_noise.stats.npts)
    noise_n = np.random.normal(mean,std,size=trn_noise.stats.npts)
    noise_e = np.random.normal(mean,std,size=tre_noise.stats.npts)

    print('****')
    print(len(noise_z))
    print('****')

    #normalize to 1
    #noise_z /= np.max(np.abs(noise_z))
    #noise_n /= np.max(np.abs(noise_n))
    #noise_e /= np.max(np.abs(noise_e))

    #noise_z *= sig_amp
    #noise_n *= sig_amp
    #noise_e *= sig_amp

    #noise_z /= snr
    #noise_n /= snr
    #noise_e /= snr

    trz_noise.data = noise_z
    trn_noise.data = noise_n
    tre_noise.data = noise_e

    trz_noise.taper(0.05)
    trn_noise.taper(0.05)
    tre_noise.taper(0.05)

    trz_noise.filter('bandpass',freqmin=freqmin,freqmax=freqmax,corners=2)
    trn_noise.filter('bandpass',freqmin=freqmin,freqmax=freqmax,corners=2)
    tre_noise.filter('bandpass',freqmin=freqmin,freqmax=freqmax,corners=2)

    trz_noise.normalize()
    trn_noise.normalize()
    tre_noise.normalize()

    trz_noise.data *= sig_amp
    trn_noise.data *= sig_amp
    tre_noise.data *= sig_amp

    trz_noise.data /= snr
    trn_noise.data /= snr
    tre_noise.data /= snr

    #add noise
    stc.select(channel='*Z')[0].data += trz_noise.data
    stc.select(channel='*N')[0].data += trn_noise.data
    stc.select(channel='*E')[0].data += tre_noise.data

    np.savetxt('noise_z_snr{}.dat'.format(snr),np.c_[trz_noise.data])
    np.savetxt('noise_n_snr{}.dat'.format(snr),np.c_[trn_noise.data])
    np.savetxt('noise_e_snr{}.dat'.format(snr),np.c_[tre_noise.data])

    return stc


#------------------------------------------------------------
#S0235b info
#------------------------------------------------------------
origin = UTCDateTime('2019-07-26 12:15:17')
#gcarc = 28.3
gcarc = 44.0
#find geographic coordinates

# Keep the original source direction but place the event exactly 44 degrees
# from InSight. Instaseis derives distance from coordinates, not ``gcarc``.
evlo = 179.45783163994645
evla = 12.2608185206491
stlo = 135.62
stla = 4.50
evdp = 33.0
print(evlo,evla,evdp)

#instaseis_path = '/Volumes/G-DRIVE/instaseis_databases/kks21_2s_Q600_database/'
instaseis_path = '/Volumes/G-DRIVE/instaseis_databases/kks21_1s_d45_noatten_database'
db = instaseis.open_db(instaseis_path)

#Event 1: s/d/r; 175, 47, -90 | depth; 33km | mag; 3.6 | location based on S0235b
strike = 175
dip = 47
rake = -90
M0 = 3.16e14 

source = instaseis.Source.from_strike_dip_rake(latitude=evla,longitude=evlo,depth_in_m = evdp*1000.,strike=strike,dip=dip,rake=rake,M0=M0,origin_time=origin)
receiver = instaseis.Receiver(latitude=stla,longitude=stlo,network='XB',station='ELYSE',location='02')

st = db.get_seismograms(source=source,receiver=receiver,kind='displacement',dt=0.05)
st.write('synth_mt05_44deg_nonoise.mseed',format='MSEED')


#write snr 4
st_snr4 = add_noise(st,snr=4,gcarc=gcarc,freqmin=1./30,freqmax=1./2.)
st_snr4.write('synth_mt05_44deg_snr4.mseed',format='MSEED')

#write snr 2
st_snr2 = add_noise(st,snr=2,gcarc=gcarc,freqmin=1./30,freqmax=1./2.)
st_snr2.write('synth_mt05_44deg_snr2.mseed',format='MSEED')
