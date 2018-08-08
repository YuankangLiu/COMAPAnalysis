import numpy as np
from matplotlib import pyplot

# modification start - yliu
import os
import scipy.optimize as optimization
from scipy.optimize import curve_fit
from astropy.io import ascii
from scipy.interpolate import interp1d
# modification end - yliu

# def SimpleRemoval(tod, el, stride):
def SimpleRemoval(tod, el, az, stride, prefix, nu, anu):
    """
    Removes the atmospheric fluctuations in place
    
    returns the fitted amplitudes for the atmosphere
    """


    if stride < 0:
        stride = tod.shape[3]
    nHorns = tod.shape[0]
    nSidebands = tod.shape[1]
    nChans = tod.shape[2]
    # modification - yliu
    nEvents = tod.shape[3]
    # modification - yliu

    nSteps = tod.shape[-1]//stride
    Aout = np.zeros((nHorns, nSidebands, nChans, nSteps))

    # tod[horn, sideband, frequency, sample] # units of volts
    # el[horn, sample] # units of degrees
    # az[horn, sample]
    # ra/dec[horn, sample]
    # mjd[sample] # modified julian date ... convert to gregorian calander dates using jdcal package
    # write hdf5 files easily using: FileTools.WriteH5Py(filename, {'key1':data1, 'key2':data2})
    # read back in : d = FileTools.ReadH5Py(filename)
    # pyplot.plot(1./np.sin(el[0,:]*np.pi/180.),tod[0,0,31,:], ',')
    # pyplot.show()
    print('Output is below:')
    print(nHorns, nSidebands, nChans, nEvents)
    # there is no elevation > 60 in 103101? yes, in file 103101, elevation range from 5 - 55

    '''
    # modification - yliu
    samples = np.arange(nEvents)
    newpath = '/local/scratch/yliu/codes/COMAPAnalysis-master/comap_reduce/tempPlot/'+prefix+'/'
    if not os.path.isdir(newpath):
        os.makedirs(newpath)
    for m in range(nHorns): # nHorns
        for n in range(2): # nSidebands
            # Waterfall plot - channels vs. time
            fig = pyplot.figure()
            ax = fig.add_subplot(1,1,1)
            ax.imshow(tod[m, n, :, :], extent=[0, nEvents, 0, nChans], aspect='auto')
            ax.set_xlabel('Time')
            ax.set_ylabel('Channels')
            fig.savefig(newpath+'{}_Horn{}_Sideband{}_AllChannels_2dplot_new.png'.format(prefix, m, n))
            pyplot.clf()
            # Single frequency plot
            for f in range(15): # nChans
                fig = pyplot.figure()
                ax = fig.add_subplot(1,1,1)
                pyplot.plot(samples, tod[m, n, f, :])
                ax.set_xlabel('Time')
                ax.set_ylabel('TOD Values / Volts')
                fig.savefig(newpath+'{}_Horn{}_Sideband{}_Channel{}_1dplot_new.png'.format(prefix, m, n, f))
                pyplot.clf()
    
    '''
    # modification - yliu
    
    fig = pyplot.figure()
    ax = fig.add_subplot(2,1,1)
    pyplot.plot(tod[0,0,10,:])
    pyplot.plot(tod[1,0,10,:])
    fig.add_subplot(2,1,2)
    pyplot.plot(el[0,:])
    pyplot.plot(el[1,:], 'r-')
    pyplot.show()

    gddata = (el[0,:] > 30)
    gdtod = tod[0, 2, 10, gddata]
    gdel = el[0, gddata]
    pyplot.plot(1./np.sin(gdel*np.pi/180.), gdtod,',')
    pyplot.show()
    
    pyplot.plot(az[1,:], 'r-')
    pyplot.plot(az[0,:])
    pyplot.show()

    stop

    newpath = '/local/scratch/yliu/codes/COMAPAnalysis-master/comap_reduce/tempPlot/'+prefix+'/'
    '''
    print(nu)
    print(anu)
    gddata = (el[0,:] > 34) & (el[0,:] < 41)
    for v in range(nChans):
        print(len(tod[0,1,v,gddata]))
        print(tod[0,1,v,gddata])
    
    stop
    '''
    #t = 285 # Temperature at OVRO: 285 K

    # gddata = (el[0,:] > 60)
    # needs to be modofied
    # frequency gain array to compare with real Jupyter data
    freq_gain = np.zeros((nChans, 2))
    # freq vs gradient data - to compare with Jupiter data
    #freq_grad = np.zeros((nChans, 2))

    # freq vs TOD, test the telescope hovering region
    #freq_tod = np.zeros((nChans, 2))
    jupiter_data = ascii.read('COMAP_Gain.dat')
    f_p1 = interp1d(jupiter_data['col1'], jupiter_data['col2'], kind = 'cubic')
    f_p2 = interp1d(jupiter_data['col1'], jupiter_data['col3'], kind = 'cubic')

    
    freq_grad = np.zeros((nChans * 4, 2))
    freq_tod = np.zeros((nChans * 4, 2))
    
    # these regions are calculated from 2018-05-03-210617, 2018-05-04-034552, 2018-05-04-035349 files
    # the sequency must go from small values to greater values
    stat_el = ['39.991_40.007','49.969_50.021','59.889_60.060','69.516_70.250','79.209_80.468']
    mov_el = ['30.002_39.991','40.007_49.969','50.021_59.889','60.060_69.516','70.250_79.209']
    
    '''
    # these regions are from 2018-05-04-205121, 2018-05-05-053203, 2018-05-06-115446
    stat_el = ['34.996_35.006', '44.927_45.089', '54.918_55.119', '64.492_65.380', '74.057_76.137', '84.574_85.213']
    mov_el = ['30.002_34.996', '35.006_44.927', '45.089_54.918', '55.119_64.492', '65.380_74.057', '76.137_84.574']
    '''
    '''
    # these regions are from 2018-05-17-105217, 2018-05-17-103101
    mov_el = ['30.166_34.623', '35.079_41.471', '42.155_52.528']
    stat_el = ['34.623_35.079', '41.471_42.155', '52.528_53.75']
    '''
    ttau_arr = np.zeros((nHorns, len(mov_el))) # store the ttau values in an array
    for z in range(len(stat_el)):
        stat_str, stat_end = stat_el[z].split('_')
        mov_str, mov_end = mov_el[z].split('_')
        print(mov_str.split('.')[0], mov_end.split('.')[0])
        
        for h in range(nHorns):
            gddata = (el[h, :] > float(mov_str)) & (el[h, :] < float(mov_end)) # corresponding to 1/sinel ~ 1.** _ 1.**
            
            for s in range(nSidebands):
                for f in range(nChans):
                    gdtod = tod[h, s, f, gddata]
                    gdel = el[h, gddata]
                    sinel = 1./np.sin(gdel*np.pi/180)
                    para, covM = curve_fit(func, sinel, gdtod)
                    '''
                    sigma= np.std(gdtod[:(gdtod.size//2)*2:2]-gdtod[1:(gdtod.size//2)*2:2])/np.sqrt(2)
                    chi2 = np.sum((gdtod - (para[0] + para[1] * sinel))**2/sigma**2)
                    rdchi2 = chi2 / (len(sinel) - 2) # chi^2 divided by d.o.f, note that len(sinel)
                    '''
                    if s == 0 or s ==2:
                        # freq gradient data 
                        freq_grad[s * nChans + f][0] = 26 + s * 2 + anu[f]
                        freq_grad[s * nChans + nChans - f - 1][1] = para[1]
                    else: 
                        freq_grad[s * nChans + f][0] = 26 + s * 2 + anu[f]
                        freq_grad[s * nChans + f][1] = para[1]

            newprefix2 = prefix + '_Freq_Grad_Horn_'+ str(h) + '_el_' + mov_str.split('.')[0] + '_' +mov_end.split('.')[0]  + '.txt'
            np.savetxt(newprefix2, freq_grad)
            pyplot.figure()

            if h == 0: 
                pyplot.suptitle('Freq vs Grad Plot')
                pyplot.plot(freq_grad[:,0], freq_grad[:,1], 'b-', label = '1st Pixel')
                #pyplot.plot(jupiter_data['col1'], jupiter_data['col2'], 'r-', label = 'calbibrated')
                pyplot.legend()
                pyplot.xlabel('Frequency [GHz]')
                pyplot.ylabel('Gain * Tatm * tau')
                #pyplot.show()
                fig_name = prefix + '_Freq_Grad_Pixel_1_el_' + mov_str.split('.')[0] + '_' + mov_end.split('.')[0] + '.png'
                #pyplot.savefig(fig_name)
            else:
                pyplot.suptitle('Freq vs Grad plot')
                pyplot.plot(freq_grad[:,0], freq_grad[:,1], 'b-', label = '12th Pixel')
                #pyplot.plot(jupiter_data['col1'], jupiter_data['col3'], 'r-', label = 'calibrated')
                pyplot.legend()
                pyplot.xlabel('Frequency [GHz]')
                pyplot.ylabel('Gain * Tatm * tau')
                #pyplot.show()
                fig_name = prefix + '_Freq_Grad_Pixel_12_el_' + mov_str.split('.')[0] + '_' + mov_end.split('.')[0] + '.png'
                #pyplot.savefig(fig_name)

            gddata = (el[h, :] > float(stat_str)) & (el[h, :] < float(stat_end)) # correponding to i/sinel = 1.3052 _ 1.3056
            for s in range(nSidebands):
                for f in range(nChans):
                    gdtod = tod[h, s, f, gddata]
                    avgtod = np.sum(gdtod) / len(gdtod) # average tod value to get TOD-freq plot
                    if s == 0 or s == 2:
                        # freq TOD data 
                        freq_tod[s * nChans + f][0] = 26 + s * 2 + anu[f]
                        freq_tod[s * nChans + nChans - f - 1][1] = avgtod
                    else: 
                        freq_tod[s * nChans + f][0] = 26 + s * 2 + anu[f]
                        freq_tod[s * nChans + f][1] = avgtod
            newprefix3 = prefix + '_Freq_TOD_Horn_'+ str(h) + '_el_' + stat_str.split('.')[0] +'.txt'
            np.savetxt(newprefix3, freq_tod)
            pyplot.figure()
            if h == 0:
                pyplot.suptitle('Freq vs TOD plot')
                pyplot.plot(freq_tod[:,0], freq_tod[:,1], 'b-', label = '1st Pixel')
                #pyplot.plot(jupiter_data['col1'], jupiter_data['col2'], 'r-', label = 'calbibrated')
                pyplot.legend()
                pyplot.xlabel('Frequency [GHz]')
                pyplot.ylabel('TOD values')
                #pyplot.show()
                fig_name = prefix + '_Freq_TOD_Pixel_1_' + stat_str.split('.')[0] + '.png'
                #pyplot.savefig(fig_name)
            else:
                pyplot.suptitle('Freq vs TOD plot')
                pyplot.plot(freq_tod[:,0], freq_tod[:,1], 'b-', label = '12th Pixel')
                #pyplot.plot(jupiter_data['col1'], jupiter_data['col3'], 'r-', label = 'calibrated')
                pyplot.legend()
                pyplot.xlabel('Frequency [GHz]')
                pyplot.ylabel('TOD values')
                #pyplot.show()
                fig_name = prefix + '_Freq_TOD_Pixel_12_el_' + stat_str.split('.')[0] + '.png'
                #pyplot.savefig(fig_name)

            # calculate T tau value and hence the system temperature
            T_tau, covM_ttau = curve_fit(func, f_p1(freq_grad[:,0]), freq_grad[:,1])
            print(T_tau)
            print('The covariant matrix is shown:')
            print(covM_ttau)
            print('The error in Ttau is')
            error_ttau = np.sqrt(covM_ttau[1][1])
            print(error_ttau)


            ttau_arr[h][z] = T_tau[1] # store the T-tau values
       
            # store the Tsys data
            nu_Tsys = np.zeros((len(freq_tod),2))
            for q in range(len(freq_tod)):
                nu_Tsys[q][0] = freq_tod[q][0]
                if h == 0:
                    nu_Tsys[q][1] = (freq_tod[q][1]*np.sin(50*np.pi/180)/f_p1(freq_tod[q][0]) - T_tau[1]) / np.sin(50 * np.pi / 180)
                else: 
                    nu_Tsys[q][1] = (freq_tod[q][1]*np.sin(50*np.pi/180)/f_p2(freq_tod[q][0]) - T_tau[1]) / np.sin(50 * np.pi / 180)
            newprefix7 = prefix + '_System_Temperature_Method_1_Pixel_'+ str(h) + '_el_' + stat_str.split('.')[0] +'.txt'
            np.savetxt(newprefix7, nu_Tsys)

            pyplot.figure()
            pyplot.title('System Temperature vs. Frequency')
            if h == 0:
                pyplot.plot(freq_tod[:, 0], (freq_tod[:, 1]*np.sin(50*np.pi/180)/f_p1(freq_tod[:, 0]) - T_tau[1]) / np.sin(50 * np.pi / 180), 'm-', label = '1st Pixel')
                pyplot.legend()
                pyplot.xlabel('Frequency [GHz]')
                pyplot.ylabel('System temperature [K]')
                #pyplot.show()
                fig_name = prefix + '_Sys_Temp_Pixel_1_el_' + stat_str.split('.')[0] + '.png'
                #pyplot.savefig(fig_name)
            else:
                pyplot.plot(freq_tod[:, 0], (freq_tod[:, 1]*np.sin(50*np.pi/180)/f_p2(freq_tod[:, 0]) - T_tau[1]) / np.sin(50 * np.pi / 180), 'm-', label = '12th Pixel')
                pyplot.legend()
                pyplot.xlabel('Frequency [GHz]')
                pyplot.ylabel('System temperature [K]')
                #pyplot.show()
                fig_name = prefix + '_Sys_Temp_Pixel_12_el_' + stat_str.split('.')[0] + '.png'
                #pyplot.savefig(fig_name)
            pyplot.clf()
    print('The Ttau values are shown below')
    print(ttau_arr)
    print('Areraged Ttau values for Pixel 1:')
    print(np.mean(ttau_arr[0,2:])) # omit the values at low elevations
    print('Averaged Ttau values for Pixel 12:')
    print(np.mean(ttau_arr[1,2:]))

    # calculate system temperature using another method --- fitting to get the offset
    name_index = np.zeros(len(stat_el))
    for v in range(len(stat_el)):
        name_index[v] = str(stat_el[v].split('.')[0]) # substract the interger part
    print(name_index) # why is it still float type?
    
    
    for h in range(nHorns):
        filenames = ['' for i in range(len(name_index))]
        for i in range(len(filenames)):
            filenames[i] = prefix + '_Freq_TOD_Horn_' + str(h) + '_el_' + str(int(name_index[i])) + '.txt'

        #filename = prefix_list + word1_list + pixel_list + word2_list + name_index + word3_list
        #print(filenames)
        data_dict = dict.fromkeys(name_index) # create an empty dictionary
        for x in range(len(name_index)):
            data_dict[name_index[x]] = np.loadtxt(filenames[x]) # assign a 2D array to a key of the dict
        #print(data_dict)
        gain_Tsys = np.zeros((len(data_dict[name_index[0]]),2)) # uncalibrated system temperature
        
        sinel_fitting = np.zeros(len(name_index))
        for i in range(len(name_index)):
      
            sinel_fitting[i] = 1/np.sin(float(stat_el[i].split('_')[0])) # hope the data sequency remains   
        tod_fitting = np.zeros(len(name_index))
        
        for i in range(len(data_dict[name_index[0]])): # access all frequency values
            for j in range(len(name_index)):
                tod_fitting[j] = data_dict[name_index[j]][i][1]
            gain_Tsys[i][0] = data_dict[name_index[0]][i][0]
            gain_Tsys[i][1] = curve_fit(func, sinel_fitting, tod_fitting)[0][0]
        
        nu_Tsys2 = np.zeros((len(data_dict[name_index[0]]),2)) # calibrated system temperature
        for i in range(len(data_dict[name_index[0]])):
            nu_Tsys2[i][0] = data_dict[name_index[0]][i][0]
            if h == 0:
                nu_Tsys2[i][1] = gain_Tsys[i][1] / f_p1(gain_Tsys[i][0])
            else:
                nu_Tsys2[i][1] = gain_Tsys[i][1] / f_p2(gain_Tsys[i][0])

        newprefix8 = prefix + '_System_Temperature_Method_2_Pixel_'+ str(h) + '.txt'
        np.savetxt(newprefix8, nu_Tsys2)

        pyplot.figure()
        if h == 0:
            
            pyplot.plot(gain_Tsys[:,0], gain_Tsys[:,1]/f_p1(gain_Tsys[:,0]), label = '1st Pixel')
            pyplot.xlabel('Frequency [GHz]')
            pyplot.ylabel('System Temperature [K]')
            pyplot.legend()
            pyplot.savefig(prefix + '_Pixel_' + str(h) + '_Method_2_SysTemp.png')
        else:
            newprefix8 = prefix + '_System_Temperature_Method_2_Pixel_'+ str(h) + '.txt'
            np.savetxt(newprefix8, gain_Tsys)
            pyplot.plot(gain_Tsys[:,0], gain_Tsys[:,1]/f_p2(gain_Tsys[:,0]), label = '12th Pixel')
            pyplot.xlabel('Frequency [GHz]')
            pyplot.ylabel('System Temperature [K]')
            pyplot.legend()
            pyplot.savefig(prefix + '_Pixel_' + str(h) + '_Method_2_SysTemp.png')
        





    '''
    gddata = (el[0,:] > 40)
    for h in range(nHorns):
        for s in range(3): # nsidebands
            for f in range(15):
                fig = pyplot.figure()
                gdtod = tod[h, s, f, gddata]
                gdel = el[h, gddata]
                nbins = 200
                sinbins = np.linspace(1, 2, nbins+1)
                row = np.histogram(1./np.sin(gdel*np.pi/180.), sinbins, weights=gdtod)[0]/ np.histogram(1./np.sin(gdel*np.pi/180.), sinbins)[0]
                binmids = (sinbins[1:] + sinbins[:-1])/2.
                ax = fig.add_subplot(111)
                pyplot.plot(binmids, row)
                ax.set_xlabel('1/sin(EL)')
                ax.set_ylabel('Gain*T')
                


                pyplot.show()
                fig.savefig(newpath+'{}_Horn{}_Sideband{}_Channels{}_Opacity_Fitting.png'.format(prefix, h, s, f))
                # get rid of the nan values
                xdata = []
                ydata = []
                gt = []
                for m in range(len(row)):
                    if str(row[m]) != 'nan':
                        xdata.append(sinbins[m])
                        ydata.append(row[m])
                a, k = curve_fit(func, xdata, ydata)[0]
                err = curve_fit(func, xdata, ydata)[1]
                gt.append(k)
                print(k, a)
                print(err)
    '''
                
    '''
                coeff = np.polyfit(xdata, ydata, 2)
                fity = xdata*coeff[0] + coeff[1]
                print(np.polyfit(xdata, ydata, 1))
                pyplot.plot(xdata, fity, '--')
                pyplot.show()
    '''
    '''
                x0 = np.array([0.0, 0.0, 0.0])
                sigma = np.empty(len(xdata))
                sigma.fill(1)
                print(len(xdata))
                print(len(ydata))
                print(optimization.curve_fit(func, xdata, ydata, x0, sigma))
    '''
            
                


    '''
    gddata = (el[0,:] > 40) # data of elevation greater than 30 
    gdtod = tod[0, 1, 10, gddata]
    gdel = el[0, gddata]
    #pyplot.plot(1./np.sin(gdel*np.pi/180.), gdtod, ',')
    #pyplot.show()
                                                                 
    nbins = 100
    sinbins = np.linspace(1, 2, nbins+1)
    row = np.histogram(1./np.sin(gdel*np.pi/180.), sinbins, weights=gdtod)[0]/ np.histogram(1./np.sin(gdel*np.pi/180.), sinbins)[0]
    binmids = (sinbins[1:] + sinbins[:-1])/2.
    pyplot.plot(binmids, row)
    pyplot.show()
    '''
    for i in range(nSteps):
        lo = i*stride
        hi = (i+1)*stride
        for j in range(nHorns):
            # Simple atmosphere removal:
            A = 1./np.sin(el[j,:]*np.pi/180.)
            for k in range(nSidebands):
                for l in range(nChans):
                    pmdl = np.poly1d( np.polyfit(A, tod[j,k,l,lo:hi], 1))
                    tod[j,k,l,lo:hi] -= pmdl(A)
                    Aout[j,k,l,i] = pmdl[1]

    return Aout


# modification start - yliu
def func(x, a, k):
    return a + k*x

'''
def SingleFreqPlot(tod, el, horns, sidebands,prefix):
    fig = pyplot.figure()
    ax = fig.add_subplot(2,1,1)
    pyplot.plot(tod[0,0,10,:])
    pyplot.plot(tod[1,0,10,:])
    fig.add_subplot(2,1,2)
    pyplot.plot(el[0,:])
    pyplot.plot(el[1,:])
    pyplot.show()

    gddata = (el[horns,:] > 30)
    gdtod = tod[horns, sidebands, 10, gddata]
    gdel = el[horns, gddata]
    pyplot.plot(1./np.sin(gdel*np.pi/180.), gdtod,',')
    pyplot.show()

    nbins = 100
    sinbins = np.linspace(1, 3, nbins+1)
    row = np.histogram(1./np.sin(gdel*np.pi/180.), sinbins, weights=gdtod)[0]/ np.histogram(1./np.sin(gdel*np.pi/180.), sinbins)[0]
    binmids = (sinbins[1:] + sinbins[:-1])/2.
    pyplot.plot(binmids, row)
    pyplot.show()
#   pyplot.clf()
'''
# modification - yliu
