#-------------------------------------------------------------------------------
# Name:        Copula fitting using maximum likelihood
#
# Author:      Dr.-Ing. P. Guthke, Dr.-Ing. S. Hoerning
#
# Created:     02.05.2018, University Stuttgart, Stuttgart, Germany
#                          The University of Queensland, Brisbane, QLD, Australia
# Copyright:   (c) Hoerning 2018
#-------------------------------------------------------------------------------
import numpy as np
import os
import matplotlib.pylab as plt
import scipy.stats as st
import scipy.spatial as sp
import scipy.optimize as opt
import datetime
import covariancefunction as variogram
import gaussian_copula as thcopula


def paraest_multiple_tries(
                x,                          # coordinates [n x d] in blockmode [n x d x disc], where disc is discretization
                u,                          # values in copula space  [n x 1]
                ntries=[6,6],               # [no. of tries with same subsets (diff starting parameters), no. of tries with different subsets]
                n_in_subset=8,              # number of values in subset
                neighbourhood='nearest',    # subset search algorithm
                covmods=['Mat', 'Exp', 'Sph',] ,
                outputfile=None,            # if valid path+fname string --> write output file
                talk_to_me=True,
                ):

    if outputfile != None:
        if os.path.isdir(os.path.dirname(os.path.abspath(outputfile))) == True:
            fout = open(outputfile, 'a')
            fout.write('# ------------------------------------------------------------- #\n')
            fout.write('# ------------------------------------------------------------- #\n')
            fout.write('# OPTIMIZED PARAMETERS: %s - nested --- #\n'%str(datetime.datetime.now()))
            fout.write('# neighbourhood:      "%i" values with "%s"\n'%(n_in_subset, neighbourhood))
            fout.write('# number of different subsets built: %i\n'%(ntries[1]))
            fout.write('# number of tries for each subset selection: %i, but only the one with best likelihood is displayed\n'%(ntries[0]))
            fout.write('# covariance model \n')
            fout.flush()

    # number of spatial models
    nspamods = len(covmods)

    # draw ntries random states for the setting of starting parameters
    randstates_startpar = []
    for i in range(ntries[0]):
        np.random.rand(np.random.randint(10000, 100000))
        state = np.random.get_state()
        randstates_startpar.append(state)

    # draw ntries random states for the building of subsets
    randstates_subsets = []
    for i in range(ntries[0]):
        np.random.rand(np.random.randint(10000, 100000))
        state = np.random.get_state()
        randstates_subsets.append(state)

    # loop over different spatial models
    out = []
    for mod in range(nspamods):
        covmods0    = covmods[mod]

        # lop over different subsets
        out0 = []
        for subset in range(ntries[1]):

            # loop for different starting parameters for one model
            for startings in range(ntries[0]):
                np.random.set_state(randstates_startpar[startings])
                out000 = paraest_g(
                        x,
                        u,
                        n_in_subset=n_in_subset,
                        neighbourhood=neighbourhood,
                        seed = randstates_subsets[subset],
                        covmods = covmods0,
                        outputfile=None,
                        talk_to_me=True,
                        )
                    
                # only take best of optimizations of same subsets
                if startings == 0:
                    out00 = out000
                else:
                    if out000[1] < out00[1]:
                        out00 = out000

            if outputfile != None:
                if os.path.isdir(os.path.dirname(os.path.abspath(outputfile))) == True:
                    fout.write('#\n')
                    # reconstruct parameters
                    p           = out00[0]
                    Likelihood  = out00[1] * -1
                    message     = out00[2]
                    
                    cov_model = reconstruct_parameters(p,covmods0)
                    
                    fout.write('%s \n'%(cov_model))
                    fout.write('# Likelihood: %1.3f\n'%Likelihood)
                    fout.write('# message: %s\n'%message)
                    fout.flush()        

            out0.append(out00)
        out.append(out0)
    if outputfile != None:
        fout.write('# ------------------------------------------------------------- #\n')
        fout.close()
    return out



def paraest_g(
                x,                          # coordinates [n x d] OR [n x d x disc]
                u,                          # values in copula space  [n x 1]
                n_in_subset=8,              # number of values in subset
                neighbourhood='random',    # subset search algorithm
                seed = None,               
                covmods = ['Mat'],    
                outputfile=None,            
                talk_to_me=True,
                ):
   
    # BUILD SUBSETS  
    if seed != None:
        curstate = np.random.get_state()
        np.random.set_state(seed)

    ind = build_subsets(
                        x,
                        n_in_subset=8,
                        how=neighbourhood,
                        talk_to_me=talk_to_me,
                        plot_me=0)

    x0 = x[ind.astype(int)]
    u0 = u[ind.astype(int)]
    if seed != None:
         np.random.set_state(curstate)

    # calc distance matrices in each subset
    d0 = np.zeros((x0.shape[0], x0.shape[1], x0.shape[1]))
    for i in range(x0.shape[0]):
        d0[i] = distance_matrix_block(x0[i], x0[i])

    args = (d0,                 
            u0,                 
            covmods,
            talk_to_me)

    p_bounds=[]   
    Rangebounds = [[1,  d0.max()*3]]
    p_bounds += Rangebounds

    if covmods == 'Mat':
        Extrabounds = [[0.05, 50.0]]
        p_bounds += Extrabounds
  
    p_bounds = tuple(p_bounds)

    # random starting parameter set
    p_start = []
    for i in range(len(p_bounds)):
        p0 = np.random.uniform(p_bounds[i][0], p_bounds[i][1])
        p_start.append(p0)        

    out = opt.fmin_l_bfgs_b(Likelihood,
                           p_start,
                           bounds=list(p_bounds),
                           args=args,
                           approx_grad=True,
                           )

    
    # another optimizer in case lbgfs estimate is at bound
    if p_bounds[0][0] == out[0][0]:
        out2 = opt.minimize(Likelihood,
                           [100],
                           method = 'Nelder-Mead',
                           bounds=list(p_bounds),
                               args=args,
                               #approx_grad=True,
                               )
        out = [out2.x, out2.fun, 'nelder-mead']
    p = out[0]
    # Plot optimum:
    #iiii = []
    #for ii in np.linspace(10, Rangebounds[0][1], 100):
    #    iiii.append(Likelihood([ii], d0, u0, covmods))
    #plt.plot(iiii)
    #plt.show()    
    #print(out)
    cov_models = reconstruct_parameters(p,covmods)
    if outputfile != None:
        if os.path.isdir(os.path.dirname(os.path.abspath(outputfile))) == True:
            Like = -1*out[1]
            fout = open(outputfile, 'a')
            fout.write('# OPTIMIZED PARAMETERS: %s - nested --- #\n'%str(datetime.datetime.now()))
            fout.write('# neighbourhood:      "%i" values with "%s"\n'%(n_in_subset, neighbourhood))
            fout.write('# Likelihood: \n%i\n'%Like)
            fout.write('# message: %s'%out[2])
            fout.write('# covariance model \n')   
            fout.write('%s \n'%(cov_models))
            fout.write('# ------------------------------------------------------------- #\n')
            fout.close()

    return out

def reconstruct_parameters(p, covmods):
    
    counter = 0
   
    cov_model = ''
    Range = p[counter]
    counter += 1
    cov_model += '1.0 %s(%1.12f)'%(covmods, Range)

    if covmods=='Mat':
        Param = p[counter]
        counter += 1
        cov_model += '^%1.12f'%Param    
    cov_models = cov_model

    return cov_models


def Likelihood(p, Ds, us, covmods, talk_to_me=True):
    # p - parameter for cov model
    # Ds - distances
    # us - values copula space for this neighbourhood 
    # covmods - covmod to try

    cov_models = reconstruct_parameters(p, covmods)
    
    # covariance matrix between possitions
    Qs = np.array(variogram.Covariogram(Ds, cov_models))
    # copula densities
    cs = []
    for i in range(us.shape[0]):
        # vector of likelihood of each observation (distance)
        cs.append(thcopula.multivariate_normal_copula_pdf(us[i], Qs[i]))
    cs = np.array(cs)
    
    # avoid numerical errors
    cs[np.where(cs==0)] = 0.000000000000001

    # log Likelihood
    L = (np.log(cs)).sum()

    return -L   # negative likelihood --> minimization

# TODO: the coordinate vector has a different descrition here. 
def build_subsets(  coords,             # 
                    n_in_subset=6,      # number of points in subset
                    how='random',      # subset building routine
                    talk_to_me=False,
                    plot_me=False
                    ):

    n_coords  = coords.shape[0]
    n_dims = len(coords.shape)
    n_subsets = int(np.floor(float(n_coords)/n_in_subset))
    n_used    = int(n_subsets*n_in_subset)

    if how == 'random':
        ind = np.arange(n_used)
        np.random.shuffle(ind)
        ind = (ind.reshape((n_subsets, n_in_subset))).astype('int')

    if how == 'nearest':
        ind = [] 
        not_taken = np.ones(coords.shape[0]).astype('bool')

        for subset in range(n_subsets):
            # take one point randomly
            i = np.where(not_taken==True)[0] # idices of all that are not taken
            np.random.shuffle(i) 
            i_1 = i[0] # first random indice
            # mark it
            not_taken[i_1] = False # take it...

            # calc distances to other coordinates, zero ind only removes nested list
            #d = sp.distance_matrix(coords[i_1][np.newaxis], coords[not_taken])[0]
            d = distance_matrix_block(coords[i_1][np.newaxis], coords[not_taken])[0]

            i_closest = np.argsort(d)[:n_in_subset-1]

            # retransform indices to coords array
            i_closest = np.arange(coords.shape[0])[not_taken][i_closest]
            not_taken[i_closest] = False

            i_subset = np.concatenate(([i_1], i_closest))
            i_subset = np.sort(i_subset)

            ind.append(i_subset)
        ind = np.array(ind)
        
        
    if plot_me == True: # modified, erlend
        x = coords[ind]
        if len(x.shape) == 4:
            print(x.shape) 
            # x: 83 neigbourhoods, 5 x-y pairs in each neighbourhood, 2 x,y, 17 different variants alon the line
            plt.figure()
            for i,xy in enumerate(x): # x neightbourhoods
                for ii in range(x.shape[3]):
                    xy_plot = xy[np.argsort(xy[:,0, ii])][:, :, ii]
                    plt.plot(xy_plot[:,0], xy_plot[:,1], '.-', alpha=0.5)
            plt.show()
            
            # only first for comparison..
            for i,xy in enumerate(x): # x neightbourhoods
                xy_plot = xy[np.argsort(xy[:,0, 0])][:, :, 0]
                plt.plot(xy_plot[:,0], xy_plot[:,1], '.-', alpha=0.5)
            plt.show()
            print('##################')
        else:
            plt.figure()
            for i,xy in enumerate(x):
                xy = xy[np.argsort(xy[:,0])]
                plt.plot(xy[:,0], xy[:,1], '.-', alpha=0.5)
            plt.show()

    return ind

# TODO: Upgrade to full linearization?  
def distance_matrix_block(device_1, device_2):
    """
    distance matrix between blocks (CML) according to Øydvin 2023(24?)
    """
    #print(device_1)
    #print(device_2)
    if len(device_1.shape) == 3:
                
        # Lengths between all points in all blocks:
        delta_x = np.array([device_1[i][1] - device_2[j][1].reshape(
            -1, 1) for i in range(device_1.shape[0]) for j in range(
                device_2.shape[0])])
        
                
        delta_y = np.array([device_1[i][0] - device_2[j][0].reshape(
            -1, 1) for i in range(device_1.shape[0]) for j in range(
                device_2.shape[0])])
                                   
        lengths_point_l = np.sqrt(delta_x**2 + delta_y**2)
        
        lengths_blokkblokk = lengths_point_l.mean(axis=(1, 2)).reshape(
            int(device_1.shape[0]), int(device_2.shape[0]))
        
        
        #xx = np.arange(lengths_blokkblokk.shape[0])
        #diag = lengths_blokkblokk[xx, xx]
        #    lengths_blokkblokk[xx, xx] = np.sqrt(lengths_blokkblokk[xx, xx])
            #np.fill_diagonal(lengths_blokkblokk, 0)
            
    
        if lengths_blokkblokk.shape[0] != 1:
            np.fill_diagonal(lengths_blokkblokk, 0)
            
        #else: 
         #   pass #lengths_blokkblokk = np.flip(lengths_blokkblokk)


        #print(lengths_blokkblokk)
        #print('yolo')
        # since two blocks, in principle can be on top of each other:
        # we subtract within lengths, just as a number... 
        # now the diagonal should become zero! 
        
        # withinblock length of links, that is all links inside device_2 list,
        # note that links in device_1 list is also in device_2 list, but not 
        
        # the other way around
        #delta_x = np.array([device_2[i][1] - device_2[i][1].reshape(
        #    -1, 1) for i in range(device_2.shape[0])])
        #
        #delta_y = np.array([device_2[i][0] - device_2[i][0].reshape(
        #    -1, 1) for i in range(device_2.shape[0])])
                
        #lengths_withinblock = np.sqrt(delta_x**2 + delta_y**2).mean(axis=(1, 2))
        #lengths_withinblock = (1/2)*(
        #    lengths_withinblock + lengths_withinblock.reshape(-1, 1)) # mean of the two blocks
        
        #lengths_withinblock[xx, xx] = np.zeros(xx.size)
        #lengths_blokkblokk = lengths_blokkblokk - lengths_withinblock
        

        return lengths_blokkblokk
    else:
        #lengths_blokkblokk = sp.distance_matrix(device_1, device_2)

        #print(lengths_blokkblokk)
        #print('yolo')

        return sp.distance_matrix(device_1, device_2)
        
    
    
