#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:45:02 2023

@author: erlend
"""

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
                maxrange=None,
                minrange=None,
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
                        maxrange = maxrange,
                        minrange = minrange,
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
                maxrange = None,
                minrange = None, 
                ):
   
    # BUILD SUBSETS  
    if seed != None:
        curstate = np.random.get_state()
        np.random.set_state(seed)
    #np.random.seed(2)
    ind = build_subsets(
                        x,
                        n_in_subset=n_in_subset,
                        how= neighbourhood, # nearest, random
                        talk_to_me=talk_to_me,
                        plot_me=0)
    # ind has shape [n_subsets, n_in_subset] thus 
    x0 = x[ind]
    u0 = u[ind]
    if seed != None:
         np.random.set_state(curstate)
    
    # d0 [blockblock/withinblock, neighbourhood, link, link, disc, disc]
    d0 =  np.zeros((2, x0.shape[0], x0.shape[1], x0.shape[1], 
                    x0.shape[3], x0.shape[3]))
    for i in range(x0.shape[0]): # for through neighbourhoods
        d0[0][i] = block_block_l(x0[i], x0[i])  
        d0[1][i] = within_block_l(x0[i], x0[i]) 
    
    args = (d0,                 
            u0,                 
            covmods,
            talk_to_me)
    

    p_bounds=[]   
    
    # max value observed from midpoint method in 3_RM
    if maxrange is not None:
        if minrange is not None:
            Rangebounds = [[minrange,  maxrange], [1, 5]] # maximum distance for 
        else: 
            Rangebounds = [[3,  maxrange], [1, 5]] # maximum distance for 
    else:
        Rangebounds = [[3,  60], [1, 5]] # not used in analysis

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
    
    
    # 
    
    #print('copula oob be like: ', str(Likelihood_exp_block_debug([10, 3],d0, u0, covmods)))
    #print('copula instide be like: ', str(Likelihood_exp_block_debug([10, 0.5],d0, u0, covmods)))

    #raise
    out = opt.fmin_l_bfgs_b(Likelihood_exp_block,
                           p_start,
                           bounds=list(p_bounds),
                           args=args,
                           approx_grad=True,
                           )    
    
    p = out[0]
    
    # out2 = opt.differential_evolution(Likelihood_exp_block,
    #                         #p_start,
    #                         bounds=list(p_bounds),
    #                         args=args,
    #                         disp = False,
    #                         #approx_grad=True,
    #                         )
    
    # p2 = np.array([out2.x[0], out2.x[1]])

    # #  # plot optimum 
    # hh = np.linspace(Rangebounds[0][0], Rangebounds[0][1], 15)
    # ss = np.linspace(Rangebounds[1][0], Rangebounds[1][1], 15)
    # xx = np.zeros([hh.size, ss.size])
    
    # for hh_i in range(len(hh)):
    #     for ss_i in range(len(ss)):
    #         xx[hh_i, ss_i] = Likelihood_exp_block(
    #             [hh[hh_i], ss[ss_i]],
    #             d0, 
    #             u0, 
    #             covmods
    #             )
    # ss_grid, hh_grid = np.meshgrid(ss, hh)
    # plt.pcolormesh(ss_grid, hh_grid, xx)
    
    # plt.plot(p[1], p[0], 'ob')
    # plt.plot(p2[1], p2[0], 'xr')
    # plt.colorbar()
    # plt.show()
    # print(Likelihood_exp_block([p[0], p[1]],d0, u0, covmods))
    #print('copula looks like: ', str(Likelihood_exp_block_debug([out.x[0], out.x[1]],d0, u0, covmods)))

    # lengths for the standard point method, for comparison

    # ### Ds - [blockblock/withinblock, neighbourhood, link, link, disc, disc]
    cov_models = reconstruct_parameters(p, covmods)
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

def Likelihood_exp_block_cov(p, Ds, us, covmods, talk_to_me=True):
    # p - parameter for cov model
    # Ds - [blockblock/withinblock, neighbourhood, link, link, disc, disc]
    # us - values copula space for this neighbourhood 
    # covmods - covmod to try  
    
    # The rank transformation makes the block data have a variance of 1, but
    # but the model has a different variance? 

    lengths_point_l = Ds[0]
    lengths_withinblock_l = Ds[1]

    # 1 calculate block block variance
    def type_exp_block(h):
        return p[1]*(np.exp(-h/p[0]))
    
    
    # Point variance eq 23, 24
    # this is the average variogram between two blocks
    cov_point = type_exp_block(lengths_point_l).mean(axis=(3, 4))
    # NOTE mean(3, 4): This takes the average of the whole matrix which
    # is all cross terms between the two blocks. The diagonal is not zero, but
    # the 0-0, 1-1 etc terms. 

    # Within variance eq 22
    cov_a =  type_exp_block(lengths_withinblock_l).mean(axis = (3, 4))     
    
    # transpose to get compute crosscorrelation that can be quickly substracted
    cov_b =  np.transpose(cov_a, axes = (0, 2, 1))
    cov_withinblock = 0.5*(cov_a + cov_b) # eq 22
        
    # General expression from geostatistics
    Qs = cov_point - cov_withinblock  + np.ones(cov_point.shape)
    
    return Qs # negative likelihood --> minimization



def Likelihood_exp_block(p, Ds, us, covmods, talk_to_me=True):
    # p - parameter for cov model
    # Ds - [blockblock/withinblock, neighbourhood, link, link, disc, disc]
    # us - values copula space for this neighbourhood 
    # covmods - covmod to try  
    
    # The rank transformation makes the block data have a variance of 1, but
    # but the model has a different variance? 

    lengths_point_l = Ds[0]
    lengths_withinblock_l = Ds[1]

    # 1 calculate block block variance
    def type_exp_block(h):
        return p[1]*(np.exp(-h/p[0]))
    
    
    # Point variance eq 23, 24
    # this is the average variogram between two blocks
    cov_point = type_exp_block(lengths_point_l).mean(axis=(3, 4))
    # NOTE mean(3, 4): This takes the average of the whole matrix which
    # is all cross terms between the two blocks. The diagonal is not zero, but
    # the 0-0, 1-1 etc terms. 

    # Within variance eq 22
    cov_a =  type_exp_block(lengths_withinblock_l).mean(axis = (3, 4))     
    
    # transpose to get compute crosscorrelation that can be quickly substracted
    cov_b =  np.transpose(cov_a, axes = (0, 2, 1))
    cov_withinblock = 0.5*(cov_a + cov_b) # eq 22
        
    # General expression from geostatistics
    Qs = cov_point - cov_withinblock  + np.ones(cov_point.shape)
    
    # regularization for numerical stability when sugested covariance is less than zero
    for i in range(Qs.shape[0]):
        min_eig = np.linalg.eigvals(Qs[i]).min()
        if min_eig < 0:
            Qs[i] = Qs[i] - 50*min_eig*np.eye(Qs[i].shape[0])   

    # copula densities
    cs = []
    for i in range(us.shape[0]):
        # vector of likelihood of each observation (distance)
        cs.append(thcopula.multivariate_normal_copula_pdf(us[i], Qs[i]))
    cs = np.array(cs) # likelihood for all neigbourhoods
    # avoid numerical errors
    cs[np.where(cs==0)] = 0.000000000000001

    # log Likelihood
    L = (np.log(cs)).sum()
    L = np.nan_to_num(L) # set nan to zero
    return -L 


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
        if len(x.shape) == 4: # for plotting the block version
            # x: 83 neigbourhoods, 5 x-y pairs in each neighbourhood, 2 x,y, 17 different variants alon the line
            plt.figure()
            
            colors = plt.cm.jet(np.linspace(0, 7, x.shape[1]*x.shape[0]))
            c_i = 0
            for i,xy in enumerate(x): # x neightbourhoods
                plot_order = np.argsort(xy[:,0, 0])
                plt.plot(xy[plot_order, 0, :],xy[plot_order, 1, :], '.-', alpha=0.5, color = colors[c_i])
                c_i += 1
            plt.show()
            
        else:
            plt.figure()
            for i,xy in enumerate(x):
                xy = xy[np.argsort(xy[:,0])]
                plt.plot(xy[:,0], xy[:,1], '.-', alpha=0.5)
            plt.show()

    return ind


        
def block_block_l(device_1, device_2):
    """
    Calculates distance matrix between all discretization steps in all blocs
    Return: lengths_point_l : [link, link, gosh_length, gosh_length]
    """
    delta_x = np.array([device_1[i][1] - device_2[j][1].reshape(
        -1, 1) for i in range(device_1.shape[0]) for j in range(
            device_2.shape[0])])
    delta_y = np.array([device_1[i][0] - device_2[j][0].reshape(
        -1, 1) for i in range(device_1.shape[0]) for j in range(
            device_2.shape[0])])
    
    lengths_point_l = np.sqrt(delta_x**2 + delta_y**2)
    return lengths_point_l.reshape(int(np.sqrt(lengths_point_l.shape[0])), 
                                   int(np.sqrt(lengths_point_l.shape[0])),
                                   lengths_point_l.shape[1],
                                   lengths_point_l.shape[2],
                                   )
    

def within_block_l(device_1, device_2):
    """
    distance matrix between all discretization steps inside all blocks
    Return: 
    """
    delta_x = np.array([device_2[i][1] - device_2[i][1].reshape(
        -1, 1) for i in range(device_2.shape[0])])
    delta_y = np.array([device_2[i][0] - device_2[i][0].reshape(
        -1, 1) for i in range(device_2.shape[0])])
    
    lengths_withinblock_l = np.sqrt(delta_x**2 + delta_y**2)
    lengths_withinblock_l = np.repeat(lengths_withinblock_l[:, np.newaxis, :, :], 
                                      lengths_withinblock_l.shape[0],
                                      axis = 1
                                      )
    return lengths_withinblock_l

def distance_matrix_block(device_1, device_2):
    """
    distance matrix between blocks (CML) according to EQ 3 goovaerts, gosh length
    """
    delta_x = np.array([device_1[i][1] - device_2[j][1].reshape(
        -1, 1) for i in range(device_1.shape[0]) for j in range(
            device_2.shape[0])])
    delta_y = np.array([device_1[i][0] - device_2[j][0].reshape(
        -1, 1) for i in range(device_1.shape[0]) for j in range(
            device_2.shape[0])])
    
    lengths_point_l = np.sqrt(delta_x**2 + delta_y**2)    
    lengths_blokkblokk = lengths_point_l.mean(axis=(1, 2)).reshape(
        int(device_1.shape[0]), int(device_2.shape[0]))
    return lengths_blokkblokk
