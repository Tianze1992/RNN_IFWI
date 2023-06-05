#from __future__ import print_function
"""
Implicit Full Waveform Inversion ( for isotropic-elastic ) with an implicit repreesentation neural network.
This module allows three types of network:
    - FWI:          A version full waveform inversion using RNN cell,
                    where each cell acts as a finite-difference operator, 
                    which takes the velocity (could be variable) as input and output shot gather.
    - IRN:          An implicit MLP neural network for image/velocity representation,
                    which takes coordinates as inputs, and output a (normalized) elastic model.
    - IFWI:         coords -> [NN] -> {vel} -> [RNN] -> shot_pred.

    
Main reference paper
(1) "Implicit Seismic Full Waveform Inversion With Deep Neural Representation", 
Jian Sun, Kristopher Innanen, Tianze Zhang, and Daniel Trad, 
Journal of Geophysical Research: Solid Earth, e2022JB025964

(2) "Multilayer Perceptron and Bayesian Neural Network-Based Elastic Implicit Full Waveform Inversion, 
T. Zhang, J. Sun, D. Trad and K. Innanen, 
IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-16, 2023    
"""
import os
import torch
import torch.utils.data
import numpy as np
import torch.nn as nn

from rnn_fd_elastic2 import rnn2D
from generator import gen_Segment2d


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return [repackage_hidden(v) for v in h]

#############################################################################################
# ##                            Implicit Neural Network Model                             ###
#############################################################################################
class IRN(torch.nn.Module):
    '''
    An Implicit Representation Network.
    
    neuron:             a list to define number of layers (length of the list) and number of neurons in each layer;
    activation:         'relu', 'tanh' or 'sine',
                        if 'sine' is chosen, then special initilizations are implemented (see SIREN paper).
    outermost_linear:   if True, then not activation function is applied on the last layer.
    '''
    def __init__(self, 
                 neuron=[2, 256, 256, 256, 256, 3], 
                 omega_0=30, 
                 activation='sine', 
                 bias=True, 
                 outermost_linear=False):
        super(IRN, self).__init__()
        self.omega_0 = omega_0
        self.neuron = neuron
        self.outermost_linear = outermost_linear
        
        self.linear = torch.nn.ModuleList()
        for idx in range(len(neuron)-1):
            self.linear.append(torch.nn.Linear(neuron[idx], neuron[idx+1], bias=bias))
        self.out1 = torch.nn.Linear(neuron[-1], 1)
        self.out2 = torch.nn.Linear(neuron[-1], 1)
        self.out3 = torch.nn.Linear(neuron[-1], 1)

        if activation == 'relu':
            self.omega_0 = 1
            self.activation = torch.nn.ReLU()
        elif activation == 'tanh':
            self.omega_0 = 1
            self.activation = torch.nn.Tanh()
        else:
            self.activation = torch.sin
            self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            self.linear[0].weight.uniform_(-1 / self.neuron[0], 1 / self.neuron[0])
            for ix in range(1, len(self.linear)):
                self.linear[ix].weight.uniform_(-np.sqrt(12 / self.neuron[ix]) / self.omega_0, 
                                                 np.sqrt(12 / self.neuron[ix]) / self.omega_0)
            
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        feature = self.activation(self.omega_0 * self.linear[0](coords))
        for ilayer, layer in enumerate(self.linear[1:]):
            feature = layer(feature)
            if not self.outermost_linear or ilayer + 2 < len(self.linear):
                feature = self.activation(self.omega_0 * feature)
            else:
                feature1 = self.out1(feature)
                feature2 = self.out2(feature)
                feature3 = self.out3(feature)

        return feature1, feature2, feature3, coords


#############################################################################################
# ##                       Implicit Full Waveform Inversion  Model                        ###
#############################################################################################
class IFWI2D():
    def __init__(self, 
                 segment_size,
                 mean1, 
                 std1,
                 mean2, 
                 std2,
                 mean3, 
                 std3,
                 std_scale,
                 neuron, 
                 omega_0, 
                 activation, 
                 bias, 
                 outermost_linear,
                 ns, 
                 nz,
                 nx,
                 zs,
                 xs,
                 zr, 
                 xr,
                 dz,
                 dt,
                 nt,
                 npad, 
                 order, 
                 vmax,
                 vpadding,
                 freeSurface,
                 dtype,
                 device):
        """
        Args:
            segment_size            (int)        ---- the total discrete  number of t
            mean1,                  (float32)    ---- vp mean obtained from well log                         
            std1,                   (float32)    ---- vp std obtained from well log     
            mean2,                  (float32)    ---- vs mean obtained from well log
            std2,                   (float32)    ---- vs std obtained from well log
            mean3,                  (float32)    ---- rho mean obtained from well log
            std3,                   (float32)    ---- rho std obtained from well log
            std_scale,              (float32)    ---- scaling the std in case of the well log does not well represent the target area
            neuron,                 (list of int)---- defining the number of neurons of each NN later
            omega_0,                (float32)    ---- influencing the initialization of the weights        
            activation,             (string)     ---- defining the activation functions of the neural network
            bias,                   (Boolean)    ---- if bias is included in the neural network
            outermost_linear,       (Boolean)    ---- if True, then not activation function is applied on the last layer.
            ns,                     (float32)    ---- total number of the shots
            nz,                     (float32)    ---- total number of the grid points in z direction 
            nx,                     (float32)    ---- total number of the grid points in x direction 
            zs,                     (float32)    ---- shots positions in the z direction on the computational grid
            xs,                     (float32)    ---- shots positions in the x direction on the computational grid
            zr,                     (float32)    ---- receiver positions in the z direction on the computational grid
            xr,                     (float32)    ---- receiver positions in the x direction on the computational grid
            dz,                     (float32)    ---- receiver positions in the z direction on the computational grid
            dt,                     (float32)    ---- receiver positions in the x direction on the computational grid
            nt,                     (float32)    ---- the total discrete  number of t
            npad,                   (int)        ---- number of the PML absorbing layers
            order,                  (int)        ---- order of spatial finite difference
            vmax,                   (float32)    ---- the maximum value of the vs velocity for stable condiction calculation 
            vpadding,               (Boolean)    ---- if the elastic model needs to be padded
            freeSurface,            (Boolean)    ---- if we need the freeSurface modeling condition
            dtype,                  (dtype)      ---- the default datatype
            device,                 (device)     ---- CPU or GPU
        """
        super(IFWI2D, self).__init__()

        self.std1  = std1
        self.mean1 = mean1
        self.std2  = std2
        self.mean2 = mean2
        self.std3  = std3
        self.mean3 = mean3
        self.std_scale = std_scale
        self.nz = nz
        self.nx = nx
        self.neuron= neuron
        self.omega_0=omega_0
        self.activation=activation
        self.bias=bias
        self.outermost_linear=outermost_linear
        self.ns=ns
        self.zs=zs
        self.xs=xs
        self.zr=zr 
        self.xr=xr
        self.dz=dz
        self.dt=dt
        self.nt = nt
        self.npad = npad
        self.order = order
        self.vmax=vmax
        self.vpadding=vpadding
        self.freeSurface=freeSurface
        self.dtype=dtype
        self.device = device
        self.segment_size = segment_size

        # setting up the coordinarte 
        self.nx_pad = self.nx + 2 * self.npad
        self.nz_pad = self.nz + self.npad if self.freeSurface else self.nz + 2 * self.npad
        self.Costfunction = torch.nn.MSELoss(reduction='sum')
        print("this is the the npad", self.npad)
        self.x = np.arange(0, self.nx) * self.dz  / 1000
        self.z = np.arange(0, self.nz) * self.dz  / 1000
        self.X, self.Z = np.meshgrid(self.x[None,:], self.z[:,None])
        self.X = torch.from_numpy(self.X).type(dtype = self.dtype).to(self.device)
        self.Z = torch.from_numpy(self.Z).type(dtype = self.dtype).to(self.device)
        self.coords = torch.stack([self.X, self.Z], dim=-1).to(self.device)[None, :]  # shape [1, nz, nx, 2]
        self.t = self.dt * torch.arange(0, self.nt, dtype=self.dtype)                 # create time vector
        
        self.rnn = rnn2D( self.nz, self.nx, self.zs, self.xs, self.zr, self.xr, self.dz, self.dt, self.npad, self.order, self.vmax, self.freeSurface, self.dtype, self.device).to(self.device)
        # defining the forward modeling engine
        self.vel_net = IRN(neuron=self.neuron, omega_0=self.omega_0, activation=self.activation, outermost_linear=self.outermost_linear).to(self.device)
        # defining the NN for generating elastic models
        
    def train(self, 
              MaxIter, 
              wavelet=None, 
              shots=None, 
              option=0, 
              log_interval=1, 
              resume_file_name=None):

        params = self.vel_net.parameters() 
        # defining the network weights as the parameters
        optimizer0 = torch.optim.Adam(lr=(3e-5),params = params)
          
        resume_from_epoch = 0
        train_loss_history = []
        
        
        
        #defining the list which record the training history 
        vmodel1_list = [] 
        vmodel2_list = []
        vmodel3_list = []
        for epoch in range(resume_from_epoch, MaxIter):
            loss, vmodel1, vmodel2, vmodel3 = self.train_one_epoch(optimizer0, wavelet, shots, option)
            
            if epoch % log_interval == 0 or epoch == MaxIter - 1:
                print("Epoch: {:5d}, Loss: {:.4e}".format(epoch, loss.item()))
            vmodel1_list.append(vmodel1.squeeze().cpu().detach().numpy())
            vmodel2_list.append(vmodel2.squeeze().cpu().detach().numpy())
            vmodel3_list.append(vmodel3.squeeze().cpu().detach().numpy())
            train_loss_history.append(loss.cpu().detach().numpy())

            torch.save(train_loss_history, './saved_files/train_loss_history.pt')
            torch.save(vmodel1_list,       './saved_files/vmodel1_list.pt')
            torch.save(vmodel2_list,       './saved_files/vmodel2_list.pt')
            torch.save(vmodel3_list,       './saved_files/vmodel3_list.pt')
        return train_loss_history

    def train_one_epoch(self, optimizer0, wavelet=None, shots=None, option=0):
        '''
        (1) Obtaining the elastic parameters, and the synthetic data
        (2) Calculate objective function 
        (3) backward propagating the residual and update the weights in NN
        '''
        shots = shots.to(self.device)
        loss = 0
        for iseg, (segWavelet, segData) in enumerate(gen_Segment2d(wavelet, shots, segment_size=self.segment_size, option=option)):
            optimizer0.zero_grad()
            # step (1)
            vx_save, vz_save, txx_save, tzz_save, txz_save, segment_ytPred_x,segment_ytPred_z, vmodel1, vmodel2, vmodel3  = self.forward_process(segWavelet, option) 

            vx_save = repackage_hidden(vx_save)
            vz_save = repackage_hidden(vz_save)
            txx_save = repackage_hidden(txx_save)
            tzz_save = repackage_hidden(tzz_save)
            txz_save = repackage_hidden(txz_save)
            
            # step (2)
            shots_pred= torch.cat((segment_ytPred_x.reshape(1, len(self.xs),len(self.t), len(self.xr)),segment_ytPred_z.reshape(1, len(self.xs),len(self.t), len(self.xr))),dim=0)
            loss_Seg = self.Costfunction(shots_pred, segData)

            # step (3)
            loss_Seg.backward()
            optimizer0.step()
            loss += loss_Seg.detach()

        return loss.cpu().detach(), vmodel1, vmodel2, vmodel3 
            
    def forward_process(self, wavelet=None,option=0):
        '''
        (1) using the NN to generate elastic models, vmodel1, vmodel2, and vmodel3, using the coordinate information as input
        (2) scaling the output of the NN properly, with mean, and std obtained from the well log 
        (3) feed the RNN the elastic parameters and generate the synthetic data
        '''
        # step (1)
        vmodel1, vmodel2, vmodel3, _ = self.vel_net(self.coords)
        
        # step (2)
        vmodel1 = vmodel1* (self.std1*self.std_scale) + self.mean1
        vmodel1 = torch.reshape(vmodel1,(1,self.nz,self.nx))*1000
        vmodel2 = vmodel2* (self.std2*self.std_scale) + self.mean2
        vmodel2 = torch.reshape(vmodel2,(1,self.nz,self.nx))*1000
        vmodel3 = vmodel3* (self.std3*self.std_scale) + self.mean3
        vmodel3 = torch.reshape(vmodel3,(1,self.nz,self.nx))*1000
        
        
        # step (3)
        vx_save, vz_save, txx_save, tzz_save, txz_save, \
        segment_ytPred_x,segment_ytPred_z,\
        _, _, _, _ = self.rnn(vmodel1, vmodel2, vmodel3, wavelet, option)
        
        
        return vx_save, vz_save, txx_save, tzz_save, txz_save, segment_ytPred_x,segment_ytPred_z, vmodel1, vmodel2, vmodel3 
    
    def predict(self):

        vmodel1, vmodel2, vmodel3, _ = self.vel_net(self.coords)
        vmodel1 = vmodel1* self.std1*self.std_scale + self.mean1
        vmodel1 = torch.reshape(vmodel1,(1,self.nz,self.nx))*1000
        vmodel2 = vmodel2 * self.std2*self.std_scale + self.mean2
        vmodel2 = torch.reshape(vmodel2,(1,self.nz,self.nx))*1000
        vmodel3 = vmodel3 * self.std3*self.std_scale + self.mean3
        vmodel3 = torch.reshape(vmodel3,(1,self.nz,self.nx))*1000
        
        return vmodel1, vmodel2, vmodel3
