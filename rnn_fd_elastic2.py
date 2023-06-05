"""
Recurrent Neural Network (RNN) where each cell is Finite Difference (FD) operator.

Please cite this reference: 
Jian Sun, Zhan Niu, Kristopher A. Innanen, Junxiao Li, and Daniel O. Trad, (2020), 
"A theory-guided deep-learning formulation and optimization of seismic waveform inversion," 
GEOPHYSICS 85: R87-R99.

@author: jiansun
- Penn State
- Acoustic FWI By Jian on Feb. 6, 2020
@author: Tianze Zhang
- Updated Elastic RNN By Tianze Zhang on Sep. 28, 2021
- University of calgary
"""
import numpy as np
import torch
import torch.nn.functional as F
import math

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                     2D propagator (single time step)                                                   """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """


class Propagator2D(torch.nn.Module):
    """
    This class only performs a single time-step wave propagation,
         where, num_vels is considered as batch_size;
                num_shots is considered as channels.
    """
    def __init__(self, nz, nx, dz, dt, 
                 npad=0, order=2, 
                 freeSurface=True, 
                 dtype=torch.float32, device='cpu'):
        super(Propagator2D, self).__init__()
        self.dtype = dtype
        self.device = device
        self.k_size = order + 1
        self.nz, self.nx, self.npad = nz, nx, npad
        self.nx_pad = nx + 2 * npad
        self.freeSurface = freeSurface
        if freeSurface:
            self.nz_pad = nz + npad
        else:
            self.nz_pad = nz + 2 * npad
        self.dt = dt
        self.vmax = 4000
            
        # Setup convolutional kernel (default: 2nd-order 3*3)
        dx = dz
        if order == 4:
            # Laplacian kernel for 4th-order FD forward propagation.
            self.kernel2D = self.___make_tensor([[0, 0, -(dt / dz)**2 / 12, 0, 0],
                                                [0, 0, (dt / dz)**2 / 12 * 16, 0, 0],
                                                [-(dt / dx)**2 / 12, (dt / dx)**2 / 12 * 16, 
                                                 -30 / 12 * ((dt / dx)**2 + (dt / dz)**2), 
                                                 (dt / dx)**2 / 12 * 16, -(dt / dx)**2 / 12],
                                                [0, 0, (dt / dz)**2 / 12 * 16, 0, 0],
                                                [0, 0, -(dt / dz)**2 / 12, 0, 0]])
        else:
            # Laplacian kernel for 2nd-order FD forward propagation (default).
            self.kernel2D = self.___make_tensor([[0, (dt / dz)**2, 0],
                                                [(dt / dx)**2, 
                                                 -2 * ((dt / dx)**2 + (dt / dz)**2), 
                                                 (dt / dx)**2],
                                                [0, (dt / dz)**2, 0]])
            #Laplacian kernel for 2nd-order FD forward propagation (default).
            self.kernelz2D = self.___make_tensor([[0,  0, 0, 0,0],
                                          [0,  0, (1/24)/dz,  0,0],
                                          [0,  0,  (-9/8)/dz,  0,0],
                                          [0,  0,  (9/8)/dz, 0,0],
                                          [0,  0,  (-1/24)/dz,    0,0]])
            self.kernelx2D = self.___make_tensor([[   0,   0,  0,      0,0],
                                          [   0,   0,  0,      0,0],
                                          [0, (1/24)/dx,(-9/8)/dx, (9/8)/dx, (-1/24)/dx],
                                          [   0,   0,  0,      0,0],
                                          [   0,   0,  0,      0,0]])
            self.kernelz2D_U = self.___make_tensor([[0,  0, (1/24)/dz, 0,0],
                                           [0,  0, (-9/8)/dz,  0,0],
                                           [0,  0,  (9/8)/dz,  0,0],
                                           [0,  0,  (-1/24)/dz, 0,0],
                                           [0,  0,  0,    0,0]])
            self.kernelx2D_U = self.___make_tensor([[   0,   0,  0,      0,0],
                                           [   0,   0,  0,      0,0],
                                           [(1/24)/dx,(-9/8)/dx, (9/8)/dx, (-1/24)/dx,0],
                                           [   0,   0,  0,      0,0],
                                           [   0,   0,  0,      0,0]])
        
        # #### Make kernelX and kernelZ for regularizer ####
        self.kernelX = self.___make_tensor([[-1 / 2, 0, 1 / 2]])
        self.kernelZ = self.___make_tensor([[-1 / 2], [0], [1 / 2]])

        self.ax, self.az, self.axxzz = self.PML_bcMask(4000, self.nz_pad, self.nx_pad, npad, dz, dx)
        self.ax = self.___make_tensor(self.ax)
        self.az = self.___make_tensor(self.az)
            
    def ___make_tensor(self, a):
        # Prepare a filter in shape of [1, 1, Height, Width]
        a = np.asarray(a)
        a = a.reshape([1, 1] + list(a.shape))
        return torch.as_tensor(a, dtype=self.dtype, device=self.device)
        
    def __sponge_bcMask(self):
        # Keep tmp as numpy array, because tensor does not support negative step like [::-1]
        tmp = np.exp(-0.0015**2 * np.arange(self.npad, 0, -1, dtype=np.float32)**2)  # small -- > large
        wmask = np.ones([self.nz_pad, self.nx_pad], dtype=np.float32)
        # add bottom_mask
        wmask[-self.npad:, self.npad:-self.npad] *= tmp[::-1][:, None]
        if self.freeSurface is False:
            # add top_mask
            wmask[:self.npad, self.npad:-self.npad] *= tmp[:, None]
        # add left_mask
        wmask[:, :self.npad] *= tmp[None, :]
        # add right_mask
        wmask[:, -self.npad:] *= tmp[::-1][None, :]
        return torch.as_tensor(wmask, dtype=self.dtype, device=self.device)
    
    def PML_bcMask(self, vmax, nz_pad, nx_pad, npad, dz, dx):
        ax = torch.as_tensor(np.ones([self.nz_pad, self.nx_pad]))
        az = torch.as_tensor(np.ones([self.nz_pad, self.nx_pad]))
        R = 10**(-5)
        for ind_z in range (0, self.nz_pad):
            for ind_x in range (0, self.npad):
                ax[ind_z,ind_x] = -math.log(R)*3*self.vmax*(self.npad-ind_x-1)**2/(2*(dx*self.npad)**2) 

        for ind_z in range (0,self.nz_pad):
            for ind_x in range (self.nx+self.npad,self.nx+2*self.npad):
                ax[ind_z,ind_x] = -math.log(R)*3*self.vmax*(ind_x-self.nx-self.npad)**2/(2*(dx*self.npad)**2) 
        if self.freeSurface:
            for ind_x in range (0,self.nx_pad):
                for ind_z in range (0,self.npad):
                    az[ind_z,ind_x] = az[ind_z,ind_x]
        else:
            for ind_x in range (0,self.nx_pad):
                for ind_z in range (0,self.npad):
                    az[ind_z,ind_x] = -math.log(R)*3*self.vmax*(self.npad-ind_z-1)**2/(2*(dx*self.npad)**2)
        
        
        if self.freeSurface:
          for ind_x in range (0,self.nx_pad):
              for ind_z in range (self.nz,self.nz+self.npad):
                  az[ind_z,ind_x] = -math.log(R)*3*self.vmax*(ind_z-self.nz)**2/(2*(dx*self.npad)**2)
        else:
          for ind_x in range (0,self.nx_pad):
              for ind_z in range (self.nz+self.npad,self.nz+2*self.npad):
                  az[ind_z,ind_x] = -math.log(R)*3*self.vmax*(ind_z-self.nz-self.npad)**2/(2*(dx*self.npad)**2)
        wmask = ax + az
        return ax, az, wmask
    
    def ___tensor_pad(self, input_tensor):
        #print("this is the shape of the input XXXXXX",input_tensor.shape)
        """
        This function is to padding velocity tensor for implementing the absorbing boudary condition.
            input_tensor: is a 4D tensor, shape=[batch_size, 1, nz, nx]
            output_tensor: is also a 4D tensor, shape=[batch_size, 1, nz_pad, nx_pad]
        """   
        batch_size = input_tensor.shape[0]
        if self.freeSurface:
            vpadTop = input_tensor
        else:
            vtop = torch.ones((batch_size, 1, self.npad, self.nx), dtype=self.dtype, device=self.device) * input_tensor[:, :, :1, :]
            vpadTop = torch.cat((vtop, input_tensor), -2)  # padding on axis=2 (nz)
        
        vbottom = torch.ones((batch_size, 1, self.npad, self.nx), dtype=self.dtype, device=self.device) * input_tensor[:, :, -1:, :]
        vpadBottom = torch.cat([vpadTop, vbottom], -2)  # padding on axis=2 (nz)

        vleft = torch.ones((batch_size, 1, self.nz_pad, self.npad), dtype=self.dtype, device=self.device) * vpadBottom[:, :, :, :1]
        vpadLeft = torch.cat([vleft, vpadBottom], -1)  # padding on axis=3 (nx)

        vright = torch.ones((batch_size, 1, self.nz_pad, self.npad), dtype=self.dtype, device=self.device) * vpadBottom[:, :, :, -1:]
        output_tensor = torch.cat([vpadLeft, vright], -1)  # padding on axis=3 (nx)
        return output_tensor
        
    def ___step_rnncell(self, source, vx, vz, vx_x, vx_z, vz_x, vz_z,txx, tzz, txz, txx_x, txx_z, tzz_x, tzz_z, txz_x, txz_z):
        """
        This function is to implement the forward propagation for a single time-step.
        Input:
            u_prev: state at time step t-dt,   in shape of [self.num_vels, self.num_shots, self.nz_pad, self.nx_pad]
            u_:     state at time step t, also in shape of [self.num_vels, self.num_shots, self.nz_pad, self.nx_pad]

            source: a list of tensor: [fs, zs, xs]; 
                    where fs: [self.num_vels], 
                          zs: [self.num_vels, self.num_shots],
                          xs: [self.num_vels, self.num_shots]
        Output: 
            u_next: state at time step t+dt, in shape of [self.num_vels, self.num_shots, self.nz_pad, self.nx_pad]
            u_partial = F.conv2d(u_, self.kernel2D.repeat(u_.shape[1], 1, 1, 1), padding=(self.k_size - 1) // 2, groups=u_.shape[1])
        """        
        # For conv2d (squared filter): 
        #       out_size = [in_size + 2*pad_size - kernel_size - (kernel_size-1)*(dilation-1)]/stride + 1
        # To achieve "same", the pad_size we need is: (dilation=1, stride=1)
        #       padding=(self.k_size-1)/2
        if self.npad != 0:
            wmask = self.__sponge_bcMask()[None, None, :, :]
        '''
        for ivel in range(1):
            for ishot in range(vx.shape[1]):
                if self.freeSurface:
                    txx[ivel, ishot, source[1][ivel, ishot], source[2][ivel, ishot] + self.npad] += source[0][ivel]
                    #print("this is the source",source[0][ivel])
                    tzz[ivel, ishot, source[1][ivel, ishot], source[2][ivel, ishot] + self.npad] += source[0][ivel]
                else:
                    txx[ivel, ishot, source[1][ivel, ishot] + self.npad, source[2][ivel, ishot] + self.npad] += source[0][ivel]
                    tzz[ivel, ishot, source[1][ivel, ishot] + self.npad, source[2][ivel, ishot] + self.npad] += source[0][ivel]
        '''
        for im in range(vx.shape[1]): # adding source for each shot record im represent the shot index  
            if self.freeSurface:
                #txx[:, im, source[1][0][im], source[1][1][im]+self.npad] += source[0]  
                tzz[:, im, source[1][0][im], source[1][1][im]+self.npad] += source[0]
            else:
                #txx[:, im, source[1][0][im]+self.npad, source[1][1][im]+self.npad] += source[0]  
                tzz[:, im, source[1][0][im]+self.npad, source[1][1][im]+self.npad] += source[0]


        var_txx_x = F.conv2d(txx, self.kernelx2D.repeat(txx.shape[1], 1, 1, 1), \
        padding=2, groups=txx.shape[1])/self.rho_tensor
        
        var_txz_z = F.conv2d(txz, self.kernelz2D_U.repeat(txz.shape[1], 1, 1, 1), \
        padding=2, groups=txx.shape[1])/self.rho_tensor
        
        var_txz_x = F.conv2d(txz, self.kernelx2D_U.repeat(txz.shape[1], 1, 1, 1), \
        padding=2, groups=txx.shape[1])/self.rho_tensor
        
        var_tzz_z = F.conv2d(tzz, self.kernelz2D.repeat(tzz.shape[1], 1, 1, 1), \
        padding=2, groups=txx.shape[1])/self.rho_tensor

        vx_x = (1 - self.dt * self.ax) * vx_x + self.dt * var_txx_x  # the vx in x direction 
        vx_z = (1 - self.dt * self.az) * vx_z + self.dt * var_txz_z  # the vx in z direction 
        
        vx = vx_x + vx_z
        
        vz_x = (1 - self.dt * self.ax) * vz_x + self.dt * var_txz_x # the vz in x direction 
        vz_z = (1 - self.dt * self.az) * vz_z + self.dt * var_tzz_z # the vz in z direction 
        
        vz = vz_x + vz_z 
        
        var_vx_x = F.conv2d(vx, self.kernelx2D_U.repeat(vx.shape[1], 1, 1, 1), \
        padding=2, groups=txx.shape[1])
        
        var_vx_z = F.conv2d(vx, self.kernelz2D.repeat(vx.shape[1], 1, 1, 1), \
        padding=2, groups=txx.shape[1])
        
        var_vz_x = F.conv2d(vz, self.kernelx2D.repeat(vz.shape[1], 1, 1, 1), \
        padding=2, groups=txx.shape[1])
        
        var_vz_z = F.conv2d(vz, self.kernelz2D_U.repeat(vx.shape[1], 1, 1, 1), \
        padding=2, groups=txx.shape[1])

        txx_x = (1 - self.dt * self.ax) * txx_x + \
        self.dt * self.vp_tensor * self.vp_tensor * self.rho_tensor * var_vx_x  # the txx in x direction
        
        txx_z = (1 - self.dt * self.az) * txx_z + \
        self.dt * (self.vp_tensor * self.vp_tensor * self.rho_tensor - \
               2 * self.vs_tensor * self.vs_tensor * self.rho_tensor) * var_vz_z  # the txx in x direction
        txx = txx_x + txx_z # the final txx 

        tzz_x = (1 - self.dt * self.ax) * tzz_x + \
            self.dt * (self.vp_tensor * self.vp_tensor * self.rho_tensor - \
                   2 * self.vs_tensor * self.vs_tensor * self.rho_tensor) * var_vx_x  # the tzz in x direction
        tzz_z = (1 - self.dt * self.az) * tzz_z + \
            self.dt *  self.vp_tensor * self.vp_tensor * self.rho_tensor * var_vz_z  # the tzz in x direction
        tzz = tzz_x + tzz_z # the final tzz

        txz_x = (1 - self.dt * self.ax) * txz_x + \
        self.dt * self.vs_tensor * self.vs_tensor* self.rho_tensor* var_vx_z  # the tzz in x direction
        txz_z = (1 - self.dt * self.az) * txz_z + \
        self.dt * self.vs_tensor * self.vs_tensor * self.rho_tensor * var_vz_x  # the tzz in x direction
        txz = txz_x + txz_z # the final tzz

        return vx, vz, vx_x, vx_z, vz_x, vz_z, txx, tzz, txz,txx_x, txx_z, tzz_x, tzz_z, txz_x, txz_z, \
        self.vp_tensor,self.vs_tensor,self.rho_tensor, self.ax, self.az  

    def forward(self, vp_tensor_noPad,vs_tensor_noPad,rho_tensor_noPad, \
    segment_sources,vx,vz,vx_x,vx_z,vz_x,vz_z,txx,tzz,txz,txx_x,txx_z,tzz_x,tzz_z,txz_x,txz_z):
        """
        Forward propagating for a single time-step from (t-dt & t --> t+dt).
        Input:
            sources_info: a list contains [wavelet, zs, xs, zr, xr],
                          wavelet(tensor): shape [num_vels]
                          zs(tensor): shape [num_vels, num_shots]
                          xs(tensor): shape [num_vels, num_shots]
                          zr(tensor): shape [num_vels, num_shots, num_receivers]
                          xr(tensor): shape [num_vels, num_shots, num_receivers]
            Initial states (tensor):
                        1. prev_wavefield(tensor): wavefield at time step t-dt, 
                        2. curr_wavefield(tensor): wavefield at time step t
                            size: [num_vels, num_shots, self.nz_pad, , self.nx_pad]
            vel_noPad: A PyTorch tensor for velocity model with grid interval dz=dx, 
                        - shape = [num_vels, nz, nx]. 
                        - Initial velocity model for inversion process, i.e., requires_grad=True.
                        - True velocity model for forward modeling propagation.
        Output:
            Save prev_wavefield & curr_wavefield for next time step prop.
            yt_pred: [num_vels, num_shots, num_receivers]
                extracting seismogram at receiver locations using [zr, xr].
        """
        # Padding the velocity model (Should repadding velocity every time for absorbing)
        #print("==========>",self.nz, self.nx)
        self.vp_tensor = self.___tensor_pad(vp_tensor_noPad.reshape(1,1,self.nz,self.nx))  # After shape: [num_vels, 1, nz_pad, nx_pad]
        self.vs_tensor = self.___tensor_pad(vs_tensor_noPad.reshape(1,1,self.nz,self.nx))  # After shape: [num_vels, 1, nz_pad, nx_pad]
        self.rho_tensor = self.___tensor_pad(rho_tensor_noPad.reshape(1,1,self.nz,self.nx))  # After shape: [num_vels, 1, nz_pad, nx_pad]
        #print("this is the shape of the self vp_tensor",self.vp_tensor.shape)
        '''
        if isinstance(sources_info[3], int):
            num_receivers = self.nx
        else:
            num_receivers = sources_info[4].shape[2]
        row = torch.arange(vx.shape[0])[:, None, None].repeat([1, vx.shape[1], num_receivers])
        col = torch.arange(vx.shape[1])[None, :, None].repeat([vx.shape[0], 1, num_receivers])
        #print("Debug Check 1: row{}, col: {}, vel_noPad: {}".format(row.shape, col.shape, vel_noPad.shape))
        #print("Debug Check 1: prev_wavefield: {}, curr_wavefield: {}, vel_noPad: {}".format(prev_wavefield.shape, curr_wavefield.shape, vel_noPad.shape))
        '''
        vx, vz, vx_x, vx_z, vz_x, vz_z, txx, tzz, txz,txx_x, txx_z, tzz_x, tzz_z, txz_x, txz_z, \
        self.vp_tensor,self.vs_tensor,self.rho_tensor, \
        ax, az = \
        self.___step_rnncell(segment_sources,vx,vz,vx_x,vx_z,vz_x,vz_z,txx,tzz,txz,txx_x,txx_z,tzz_x,tzz_z,txz_x,txz_z)

        '''
        if self.freeSurface:
            depth_index = sources_info[3]
        else:
            depth_index = sources_info[3] + self.npad
        if isinstance(sources_info[3], int):
            yt_pred_x = vx[:, :, sources_info[3], self.npad:self.npad + num_receivers].reshape(list(sources_info[1].shape) + [-1])
            print("=========>",yt_pred_x.shape)
            yt_pred_z = vz[:, :, sources_info[3], self.npad:self.npad + num_receivers].reshape(list(sources_info[1].shape) + [-1])
        else:
            yt_pred_x = vx[row.view(-1), col.view(-1), depth_index.view(-1), sources_info[4].view(-1) + self.npad].\
            reshape(list(sources_info[1].shape) + [-1])
            yt_pred_z = vz[row.view(-1), col.view(-1), depth_index.view(-1), sources_info[4].view(-1) + self.npad].\
            reshape(list(sources_info[1].shape) + [-1])
        '''
        if self.freeSurface:
                yt_pred_x  = vx[0, :, [zr for zr in segment_sources[2][0]], [xr+self.npad-1 for xr in segment_sources[2][1]] ]
                yt_pred_z  = vz[0, :, [zr for zr in segment_sources[2][0]], [xr+self.npad-1 for xr in segment_sources[2][1]] ]
                yt_pred_xz = txz[0, :, [zr for zr in segment_sources[2][0]], [xr+self.npad-1 for xr in segment_sources[2][1]] ]
        else:
            yt_pred_x  = vx[0, :,  [zr+self.npad-1 for zr in segment_sources[2][0]], [xr+self.npad-1 for xr in segment_sources[2][1]] ]
            yt_pred_z  = vz[0, :,  [zr+self.npad-1 for zr in segment_sources[2][0]], [xr+self.npad-1 for xr in segment_sources[2][1]] ]
            yt_pred_xz = txz[0, :, [zr+self.npad-1 for zr in segment_sources[2][0]], [xr+self.npad-1 for xr in segment_sources[2][1]] ]
        
        # Add Regularization term, shape: [num_vels, 1]
        # 1. for sparse model
        # regularizer = vel_noPad.pow(2).sum(dim=-1)  
        # 2.1 for flatest model (1D derivative)
        # gradVel = vel_noPad[:, :, 1:] - vel_noPad[:, :, :-1]
        # regularizer = gradVel.pow(2).sum(dim=-1)  
        # 2.2 for flatest model (2D derivatives)
        # gradVelZ = F.conv1d(vel_noPad, self.kernelZ, padding=0)
        # regularizer = gradVelZ.pow(2).sum(dim=-1)
        regularizer = torch.tensor([[0]], dtype=self.dtype, device=self.device)
        
        #yt_pred = None
        #regularizer  = None
        return vx, vz, vx_x, vx_z, vz_x, vz_z, txx, tzz, txz,txx_x, txx_z, tzz_x, tzz_z, txz_x, txz_z,\
        yt_pred_x, yt_pred_z,regularizer, self.vp_tensor,self.vs_tensor,self.rho_tensor, ax, az


""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                     2D RNN forward modeling (full/truncated time step)                                 """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """


class rnn2D(torch.nn.Module):
    """
    Using Propagator1D to perform the forward modeling with given wavelets through multiple time-steps
    """
    def __init__(self, nz, nx, zs, xs, zr, xr, dz, dt, 
                 npad=0, order=2, vmax=6000, 
                 freeSurface=True, 
                 dtype=torch.float32, device='cpu'):
        super(rnn2D, self).__init__()
        """
        zs(tensor): in shape of [num_vels, num_shots]
        xs(tensor): in shape of [num_vels, num_shots]
        zr(tensor): in shape of [num_vels, num_shots, num_receivers]
        xr(tensor): in shape of [num_vels, num_shots, num_receivers]
        """
        # Stability condition for forward modeling (Lines et al., 1999, Geophysics)
        # vmax*dt/dz < XXX
        # for 1D, 2nd-order: 1; 4th-order: np.sqrt(3)/2
        # for 2D, 2nd-order: 1/np.sqrt(2); 4th-order: np.sqrt(3/8)
        # for 3D, 2nd-order: 1/np.sqrt(3); 4th-order: 1/2
        #if order == 2:
        #    # print("Stability Condition: {} < {}".format(vpred.max().cpu() * dt / dz, 1))
        #    assert vmax * dt / dz < 1 / np.sqrt(2), "Current parameters setting do NOT meet the stability condition."
        #elif order == 4:
        #    # print("Stability Condition: {} < {}".format(vpred.max().cpu() * dt / dz, np.sqrt(3) / 2))
        #    assert vmax * dt / dz < np.sqrt(3 / 8), "Current parameters setting do NOT meet the stability condition."
        self.dtype = dtype
        self.device = device
        self.zs, self.xs = zs, xs
        self.zr, self.xr = zr, xr
        self.nx_pad = nx + 2 * npad
        if freeSurface:
            self.nz_pad = nz + npad
        else:
            self.nz_pad = nz + 2 * npad
        self.num_shots = len(self.xs)
            
        # define the finite-difference operator for each time step
        self.fd = Propagator2D(nz, nx, dz, dt, npad, order, freeSurface, dtype, device).to(device)

    def forward(self, vp_tensor,vs_tensor,rho_tensor, segment_wavelet, option=0):
        """
        Input:
            vmodel(tensor): require_grad=True for training RNN, shape: [num_vels, nz, nx]
            prev_state(tensor): 4D tensor, shape: [num_vels, num_shots, nz_pad, nx_pad]
            curr_state(tensor): 4D tensor, shape: [num_vels, num_shots, nz_pad, nx_pad]
            segment_wavelet(tensor): [num_vels, len_tSeg] or [len_tSeg]
        Ouput:
            prev_state & curr_state for next time-segment, which is depending on option you choose.
            segment_ytPred(tensor): [num_vels, num_shots, nt(len_tSeg), num_receivers]
        option:
            option only affects the returned prev_state & curr_state, 
            which is related to the segment option for data and wavelet in gen_Segment2d(xxx, option=0);
            where,
                option=0, the returned prev_state & curr_state are wavefileds at two last time steps of the current time-segment;
                option=1, the returned prev_state & curr_state are wavefileds at the midterm of the current time-segment;
                option=2, the returned prev_state & curr_state are zero-initialized wavefields (time_step=0).
        """

        vx =  torch.zeros([1,   self.num_shots, self.nz_pad, self.nx_pad], 
                                  dtype=self.dtype, device=self.device)
        vz =  torch.zeros([1,   self.num_shots, self.nz_pad, self.nx_pad], 
                                  dtype=self.dtype, device=self.device)
        vx_x =  torch.zeros([1, self.num_shots, self.nz_pad, self.nx_pad], 
                                  dtype=self.dtype, device=self.device)
        vx_z =  torch.zeros([1, self.num_shots, self.nz_pad, self.nx_pad], 
                                  dtype=self.dtype, device=self.device)
        vz_x =  torch.zeros([1, self.num_shots, self.nz_pad, self.nx_pad], 
                                  dtype=self.dtype, device=self.device)
        vz_z =  torch.zeros([1, self.num_shots, self.nz_pad, self.nx_pad], 
                                  dtype=self.dtype, device=self.device)
        txx = torch.zeros([1,   self.num_shots, self.nz_pad, self.nx_pad], 
                                  dtype=self.dtype, device=self.device)
        txx_x = torch.zeros([1, self.num_shots, self.nz_pad, self.nx_pad], 
                                  dtype=self.dtype, device=self.device)
        txx_z = torch.zeros([1, self.num_shots, self.nz_pad, self.nx_pad], 
                                  dtype=self.dtype, device=self.device)
        tzz = torch.zeros([1,   self.num_shots, self.nz_pad, self.nx_pad], 
                                  dtype=self.dtype, device=self.device)
        tzz_x = torch.zeros([1, self.num_shots, self.nz_pad, self.nx_pad], 
                                  dtype=self.dtype, device=self.device)
        tzz_z = torch.zeros([1, self.num_shots, self.nz_pad, self.nx_pad], 
                                  dtype=self.dtype, device=self.device)
        txz = torch.zeros([1,   self.num_shots, self.nz_pad, self.nx_pad], 
                                  dtype=self.dtype, device=self.device)
        txz_x = torch.zeros([1, self.num_shots, self.nz_pad, self.nx_pad], 
                                  dtype=self.dtype, device=self.device)
        txz_z = torch.zeros([1, self.num_shots, self.nz_pad, self.nx_pad], 
                                  dtype=self.dtype, device=self.device)

        #print("Debug Check 2: \n Vx shape: {} \n ,Vz shape: {} \n".format(vx.shape, vz.shape))
        #print("Debug Check 3: \n txx shape: {} \n ,tzz shape: {} \n, txz shape: {} \n".format(txx.shape, tzz.shape,txz.shape))

        # if segment_wavelet.ndim == 1:
        if len(segment_wavelet.shape) == 1:
            segment_wavelet = segment_wavelet.repeat(1, 1)
        
        segment_ytPred_x = []
        segment_ytPred_z = []
        avg_regularizer = []
        for it in range(segment_wavelet.shape[1]):
            segment_sources = [segment_wavelet[:, it], [self.zs, self.xs], [self.zr, self.xr]]
            #print("this is the segmented wavelet",segment_wavelet[:, it])
            #print("this is the zs",self.zs)
            #print("this is the zs[:vmodel.shape[0], :]",self.zs[:vmodel.shape[0], :])
            #print("this is zr.shape[0]",self.zr)
            #print(segment_sources[0], \
            #segment_sources[1], \
            #segment_sources[2], \
            #segment_sources[3], \
            #segment_sources[4])
            vx, vz, vx_x, vx_z, vz_x, vz_z, txx, tzz, txz,txx_x, txx_z, tzz_x, tzz_z, txz_x, txz_z,\
            seg_ytPred_x, seg_ytPred_z, regularizer, self.vp_tensor,self.vs_tensor,self.rho_tensor, \
            ax,az = \
            self.fd(vp_tensor,vs_tensor,rho_tensor, \
                    segment_sources, vx,vz,vx_x, vx_z, vz_x, vz_z,txx, tzz, txz, txx_x, txx_z, tzz_x, tzz_z, txz_x, txz_z)
            segment_ytPred_x.append(seg_ytPred_x)
            segment_ytPred_z.append(seg_ytPred_z)
            avg_regularizer.append(regularizer)
            # for option 1, we want save the middle states for next time segement
            if option == 1 and it == (len(segment_sources[0]) - 1) // 2:
                vx_save  = vx.detach().clone()  
                vz_save  = vz.detach().clone()
                txx_save = txx.detach().clone()  
                tzz_save = tzz.detach().clone()
                txz_save = txz.detach().clone()   

        # for option 0, we save the last two wavefields for next time-segment
        if option == 0:
              vx_save  = vx
              vz_save  = vz
              txx_save = txx 
              tzz_save = tzz
              txz_save = txz
        # for next time-segement, option 2 makes sure it always start from time_step 0 with zero initials
        elif option == 2:
            prev_save = prev_state.new_zeros(prev_state.shape, dtype=self.dtype, device=prev_state.device)
            curr_save = curr_state.new_zeros(curr_state.shape, dtype=self.dtype, device=curr_state.device)
        
        segment_ytPred_x = torch.stack(segment_ytPred_x, dim=-2)  # shape: [num_vels, num_shots, nt(len_tSeg), num_receivers]
        segment_ytPred_z = torch.stack(segment_ytPred_z, dim=-2)  # shape: [num_vels, num_shots, nt(len_tSeg), num_receivers]
        avg_regularizer = torch.stack(avg_regularizer, dim=-1).mean(dim=-1)  # shape:[num_vels]
        
        #prev_save, curr_save, segment_ytPred, avg_regularizer = None, None, None, None
        
        return vx_save, vz_save, txx_save, tzz_save, txz_save, \
               segment_ytPred_x,segment_ytPred_z,\
               avg_regularizer, self.vp_tensor,self.vs_tensor,self.rho_tensor

