"""
Generator for wavelet or segements.

@author: jiansun
"""
import os
import torch
import numpy as np
import torch.nn.functional as F
from math import exp
from PIL import Image
from scipy.special import binom


#############################################################################################
# ##                   Differentiable structural similarity (SSIM) index                  ###
# ##        from https://github.com/Po-Hsun-Su/pytorch-ssim                               ###
#############################################################################################
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    C1 = 0.01**2
    C2 = 0.03**2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())

    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


#############################################################################################
# ##                     Velocity image loader & Shot gather Generator                    ###
#############################################################################################
def getFileList(file_dir, suffix=".png", fileNameOnly=False): 
    """
    Find all files' name under the given path
    Example:
        list_name = file_name(path)
    """
    nameList = []
    if file_dir[-1] != '/':
        file_dir += '/'
    lenPath = len(file_dir)
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == suffix:
                if fileNameOnly:
                    nameList.append(os.path.join(root, file)[lenPath:])
                else:
                    nameList.append(os.path.join(root, file))
    return sorted(nameList)


def loadvel(names_list, vmin=1500, vmax=4500, shot=False):
    """ 
    load vel models into shape [num_vels, nz, nx] 
    vel size is 200x200, but output size is 100x100
    """
    shot_list = []
    shot_data = []
    vel_models = []
    for vel_name in names_list:
        shot_name = vel_name.replace('velocity', 'shot').replace('vel_images', 'shot_gathers').replace('png', 'npz')
        vel = Image.open(vel_name)
        # tansform vel from 0~255 to vmin~vmax
        vel = np.asarray(vel) / 255 * (vmax - vmin) + vmin
        shot_list.append(shot_name)
        vel_models.append(vel[None, :, :])

        if shot:
            data = np.load(shot_name)['shot']
            shot_data.append(data[None, :, :])

    vel_models = np.concatenate(vel_models, axis=0)
    if shot:
        shot_data = np.concatenate(shot_data, axis=0)
        return vel_models[:, 0::2, 0::2], shot_data
    else:
        return vel_models[:, 0::2, 0::2], shot_list


def data_generator(vel_folder, forward_rnn, wavelet, batch_size=64, start_index=0, num_vels=None, dtype=torch.float32, device='cpu'):
    """
    Generate shot gathers with given vel images.
    Folder(Structures):
        Data:
            - vel_images
            - shot_gathers
    """
    vel_list = getFileList(vel_folder, fileNameOnly=False)
    vel_list = vel_list[start_index:]
    if num_vels is not None:
        vel_list = vel_list[start_index:start_index + num_vels]
    else:
        num_vels = len(vel_list)

    num_batches = int(np.ceil(num_vels / batch_size))
    with torch.no_grad():
        for batch_idx in range(0, num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, num_vels)
            if batch_idx == 0 or (batch_idx + 1) % 1 == 0 or batch_idx + 1 == num_batches:
                print("Propagating batch: {}/{}".format(batch_idx + 1, num_batches))

            vel_models, shot_list = loadvel(vel_list[batch_start:batch_end], vmin=1500, vmax=4500)
            vel_models = torch.tensor(vel_models, dtype=dtype, device=device)
            # forward propagation
            shot_Pred, _, _, _ = forward_rnn(vmodel=vel_models, wavelet=wavelet) 
            # shot_Pred shape: [batch_size, ns, nt, nx]
            for idx in range(shot_Pred.shape[0]):
                name = shot_list[idx]
                np.savez_compressed(name, shot=shot_Pred[idx].cpu().numpy())
        return print("Finished data generation")


#############################################################################################
# ##                                   Random velocity Generator                          ###
#############################################################################################
def velocity_gen(nx=200, nz=200, num_layer=5, vmin=1500, vmax=4000, vsalt=4500, hmin=20, hmax=40, embedding=True):
    """
    Random velocity generator
    Args:
        nz(int):     Number of grid point in depth dimension
        nx(int):     Number of grid point in horizontal dimension
        vmin(int):   Minimum velocity value, this also determines the velocity value in 1st layer.
        vmax(int):   Maximum velocity value
        hmin(int):   Minimum thickness (in grid point) for each layer
        hmax(int):   Maximum thickness (in grid point) for each layer
        salt(bool):  True, embedding a salt body with random shape 
    """
    x = np.arange(0, nx)
    phi = np.random.rand(2)  # Initial phase for sin and cos functions
    a = np.random.rand(2)    # Random coefficients for sin and cos functions
    b = np.random.randint(low=nx / 10, high=nx)   # scale factor for x

    # Generate the random interface
    fx = b / 4 * (a[0] * np.cos((x / b + phi[0]) * np.pi) + a[1] * np.sin((x / b + phi[1]) * np.pi))
    fx = np.round(fx - fx.min())

    # Initialize the velocity model with vmin
    velocity = np.ones((nz, nx)) * vmin
    thickness = hmin  # the minimum thickness of 1st layer is hmin
    vfill = vmin      # the minmum velocity of the 2nd layer is vmin

    # Generate a layered velocity model with increasing velocities along depth
    for idx in range(num_layer):
        vfill = np.random.randint(low=vfill, high=vfill + max(vmax / num_layer, 400))
        fx += thickness
        thickness = np.random.randint(low=hmin, high=hmax)
        for icol in range(nx):
            velocity[int(fx[icol]):, icol] = vfill

    # Embedding a salt body
    if embedding:
        # random width and height for salt body
        width = np.random.randint(low=70, high=150)
        height = np.random.randint(low=20, high=60)
        zone = saltZone(width, height)
        row, col = zone.shape

        # random the left-up corner location
        row0 = np.random.randint(low=60, high=nz - 60)
        col0 = np.random.randint(low=-col // 3, high=nx - 2 * col // 3)
        row1 = row0 + row
        col1 = col0 + col
        if col1 > velocity.shape[1]:
            zone = zone[:, :velocity.shape[1] - col1]
            col1 = velocity.shape[1]
        if col0 < 0:  # starting point is one the left of vel matrix
            zone = zone[:, -col0:]
            col0 = 0
        if row1 > velocity.shape[0]:
            zone = zone[:velocity.shape[0] - row1, :]
            row1 = velocity.shape[0]

        velocity[row0:row1, col0:col1] *= zone
        velocity[velocity == 0] = vsalt
    return velocity


def saltZone(width, height):
    # Generate a closed curve
    rad = 0.8
    edgy = 0.1
    a = get_random_points(n=7, scale=np.array([width, width]))
    x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
    x -= x.min()
    y -= y.min()
    poly = np.concatenate([x[:, None], y[:, None]], axis=-1)

    # create a coresponding matrix
    zone = np.ones((int(y.max()), int(x.max())))
    for irow in range(zone.shape[0]):
        for icol in range(zone.shape[1]):
            pos = (icol, irow)
            if inPoly(pos, poly):
                zone[irow, icol] = 0
    return zone


def rayInteract(pos, s_pos, e_pos): 
    """
    This function is to exclude cases that ray 
        (horizontal line  emitting from pos and towards to right) 
        not intersacts line segment 
        (starting from s_pos, ending with e_pos).
    Args:
        pos(tuple):     coordinates (x, y) for identifying points
        s_pos(tuple):   starting coordinates for line segment
        e_pos(tuple):   ending coordinates for line segment
    """                                     
    if (s_pos[1] - pos[1]) * (e_pos[1] - pos[1]) > 0:
        # this includes (y2>y0 & y1>y0) or(y2<y0 & y1<y0)
        return False
    if s_pos[1] == e_pos[1] == pos[1]:
        if s_pos[0] < pos[0]:  
            # start is on the left of pos
            return False
        elif s_pos[0] >= pos[0] and e_pos[0] >= pos[0]:
            return False
        else:
            return True
    interX = (s_pos[0] - e_pos[0]) / ((s_pos[1] - e_pos[1])) * (pos[1] - e_pos[1]) + e_pos[0]
    if interX < pos[0]:
        return False
    return True


def inPoly(pos, poly):
    """
    Args:
        pos(tuple):  coordinates (col, row) of identifying point         
        poly(array): size [num, 2], each row contains (x, y) coordinates 
    """
    count = 0
    for idx in range(poly.shape[0] - 1):
        s_pos = poly[idx]
        e_pos = poly[idx + 1]
        if rayInteract(pos, s_pos, e_pos):
            count += 1
    return True if count % 2 == 1 else False


def bernstein(n, k, t):
    return binom(n, k) * t**k * (1. - t) ** (n - k)


def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve


class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 10)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1)**2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1), 
                                           self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi),
                                           self.r * np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.p, self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def func(ang):
    return (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)


def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy) / np.pi + .5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    ang = func(ang)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x, y = c.T
    return x, y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7 / n
    a = np.random.rand(n, 2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec >= 200:
        return a * scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec + 1)


#############################################################################################
# ##                                       Wavelet Generator                              ###
#############################################################################################
class wGenerator(object):
    def __init__(self, t, freq=None):
        self.tvec = t
        self.dtype = t.dtype
        self.device = self.tvec.device
        if freq:
            self.freq = freq
        else:
            self.freq = 20  # default frequency for ricker
            self.freqOrmsby = [5, 10, 20, 30]  # default frequency for Ormsby
        
    def ricker(self):
        """
        Return a Ricker wavelet with the specified dominant self.frequency (default: 20Hz).
        """
        tmp = (np.pi * self.freq * (self.tvec - 1.0 / self.freq))**2
        #tmp = (np.pi * self.freq * self.tvec)**2
        wavelet = (1. - 2. * tmp) * torch.exp(-tmp)
        # return wavelet.type(self.dtype).to(self.device)
        return wavelet.type(self.dtype)
        
    def ricker_reform(self):
        """
        Return a reformed Ricker wavelet with the specified dominant self.frequency (default: 20Hz).
        """
        tmp = (np.pi * self.freq * (self.tvec - 1.0 / self.freq))**2
        wavelet = (1. - 2. * tmp) * torch.exp(-tmp * 2)
        # wavelet = (1. - 2. * tmp) * torch.exp(-tmp, dtype=self.dtype)
        # wavelet[0:20] = 0
        # wavelet[48:] = 0
        return wavelet.type(self.dtype).to(self.device)
    
    def gaussian(self):
        """
        Return a wavelet with a gaussian function 
        default: 
        x = t_vec
        xnot = dt
        xwid = dt
        yht = 1
        """
        x = self.tvec
        yht = 1
        xnot = self.tvec[1] - self.tvec[0]
        xwid = xnot
        sigma = xwid / 4
        wavelet = yht * torch.exp(-.5 * ((x - xnot) / sigma)**2)
        return wavelet.type(self.dtype).to(self.device)
    
    def ormsby(self):
        """
        Return a Ormsby wavelet with the specified list self.frequency (default: [5, 10, 20, 30]Hz).
        """
        if isinstance(self.freq, list):
            freqOrmsby = self.freq
        else:
            freqOrmsby = self.freqOrmsby
        wavelet = (np.pi * freqOrmsby[3]**2 / (freqOrmsby[3] - freqOrmsby[2]) 
                   * (np.sinc(np.pi * freqOrmsby[3] * self.tvec)**2) 
                   - np.pi * freqOrmsby[2]**2 / (freqOrmsby[3] - freqOrmsby[2]) 
                   * (np.sinc(np.pi * freqOrmsby[2] * self.tvec)**2)) 
        - (np.pi * freqOrmsby[1]**2 / (freqOrmsby[1] - freqOrmsby[0]) 
           * (np.sinc(np.pi * freqOrmsby[1] * self.tvec)**2) - np.pi 
           * freqOrmsby[0]**2 / (freqOrmsby[1] - freqOrmsby[0]) 
           * (np.sinc(np.pi * freqOrmsby[0] * self.tvec)**2))
        return torch.from_numpy(wavelet, device=self.device, dtype=self.dtype)


"""
For gen_Segment1d & gen_Segment2d:
input_data:     wavelet, is 1D tensor (shape: [num_vels, nt]) in time, 
                    which represents the total "RNN time steps".
                shot_records, represent N shot_records, 
                    in shape of [num_vels, num_shots, nt].
segment_size:   each segment include segment_size time_step, 
                    i.e., segment_size rnn units at each "time (RNN)" step.
option:         =0 (default), averaging partitioning the input with segement_size.
                =1, starting point for segments moving forward by step:
                    for even number segment_size: segment_size//2 step.
                    for odd number segment_size: segment_size//2+1 step.
                =2, starting point for segments at always index=0.
                    For example, segments are:
                    [0->segment_size, 0->2*segment_size, 0->3*segment_size, ...]
"""


#############################################################################################
# ##                                    Segment data Generator                            ###
# ##    Segment data/wavelet generator is for preparing truncated inputs for RNN          ###
#############################################################################################
def gen_Segment1d(wavelet=None, shot_records=None, segment_size=None, option=0):
    if shot_records is not None: 
        num_vels, num_shots, nt = shot_records.shape
    else:
        nt = len(wavelet)
    if segment_size is None:
        segment_size = nt
    
    x = None
    y = None
    if option == 1:
        num_segments = (nt - (segment_size + 1) // 2) // (segment_size // 2)
        # for even segment_size: num_segments = (nt - segment_size/2) // (segment_size/2)
        # for odd segment_size:  num_segments = (nt - segment_size//2-1) // (segment_size//2)
        for i in range(num_segments):
            if wavelet is not None:
                # prepare the input of wavelet 
                x = wavelet[i * segment_size // 2:i * segment_size // 2 + segment_size]
            if shot_records is not None:
                # partition of shot records
                y = shot_records[:, :, i * segment_size // 2:i * segment_size // 2 + segment_size]       
            yield (x, y)
    elif option == 2:
        num_segments = nt // segment_size
        if num_segments * segment_size < nt:
            num_segments += 1
        for i in range(num_segments):
            if wavelet is not None:
                # prepare the input of wavelet 
                x = wavelet[0:min((i + 1) * segment_size, nt)]
            if shot_records is not None:
                # partition of shot records
                y = shot_records[:, :, 0:min((i + 1) * segment_size, nt)]
            yield (x, y)
    else:  # option==0
        num_segments = nt // segment_size
        for i in range(num_segments):
            if wavelet is not None:
                # prepare the input of wavelet 
                x = wavelet[i * segment_size:(i + 1) * segment_size]
            if shot_records is not None:
                # partition of shot records
                y = shot_records[:, :, i * segment_size:(i + 1) * segment_size]       
            yield (x, y)
    

def gen_Segment2d(wavelet=None, shot_records=None, segment_size=None, option=0):
    if shot_records is not None:                
        num_batch, num_shots, nt, nx = shot_records.shape
    else:
        nt = len(wavelet)
    if segment_size is None:
        segment_size = nt

    x = None
    y = None
    if option == 1:
        num_segments = (nt - (segment_size + 1) // 2) // (segment_size // 2)
        # for even segment_size: num_segments = (nt - segment_size/2) // (segment_size/2)
        # for odd segment_size:  num_segments = (nt - segment_size//2-1) // (segment_size//2)
        for i in range(num_segments):
            if wavelet is not None:
                # prepare the input of wavelet 
                x = wavelet[i * segment_size // 2:i * segment_size // 2 + segment_size]
            if shot_records is not None:
                # partition of shot records
                y = shot_records[:, :, i * segment_size // 2:i * segment_size // 2 + segment_size, :]
            yield (x, y)
    elif option == 2:
        num_segments = nt // segment_size
        if num_segments * segment_size < nt:
            num_segments += 1
        for i in range(num_segments):
            if wavelet is not None:
                # prepare the input of wavelet 
                x = wavelet[0:min((i + 1) * segment_size, nt)]
            if shot_records is not None:
                # partition of shot records
                y = shot_records[:, :, 0:min((i + 1) * segment_size, nt), :]
            yield (x, y)
    else:  # option==0
        num_segments = nt // segment_size
        for i in range(num_segments):
            if wavelet is not None:
                # prepare the input of wavelet 
                x = wavelet[i * segment_size:(i + 1) * segment_size]
            if shot_records is not None:
                # partition of shot records
                #print("this is the time segments",i * segment_size, (i + 1)* segment_size)
                y = shot_records[:, :, i * segment_size:(i + 1) * segment_size, :]
                #print("this is the shape of the y",y.shape)       
            yield (x, y)


# class segGenerator(object):
#     def __init__(self, input_data, segment_size):
#         """
#         input_data: a tuple of tensors, (wavelet,shot_records)
#                     wavelet, is 1D tensor (shape: [num_vels, nt]) in time, which represents the total "RNN time steps".
#                     shot_records, represent N shot_records, in shape of [num_vels, num_shots, nt].
#         segment_size: each segment include segment_size time_step, i.e., segment_size rnn units at each "time (RNN)" step.
#         option: =0 (default), averaging partitioning the input with segement_size.
#                 =1, starting point for segments moving forward by step:
#                     for even number segment_size: segment_size//2 step.
#                     for odd number segment_size: segment_size//2+1 step.
#                 =2, starting point for segments at always index=0.
#                     For example, segments are:[0->segment_size, 0->2*segment_size, 0->3*segment_size, ...]
#         """
#         self.input = input_data
#         self.segment_size = segment_size
        
#     def gen_Segment1d(self, option=0):
#         wavelet, shot_records = self.input
#         if shot_records is not None: 
#             num_vels, num_shots, nt = shot_records.shape
#         else:
#             num_vels, nt = wavelet.shape
    
#         if option == 1:
#             num_segments = (nt - (self.segment_size + 1) // 2) // (self.segment_size // 2)
#             # for even segment_size: num_segments = (nt - segment_size/2) // (segment_size/2)
#             # for odd segment_size:  num_segments = (nt - segment_size//2-1) // (segment_size//2)
#             if shot_records is not None:
#                 for i in range(num_segments):
#                     # prepare the input of wavelet 
#                     x = wavelet[:, i * self.segment_size // 2:i * self.segment_size // 2 + self.segment_size]
#                     # partition of shot records
#                     y = shot_records[:, :, i * self.segment_size // 2:i * self.segment_size // 2 + self.segment_size]       
#                     yield (x, y)
#             else:
#                 for i in range(num_segments):
#                     x = wavelet[:, i * self.segment_size // 2:i * self.segment_size // 2 + self.segment_size]
#                     yield (x, None)
#         elif option == 2:
#             num_segments = nt // self.segment_size
#             if num_segments * self.segment_size < nt:
#                 num_segments += 1
            
#             if shot_records is not None:
#                 for i in range(num_segments):
#                     # prepare the input of wavelet 
#                     x = wavelet[:, 0:min((i + 1) * self.segment_size, nt)]
#                     # partition of shot records
#                     y = shot_records[:, :, 0:min((i + 1) * self.segment_size, nt)]
#                     yield (x, y)
#             else:
#                 for i in range(num_segments):
#                     x = wavelet[:, 0:min((i + 1) * self.segment_size, nt)]
#                     yield (x, None)
#         else:  # option==0
#             num_segments = nt // self.segment_size
#             if shot_records is not None:
#                 for i in range(num_segments):
#                     # prepare the input of wavelet 
#                     x = wavelet[:, i * self.segment_size:(i + 1) * self.segment_size]
#                     # partition of shot records
#                     y = shot_records[:, :, i * self.segment_size:(i + 1) * self.segment_size]       
#                     yield (x, y)
#             else:
#                 for i in range(num_segments):
#                     # prepare the input of wavelet 
#                     x = wavelet[:, i * self.segment_size:(i + 1) * self.segment_size]
#                     yield (x, None)
    
#     def gen_Segment2d(self, option):
#         wavelet, shot_records = self.input
#         if shot_records is not None:                
#             num_vels, num_shots, nt, nx = shot_records.shape
#         else:
#             num_vels, nt = wavelet.shape
    
#         if option == 1:
#             num_segments = (nt - (self.segment_size + 1) // 2) // (self.segment_size // 2)
#             # for even segment_size: num_segments = (nt - segment_size/2) // (segment_size/2)
#             # for odd segment_size:  num_segments = (nt - segment_size//2-1) // (segment_size//2)
            
#             if shot_records is not None:
#                 for i in range(num_segments):
#                     # prepare the input of wavelet 
#                     x = wavelet[:, i * self.segment_size // 2:i * self.segment_size // 2 + self.segment_size]
#                     # partition of shot records
#                     y = shot_records[:, :, i * self.segment_size // 2:i * self.segment_size // 2 + self.segment_size, :]       
#                     yield (x, y)
#             else:
#                 for i in range(num_segments):
#                     x = wavelet[:, i * self.segment_size // 2:i * self.segment_size // 2 + self.segment_size]
#                     yield (x, None)
#         elif option == 2:
#             num_segments = nt // self.segment_size
#             if num_segments * self.segment_size < nt:
#                 num_segments += 1
            
#             if shot_records is not None:
#                 for i in range(num_segments):
#                     # prepare the input of wavelet 
#                     x = wavelet[:, 0:min((i + 1) * self.segment_size, nt)]
#                     # partition of shot records
#                     y = shot_records[:, :, 0:min((i + 1) * self.segment_size, nt), :]
#                     yield (x, y)
#             else:
#                 for i in range(num_segments):
#                     x = wavelet[:, 0:min((i + 1) * self.segment_size, nt)]
#                     yield (x, None)
#         else:  # option==0
#             num_segments = nt // self.segment_size
#             if shot_records is not None:
#                 for i in range(num_segments):
#                     # prepare the input of wavelet 
#                     x = wavelet[:, i * self.segment_size:(i + 1) * self.segment_size]
#                     # partition of shot records
#                     y = shot_records[:, :, i * self.segment_size:(i + 1) * self.segment_size, :]       
#                     yield (x, y)
#             else:
#                 for i in range(num_segments):
#                     # prepare the input of wavelet 
#                     x = wavelet[:, i * self.segment_size:(i + 1) * self.segment_size]
#                     yield (x, None)

                    