import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def add_colorbar(ax, im, bbox_transform, #ax.transAxes
                 width="5%", 
                 height="100%",
                 loc='lower left',
                 bbox_to_anchor=(1.01, 0., 1, 1),
                 borderpad=0,
                 ctitle=''):
    axins = inset_axes(ax,
                       width=width, 
                       height=height,
                       loc=loc,
                       bbox_to_anchor=bbox_to_anchor,
                       bbox_transform=bbox_transform,
                       borderpad=borderpad)
    cbar = plt.colorbar(im, cax=axins)
    axins.set(title=ctitle)
    return cbar
    
def imagesc(fig,
            images,
            vmin=1.5,
            vmax=4.5,
            extent=[0, 1.01, 1.01, 0],
            aspect=1,
            nRows_nCols=(1, 1),
            cmap='coolwarm',
            ylabel="Depth (km)",
            xlabel="Position (km)",
            clabel="km/s",
            fontsize=2,
            xticks=np.arange(0., 1.01, 0.4),
            yticks=np.arange(0., 1.01, 0.4),
            cbar_width="5%",
            cbar_height="100%",
            cbar_loc='lower left',
            cbar_mode="corner",
            bbox_to_anchor=(1.05, 0., 1, 1.),
            ):
    (nrow, ncol) = nRows_nCols
    if not isinstance(vmin, (list, tuple, np.ndarray)) or not isinstance(vmax, (list, tuple, np.ndarray)):
        vmin = [vmin] * nrow
        vmax = [vmax] * nrow
    
    gs = fig.add_gridspec(nrow, ncol)
    for irow in range(nrow):
        for icol in range(ncol):
            ax = fig.add_subplot(gs[irow, icol])
            #print("irow and icol", irow, icol)
            #print(images[irow, icol].shape)
            im = ax.imshow(images[irow, icol], 
                           vmin=vmin[irow], vmax=vmax[irow], 
                           extent=extent,
                           aspect=aspect,
                           cmap=cmap)
            if icol == 0:
                ax.set_ylabel(ylabel, fontsize=fontsize)
                if yticks is not None:
                    ax.set_yticks(yticks)
            else:
                ax.set_yticks([])
            if irow == nrow - 1:
                ax.set_xlabel(xlabel, fontsize=fontsize)
                if xticks is not None:
                    ax.set_xticks(xticks)
            else:
                ax.set_xticks([])
            ax.tick_params(axis='both', labelsize=fontsize, which='major', pad=0.1)

            cbar_plot = False
            if cbar_mode == 'corner' and (irow == nrow-1 and icol == ncol-1):
                cbar_plot = True
            elif cbar_mode =='row' and (icol == ncol-1):
                cbar_plot = True
            elif cbar_mode == 'each':
                cbar_plot = True
            if cbar_plot is True:
                axins = inset_axes(ax,
                               width=cbar_width, 
                               height=cbar_height,
                               loc=cbar_loc,
                               bbox_to_anchor=bbox_to_anchor,
                               bbox_transform=ax.transAxes,
                               borderpad=0,
                               )
                axins.tick_params(axis='both', labelsize=fontsize, which='major', pad=0.1)
                cbar = plt.colorbar(im, cax=axins)
                cbar.ax.set_ylabel(clabel, fontsize=fontsize)
    return