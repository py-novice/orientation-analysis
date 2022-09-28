def plot_orientations(ax, dim, vec, s = 30):
    """ Helping function for adding orientation-quiver to the plot.
    Arguments: plot axes, image shape, orientation, arrow spacing.
    """
    import numpy as np

    vx = vec[0].reshape(dim)
    vy = vec[1].reshape(dim)
    xmesh, ymesh = np.meshgrid(np.arange(dim[0]), np.arange(dim[1]), indexing='ij')
    ax.quiver(ymesh[s//2::s,s//2::s],xmesh[s//2::s,s//2::s],vy[s//2::s,s//2::s],vx[s//2::s,s//2::s],color='r',angles='xy')
    ax.quiver(ymesh[s//2::s,s//2::s],xmesh[s//2::s,s//2::s],-vy[s//2::s,s//2::s],-vx[s//2::s,s//2::s],color='r',angles='xy')
