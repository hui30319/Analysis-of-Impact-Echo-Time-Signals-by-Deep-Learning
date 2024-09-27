import numpy as np
from scipy.interpolate import CubicSpline

def jittering(x, sigma=0.):
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.):
    return x * np.random.normal(loc=1., scale=sigma)

def magnitude_warping(x, sigma=0., knot=4):
    x = x[..., np.newaxis]
    # print("MAGNITUDE_WARPING")
    orig_steps = np.arange(x.shape[0])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, 1))
    warp_steps = (np.ones((x.shape[1],1))*(np.linspace(0, x.shape[0]-1., num=knot+2))).T
    warper = np.array([CubicSpline(warp_steps[:,0], random_warps[:,0])(orig_steps)]).T
    return x[:,0] * warper[:,0]

def time_warping(x, sigma=0., knot=4):
    x = x[..., np.newaxis]
    # print("TIME_WARPING")
    orig_steps = np.arange(x.shape[0])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, 1))
    warp_steps = (np.ones((x.shape[1],1))*(np.linspace(0, x.shape[0]-1., num=knot+2))).T
    
    time_warp = CubicSpline(warp_steps[:,0], warp_steps[:,0] * random_warps[:,0])(orig_steps)
    scale = (x.shape[0]-1)/time_warp[-1]
    return np.interp(orig_steps, scale*time_warp, x[:,0])
