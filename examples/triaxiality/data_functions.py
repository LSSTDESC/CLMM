import numpy as np
import clmm


def weights(Sigma_crit, theta, Sigma_shape=1, Sigma_meas=0) :
	## EQUATION 35, 32, 33
	w  = 1 / (Sigma_crit**2 * (Sigma_shape**2 + Sigma_meas**2))
	w1 = np.cos(4*theta)**2 * w
	w2 = np.sin(4*theta)**2 * w
	return w, w1, w2

def calc_theta(x, y, rotation=-np.pi/2) :
	return np.arctan2(y, x) + rotation


def make_radial_bins(x, y, Nbins=10) :
	r = np.sqrt(x**2 + y**2)
	#r_bins = np.linspace(np.min(r), np.max(r), Nbins+1)
	r_bins = np.logspace(np.log10(np.min(r)), np.log10(np.max(r)), Nbins+1)
	inds = np.digitize(r, r_bins, right=True) - 1
	rbins_mean = np.array([np.mean(r[inds==i]) for i in range(Nbins)])
	return r, rbins_mean, inds

def Delta_Sigma_const(w, gamma1, Sigma_crit) :
	## TODO: sum over clusters
	return w * Sigma_crit * gamma1 / w

def Delta_Sigma_4theta(w1, w2, gamma1, gamma2, theta, Sigma_crit) :
	## TODO: sum over clusters
	return Sigma_crit * (w1*gamma1/np.cos(4*theta) + w2*gamma2/np.sin(4*theta)) / (w1 + w2)

def Delta_Sigma_4theta_cross(w1, w2, gamma1, gamma2, theta, Sigma_crit) :
	## TODO: sum over clusters
	return Sigma_crit * (w1*gamma1/np.cos(4*theta) - w2*gamma2/np.sin(4*theta)) / (w1 + w2)

def Delta_Sigma_const_cross(w, gamma2, Sigma_crit) :
	## TODO: sum over clusters
	return w*Sigma_crit*gamma2 / w

