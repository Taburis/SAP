
"""
parametric hypothesis test
"""

import scipy.stats.norm 
import scipy.stats.t
import numpy as np
import math

def one_sample_z(mean, x, alpha = 0.05, sigma=None):
    """
        one sample (x) z-test for:
        H0   mu = mean  vs H1  mu != mean
        where mu is the mean from sample x

        alpha: the significance level set to reject H0
        sigma: the input variance of sample, a sample variance will be used if it is absence
    """
    n = len(x)
    mu = float(np.mean(x))/n
    if not sigma:
        sigma = np.std(x)
    z = math.sqrt(n)*(mu-mean)/sigma
    pvalue = scipy.stats.norm.cdf(z)
    print(z, pvalue)
    left_tough = False
    right_tough = False
    report={"H0: mean = {mns}".format(mns=mean): "retain"}
    if pvalue > 1- alpha:
        right_tough = True
        report["H0: mean < {mns}".format(mns=mean)] = "reject"
        report["H0: mean = {mns}".format(mns=mean)] = "reject"
    else: report["H0: mean < {mns}".format(mns=mean)] = "retain"
    if pvalue < alpha:
        left_tough = True
        report["H0: mean > {mns}".format(mns=mean)] = "reject"
        report["H0: mean = {mns}".format(mns=mean)] = "reject"
    else : report["H0: mean > {mns}".format(mns=mean)] = "retain"
    return pvalue, report

def student_t(x,y, alpha = 0.05):
    """
    two sample student's t-test for probing:
    H0  mean_x = mean_y vs H1 mean_x != mean_y

    x , y : arrays of samples
    alpha : the significance level
    assuming sigma_x = sigma_y
    """

    meanx = np.mean(x)
    meany = np.mean(y)
    varx = np.std(x)
    vary = np.std(x)
    nx = len(x)
    ny = len(y)
    varp = math.sqrt(((nx-1)*varx+(ny-1)*vary)/(nx+ny-2))
    a = math.sqrt(1.0/nx+1.0/ny)
    var = varp*a
    t = (meanx-meany)/var
    df = nx+ny-2
    p = scipy.stats.t.cdf(t, df)
    
    return p

def welch_t(x, y, alpha=0.05):
    """ 
    two sample Welch's t-test for probing:
    H0  mean_x = mean_y vs H1 mean_x != mean_y

    x , y : arrays of samples
    alpha : the significance level
    assuming sigma_x != sigma_y
    """
    meanx = np.mean(x)
    meany = np.mean(y)
    varx  = np.std(x)
    vary  = np.std(y)
    nx = len(x)
    ny = len(y)
    sx2= varx/nx
    sy2= vary/ny
    t  = (meanx-meany)/math.sqrt(sx2+sy2)
    df = pow(sx2/nx+sy2/ny, 2)/(sx2*sx2/nx/nx/(nx-1)+sy2*sy2/ny/ny/(ny-1))
    p = scipy.stats.t.cdf(t, df)

    return p

