import numpy as np
from scipy import linalg

# Smooths a vector using an n-degree polynomial function
#	y: data to be smoothed
#  	x: include if y does not increment in whole numbers
#	span: number between 0 and 1 specifying the span as a percentage of y
#	degree: degree of polynomial to fit (note that 1=lowess and 2=loess)
#	total_iter: number of iterations to run, more than one uses residuals
def py_smooth(y, x=None, span = 0.3, degree = 1, total_iter = 1):

    # Checks to make sure that the span is a valid number
    if span <= 0 or span > 1: 
        print("ERROR: please enter a span between 0 and 1")
        return -1
    
    span_len = int(np.ceil(span*len(y)))
    half_span_len = int(span_len/2)
    i = 0
    estimates = np.empty(len(y))
    residuals = []
    
    # Sets x-values to default range if not specified
    if x is None:
        x = np.arange(len(y))
    else:
        if len(x) != len(y): 
            print("ERROR: vector lengths not equal")
            print("x len: " + str(len(x)) + ", y len: " + str(len(y)))
            return -1

    h = [np.sort(np.abs(x - x[i]))[span_len] for i in range(len(x))]
    weights = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    weights = (1 - weights ** 3) ** 3
    
    # Main loop to smooth each point
    for i in range(0, len(y)):
        
        # Determines span endpoints, with smaller spans at each end
        min_span = max(i-half_span_len, 0)
        max_span = min(i+half_span_len,len(y))
        #max_distance = max(np.linalg.norm(x[i]-x[min_span]), np.linalg.norm(x[i]-x[max_span]))

        max_distance = h[i]
        
        #print(max_distance)

        # Determines weights using scaled distances for each point
        #weights = [tricube(abs(float(x[j]-x[i])/max_distance)) for j in range(min_span, max_span)]
        
        # Subsets values and weights to span
        x_span = [x[j] for j in range(len(x)) if weights[i][j] != 0]
        y_span = [y[j] for j in range(len(x)) if weights[i][j] != 0]
        weights_span = [w for w in weights[i] if w != 0]
        print(weights_span)

        # Fits weighted least-squares regression to values in span
        fit = np.polyfit(x_span, y_span, deg = degree, full = True, w = weights_span)

        weights_span = weights[:, i]
        print(weights_span)
        b = np.array([np.sum(weights_span * y), np.sum(weights_span * y * x)])
        A = np.array([[np.sum(weights_span), np.sum(weights_span * x)],
                      [np.sum(weights_span * x), np.sum(weights_span * x * x)]])
        beta = linalg.solve(A, b)
        print("beta0: " + str(beta[0]) + ", beta1: " + str(beta[1]))
        
        # Records estimate at focal point and residuals for each point in the span
        estimates[i] = fit[0][1] + (x[i])*fit[0][0]
        print("poly0: " + str(fit[0][1]) + ", poly1: " + str(fit[0][0]))

        residuals.append(np.empty(len(x_span)))
        for k in range(0, len(x_span)):
            residuals[i][k] = y_span[k] - (fit[0][1] + (x[k])*fit[0][0])

    # If robust specified, we calculate weights for a given number of times
    if total_iter > 1:
        it = 0
        for it in range(0, total_iter-1):
            i = 0
            for i in range(0, len(y)):
            
                # Determines span endpoints, with smaller spans at each end
                min_span = max(i-half_span_len, 1)
            
                # Calculates the MAD and robust weights
                m = mad(residuals[i])
                robust_weights = [biopython_bisquare(r, m) for r in residuals[i]]
            
                # Subsets estimates and weights to span
                x_span = [x[j] for j in range(0, len(robust_weights))]
                y_span = [estimates[j] for j in range(0, len(robust_weights))]
            
                # If all weights are equal or NaN present, we keep the previous estimate
                is_na = np.isnan(robust_weights).any()
                if len(set(robust_weights)) == 1 or is_na: continue
           
                # Fits weighted least-squares regression using robust weights to values in span
                fit = np.polyfit(x_span, y_span, deg = degree, full = True, w = robust_weights)
                
                # Records estimate at focal point and residuals for each point in span
                estimates[i] = fit[0][1] + (x[i])*fit[0][0]
                for k in range(0, len(x_span)):
                    residuals[i][k] = y_span[k] - (fit[0][1] + (x[k])*fit[0][0])
            
    # Returns estimates
    return estimates

# Tricube weight function, assuming x is a scaled distance
def tricube(x): 
    if x < 1: return (1-(x**3))**3
    else: return 0
   
# Bisquare weight function, given the median absolute deviation.
# Removes observations that have a probability of being observed
# of less than 0.0001 from smoothing calculations
def bisquare(x, m): 
    if x < 6: return (1-(x/(6*m))**2)**2 
    else: return 0

# Biopython's bisquare implementation
def biopython_bisquare(x, m):
    y = np.clip(x/(6*m), -1, 1)
    return (1-(y**2))**2

# Found at https://gist.github.com/agramfort/850437
def git_lowess(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(np.ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]
        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

# Tests that the lowess works on a 1-d linear dataset
def lowess_test_1(output_normal, output_robust, span = 0.3):
    y = np.array([0.9,0.8,0.8,1.3,
                  1.4,1.2,1.7,1.8,
                  1.6,1.5,1.5,2.0,
                  2.5,2.7,2.9,2.5,
                  3.1,2.4,2.2,2.9,
                  2.5,2.6,3.2,3.8,
                  4.2,3.9,3.7,3.3,
                  3.7,3.9,4.1,3.8,
                  4.7,4.4,4.8,4.8,
                  4.8])
    x = np.arange(len(y))
    np.savetxt(output_normal, git_lowess(x, y, f=span, iter=1), delimiter=",")
    np.savetxt(output_robust, git_lowess(x, y, f=span, iter=5), delimiter=",")

# Tests that the lowess works on a 2-d curvy dataset
def lowess_test_2(output_normal, output_robust, span = 0.3):
    x = np.array([0.5578196, 2.0217271, 2.5773252, 3.4140288, 4.3014084, 4.7448394, 5.1073781,
                  6.5411662, 6.7216176, 7.2600583, 8.1335874, 9.1224379, 11.9296663, 12.3797674,
                  13.2728619, 14.2767453, 15.3731026, 15.6476637, 18.5605355, 18.5866354, 18.7572812])
    y = np.array([18.63654, 103.49646, 150.35391, 190.51031, 208.70115, 213.71135, 228.49353,
                  233.55387, 234.55054, 223.89225, 227.68339, 223.91982, 168.01999, 164.95750,
                  152.61107, 160.78742, 168.55567, 152.42658, 221.70702, 222.69040, 243.18828])
    np.savetxt(output_normal, git_lowess(x, y, f=span, iter=1), delimiter=",")
    np.savetxt(output_robust, git_lowess(x, y, f=span, iter=5), delimiter=",")



