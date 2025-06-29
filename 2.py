# stats
import statistics as stats

data = [10,20,30,30,40,40,50]

print("Mean:",stats.mean(data))
print("Median:",stats.median(data))
print("Mode:",stats.mode(data))
print("All Modes:",stats.multimode(data))
print("Harmonic Mean:",stats.harmonic_mean(data))
print("geometric mean:",stats.geometric_mean(data))
print("Variance:",stats.variance(data))
print("Standard Deviation:",stats.stdev(data))


#math

import math
x=16
a,b = 12,18
print("Square Root of",x,":",math.sqrt(x))
print("Integer Square Root of",x,":",math.isqrt(x))
print(math.pow(2,3))
print(math.exp(2))
print(math.log(10))
print(math.log10(1000))
print(math.factorial(5))
print(math.gcd(a,b))
print(math.lcm(a,b))
print(math.floor(4.7))
print(math.ceil(4.2))
print(math.fabs(-1))
print(math.radians(90))
print(math.degrees(math.pi/2))
print(math.pi)
print(math.e)
print(math.inf)
print(math.sin(math.radians(90)))


#numpy
import numpy as np
arr1 = np.array([1,2,3,4,5])
print(arr1)
print(np.sum(arr1))
print(np.prod(arr1))
print(np.mean(arr1))
print(np.median(arr1))
print(np.std(arr1))
print(np.var(arr1))
print(np.min(arr1))
print(np.max(arr1))
print(np.sqrt(arr1))
print(np.exp(arr1))
print(np.sin(arr1))

print(np.sort(arr1))
print(np.flip(arr1))
print(np.unique(arr1))
print(np.cumsum(arr1))
print(np.cumprod(arr1))
print(np.diff(arr1))
reshaped_arr = np.reshape(arr1,(1,5))
print(reshaped_arr)
transposed_arr = np.transpose(reshaped_arr)
print(transposed_arr)

random_arr = np.random.rand(3,3)
print(random_arr)


#scipy

from scipy import stats, linalg, integrate, optimize, spatial

arr = np.array([1,2,3,4,5])
print(stats.skew(arr))
print(stats.kurtosis(arr))

# probability dist
print(stats.norm.pdf(2))
print(stats.binom.pmf(2,10,0.5)) # 2 successes, 10 trails, 0.5 probability

def func(x):
  return x**2+5*x+6

result = optimize.minimize(func,0)
print(result.x)

# Linear Algebra - Solving Linear Equations (Ax = b)
A = np.array([[3,2],[1,4]])
b=np.array([7,10])
x=linalg.solve(A,b)
print(x)


def integrand(x):
  return x**2

integral_res, _=integrate.quad(integrand,0,1)
print(integral_res)


point1 = np.array([1,2])
point2 = np.array([4,6])
dist = spatial.distance.euclidean(point1,point2)
print(dist)


x = np.array([10,20,30,40,50])
y = np.array([15,25,35,45,55])
corr = stats.pearsonr(x,y)
print(corr)


