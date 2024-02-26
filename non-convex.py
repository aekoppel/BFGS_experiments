"""
Self-contained implementation of non-linear optimization algorithms and application to the Rosenbrock function. 
Could easily be modified to consume as input the objective function associated with portfolio hedge and associated covariance data.
--> Note: the 1-norm term would need to be approximated by a subgradient, or otherwise projection onto ell-1 ball. 
-->My recommendation is the former , as it will be faster to compute as embedded within (L-)BFGS iterations.

(first three are commented out)
- steepest descent
- conjuage gradient
- Newton's method 
- BFGS  <--
- l-BFGS <--

Following Nocedal & Wright's Numerical Optimization Chapter 5 & 8

Written by Alec Koppel, modified from original source code from Yiren Lu on Github
"""
import matplotlib.pyplot as plt 
import matplotlib as mpl
import math
import time
import numpy as np


# 2d rosenbrock function and its first and second order derivatives
#     https://en.wikipedia.org/wiki/Rosenbrock_function
def rosenbrock(x):
  return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def grad_rosen(x):
  return np.array([200*(x[1]-x[0]**2)*(-2*x[0]) + 2*(x[0]-1), 200*(x[1]-x[0]**2)])


def hessian_rosen(x):
  return np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200]])


# line-search conditions
def wolfe(f, g, xk, alpha, pk):
  c1 = 1e-4
  return f(xk + alpha * pk) <= f(xk) + c1 * alpha * np.dot(g(xk), pk)


def strong_wolfe(f, g, xk, alpha, pk, c2):
  # typically, c2 = 0.9 when using Newton or quasi-Newton's method.
  #            c2 = 0.1 when using non-linear conjugate gradient method.
  return wolfe(f, g, xk, alpha, pk) and abs(
      np.dot(g(xk + alpha * pk), pk)) <= c2 * abs(np.dot(g(xk), pk))


def gold_stein(f, g, xk, alpha, pk, c):
  return (f(xk) + (1 - c) * alpha * np.dot(g(xk), pk) <= f(xk + alpha * pk)
          ) and (f(xk + alpha * pk) <= f(xk) + c * alpha * np.dot(g(xk), pk))


# line-search step len
def step_length(f, g, xk, alpha, pk, c2):
  return interpolation(f, g,
                       lambda alpha: f(xk + alpha * pk),
                       lambda alpha: np.dot(g(xk + alpha * pk), pk),
                       alpha, c2,
                       lambda f, g, alpha, c2: strong_wolfe(f, g, xk, alpha, pk, c2))


def interpolation(f, g, f_alpha, g_alpha, alpha, c2, strong_wolfe_alpha, iters=20):
  # referred implementation here:
  # https://github.com/tamland/non-linear-optimization
  l = 0.0
  h = 1.0
  for i in range(iters):
    if strong_wolfe_alpha(f, g, alpha, c2):
      return alpha

    half = (l + h) / 2
    alpha = - g_alpha(l) * (h**2) / (2 * (f_alpha(h) - f_alpha(l) - g_alpha(l) * h))
    if alpha < l or alpha > h:
      alpha = half
    if g_alpha(alpha) > 0:
      h = alpha
    elif g_alpha(alpha) <= 0:
      l = alpha
  return alpha

'''
# optimization algorithms
def steepest_descent(f, grad, x0, iterations, error):
  x = x0
  x_old = x
  c2 = 0.9
  for i in range(iterations):
    pk = -grad(x)
    alpha = step_length(f, grad, x, 1.0, pk, c2)
    x = x + alpha * pk
    if i % 10 == 0:
      # print "  iter={}, grad={}, alpha={}, x={}, f(x)={}".format(i, pk, alpha, x, f(x))
      print(f'iter={i},x={x},f(x)={f(x)}')
      #print ("  iter={}, x={}, f(x)={}".format(i, x, f(x)))
      

    if np.linalg.norm(x - x_old) < error:
      break
    x_old = x
  return x, i


def newton(f, g, H, x0, iterations, error):
  x = x0
  x_old = x
  c2 = 0.9
  for i in range(iterations):
    pk = -np.linalg.solve(H(x), g(x))
    alpha = step_length(f, g, x, 1.0, pk, c2)
    x = x + alpha * pk
    if i % 50 == 0:
      # print "  iter={}, grad={}, alpha={}, x={}, f(x)={}".format(i, pk, alpha, x, f(x))
      print(f'iter={i},x={x},f(x)={f(x)}')
      #print ("  iter={}, x={}, f(x)={}").format(i, x, f(x))

    if np.linalg.norm(x - x_old) < error:
      break
    x_old = x
  return x, i + 1


def conjugate_gradient(f, g, x0, iterations, error):
  xk = x0
  c2 = 0.1

  fk = f(xk)
  gk = g(xk)
  pk = -gk

  for i in range(iterations):
    alpha = step_length(f, g, xk, 1.0, pk, c2)
    xk1 = xk + alpha * pk
    gk1 = g(xk1)
    beta_k1 = np.dot(gk1, gk1) / np.dot(gk, gk)
    pk1 = -gk1 + beta_k1 * pk

    if i % 10 == 0:
      # print "  iter={}, grad={}, alpha={}, x={}, f(x)={}".format(i, pk, alpha, xk, f(xk))
      print(f'iter={i},x={xk},f(x)={f(xk)}')
      #print ("  iter={}, x={}, f(x)={}").format(i, xk, f(xk))

    if np.linalg.norm(xk1 - xk) < error:
      xk = xk1
      break

    xk = xk1
    gk = gk1
    pk = pk1

  return xk, i + 1

'''

def bfgs(f, g, x0, iterations, error):
  xk = x0
  x_vec=np.zeros((iterations,len(x0)))
  f_vec=f(x0)
  c2 = 0.9
  I = np.identity(xk.size)
  Hk = I

  for i in range(iterations):
    # compute search direction
    gk = g(xk)
    pk = -Hk.dot(gk)

    # obtain step length by line search
    alpha = step_length(f, g, xk, 1.0, pk, c2)

    # update x
    xk1 = xk + alpha * pk
    gk1 = g(xk1)

    # define sk and yk for convenience
    sk = xk1 - xk
    yk = gk1 - gk

    # compute H_{k+1} by BFGS update
    rho_k = float(1.0 / yk.dot(sk))

    Hk1 = (I - rho_k * np.outer(sk, yk)).dot(Hk).dot(I - \
           rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)

    if i % 10 == 0:
      # print "  iter={}, grad={}, alpha={}, x={}, f(x)={}".format(i, pk, alpha, xk, f(xk))
      print(f'iter={i},x={xk},f(x)={f(xk)}')
      #print ("  iter={}, x={}, f(x)={}").format(i, xk, f(xk))

    if np.linalg.norm(xk1 - xk) < error:
      xk = xk1
      break

    Hk = Hk1
    xk = xk1
    f_vec=np.append(f_vec,f(xk))
    xk = xk1
    x_vec[i]=xk
  return xk, i + 1, x_vec[0:len(f_vec),:], f_vec



def l_bfgs(f, g, x0, iterations, error, m=10):
  xk = x0
  x_vec=np.zeros((iterations,len(x0)))
  f_vec=f(x0)
  c2 = 0.9
  I = np.identity(xk.size)
  Hk = I

  sks = []
  yks = []

  def Hp(H0, p):
    m_t = len(sks)
    q = g(xk)
    a = np.zeros(m_t)
    b = np.zeros(m_t)
    for i in reversed(range(m_t)):
      s = sks[i]
      y = yks[i]
      rho_i = float(1.0 / y.T.dot(s))
      a[i] = rho_i * s.dot(q)
      q = q - a[i] * y

    r = H0.dot(q)

    for i in range(m_t):
      s = sks[i]
      y = yks[i]
      rho_i = float(1.0 / y.T.dot(s))
      b[i] = rho_i * y.dot(r)
      r = r + s * (a[i] - b[i])

    return r

  for i in range(iterations):
    # compute search direction
    gk = g(xk)
    pk = -Hp(I, gk)

    # obtain step length by line search
    alpha = step_length(f, g, xk, 1.0, pk, c2)

    # update x
    xk1 = xk + alpha * pk
    gk1 = g(xk1)

    # define sk and yk for convenience
    sk = xk1 - xk
    yk = gk1 - gk

    sks.append(sk)
    yks.append(yk)
    if len(sks) > m:
      sks = sks[1:]
      yks = yks[1:]

    # compute H_{k+1} by BFGS update
    rho_k = float(1.0 / yk.dot(sk))

    if i % 10 == 0:
      print(f'iter={i},grad={pk},alpha={alpha},x={xk},f(x)={f(xk)}')
      #print ("  iter={}, grad={}, alpha={}, x={}, f(x)={}").format(i, pk, \
      #  alpha, xk, f(xk))
    if np.linalg.norm(xk1 - xk) < error:
      xk = xk1
      break
    f_vec=np.append(f_vec,f(xk))
    # x_vec.append(xk)
    
    #x_vec=np.append(x_vec,    )
    # print(np.shape(x_vec))
    xk = xk1
    x_vec[i]=xk
    np.shape(x_vec)
  return xk, i + 1, x_vec[0:len(f_vec),:], f_vec


if __name__ == '__main__':
  x0 = np.array([0, 0])
  error = 1e-4
  max_iterations = 1000
'''
 # print '\n======= Steepest Descent ======\n'
  start = time.time()
  x, n_iter = steepest_descent(rosenbrock, grad_rosen, x0,
                               iterations=max_iterations, error=error)
  end = time.time()
  elapsed=end - start
  elapsed_iter=elapsed/n_iter
  print(f'Steepest Descent terminated in {n_iter} iterations, x={x}, f(x)={rosenbrock(x)}', \
     f'time elapsed={elapsed}, time per iter={elapsed_iter}')
#
#  print '\n======= Conjugate Gradient Method ======\n'
  start = time.time()
  x, n_iter = conjugate_gradient(rosenbrock, grad_rosen, x0,
                                 iterations=max_iterations, error=error)
  end = time.time()
  elapsed=end - start
  elapsed_iter=elapsed/n_iter
  print(f'Conjugate Gradient terminated in {n_iter} iterations, x={x}, f(x)={rosenbrock(x)}', \
      f'time elapsed={elapsed}, time per iter={elapsed_iter}')

#  print '\n======= Newton\'s Method ======\n'
  start = time.time()
  x, n_iter = newton(rosenbrock, grad_rosen, hessian_rosen, x0,
                     iterations=max_iterations, error=error)
  end = time.time()
  elapsed=end - start
  elapsed_iter=elapsed/n_iter
  print(f'Newton Method terminated in {n_iter} iterations, x={x}, f(x)={rosenbrock(x)}', \
      f'time elapsed={elapsed}, time per iter={elapsed_iter}')
'''
#  print '\n======= Broyden-Fletcher-Goldfarb-Shanno ======\n'
  start = time.time()
  x, n_iter,x_vec,f_vec = bfgs(rosenbrock, grad_rosen, x0,
                   iterations=max_iterations, error=error)
  end = time.time()
  elapsed=end - start
  elapsed_iter=elapsed/n_iter
  print(f'BFGS terminated in {n_iter} iterations, x={x}, f(x)={rosenbrock(x)}', \
      f'time elapsed={elapsed}, time per iter={elapsed_iter}')

  fig, ax = plt.subplots(1,3, figsize=(12,4))
  ax[0].plot(range(len(f_vec)), f_vec)
  ax[0].set_xlabel(r'Iteration $k$', fontsize=16)
  ax[0].set_ylabel('Objective $f(x_k)$', fontsize=16)
   
  time_vec=np.arange(len(f_vec))*1.0
  ax[1].plot(time_vec*elapsed_iter, f_vec)
  ax[1].set_xlabel(r'Time (seconds)', fontsize=16)
  ax[1].set_ylabel('Objective $f(x_k)$', fontsize=16)
  

  ax[2].plot(range(len(f_vec)), x_vec)
  ax[2].set_xlabel(r'Iteration $k$', fontsize=16)
  ax[2].set_ylabel(r'Iterates $x_k$', fontsize=16)
  # plt.legend()
  plt.savefig("BFGS.png", format='png')
  plt.close(fig)

  #print '\n======= Limited memory Broyden-Fletcher-Goldfarb-Shanno ======\n'
  start = time.time()
  x, n_iter,x_vec,f_vec = l_bfgs(rosenbrock, grad_rosen, x0,
                     iterations=max_iterations, error=error)
  end = time.time()
  elapsed=end - start
  elapsed_iter=elapsed/n_iter
  print(f'L-BFGS terminated in {n_iter} iterations, x={x}, f(x)={rosenbrock(x)}',\
      f'time elapsed={elapsed}, time per iter={elapsed_iter}')
  
  fig, ax = plt.subplots(1,3, figsize=(12,4))
  # plt.figure(tight_layout=True)
  ax[0].plot(range(len(f_vec)), f_vec)
  ax[0].set_xlabel(r'Iteration $k$', fontsize=16)
  ax[0].set_ylabel('Objective $f(x_k)$', fontsize=16)
  # ax[0].set_title('Maximum Token Price', fontsize=16)
  #ax[1].plot(range(len(f_vec)), x_vec,label=rf'iterates')

  time_vec=np.arange(len(f_vec))*1.0
  ax[1].plot(time_vec*elapsed_iter, f_vec)
  ax[1].set_xlabel(r'Time (seconds)', fontsize=16)
  ax[1].set_ylabel('Objective $f(x_k)$', fontsize=16)

  ax[2].plot(range(len(f_vec)), x_vec)
  ax[2].set_xlabel(r'Iteration $k$', fontsize=16)
  ax[2].set_ylabel(r'Iterates $x_k$', fontsize=16)
  #plt.legend()
  plt.savefig("L-BFGS.png", format='png')
  plt.close(fig)

