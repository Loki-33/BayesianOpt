import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Qt5Agg')

def f(x):
	return np.sin(3*x) + x**2 - 0.7*x


def rbf_kernel(x1,x2,l=1.0, sigma_f=1.0):
	x1 = x1.ravel()
	x2 = x2.ravel()
	dist = np.subtract.outer(x1,x2)**2
	return sigma_f**2 * np.exp(-dist/(2*l**2))

def zero_mean(x):
	return np.zeros_like(x)

def gp_posterior(x_train, y_train, x_test, sigma_f=1.0, l=4.0, noise=1e-8):
	K = rbf_kernel(x_train, x_train) + noise * np.eye(len(x_train))
	K_s = rbf_kernel(x_train, x_test)
	K_ss = rbf_kernel(x_test, x_test) + noise * np.eye(len(x_test))

	K_inv = np.linalg.inv(K)
	m_train = zero_mean(x_train)
	m_test = zero_mean(x_test)
	y_train = y_train.reshape(-1, 1)
	m_train = zero_mean(x_train).reshape(-1, 1)
	m_test = zero_mean(x_test).reshape(-1, 1)

	mu_s = K_s.T @ K_inv @ y_train
	cov_s = K_ss - K_s.T @ K_inv @ K_s

	return mu_s, cov_s

def EI(X, X_sample, Y_sample, gp, xi=0.05): #xi is the small trade-off parameter(higer=more exploration)
	
	mu, sigma = gp(X)
	sigma = np.maximum(sigma, 1e-9)
	mu_sample_opt = np.min(Y_sample) # best observed value so far

	imp = mu_sample_opt - mu - xi  #omprovement
	Z = imp/sigma  #standardized improvemtn
	ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z) #expected improvement
	print(f"EI stats — imp: {imp[:5]}, sigma: {sigma[:5]}, ei: {ei[:5]}")

	return ei 


def gp_predict_wrapper(X_train, Y_train, l=1.0, sigma_f=1.0):
	def predict(X):
		mu, var = gp_posterior(X_train, Y_train, X, sigma_f, l)
		std = np.sqrt(np.maximum(np.diag(var), 1e-9))
		return mu.flatten(), std
	return predict 

X_grid = np.linspace(0,5, 1000).reshape(-1, 1)

X_sample = np.random.uniform(0, 5, size=(3,1))
Y_sample = f(X_sample)


n_iter = 10 

for i in range(n_iter):
	gp_predict = gp_predict_wrapper(X_sample, Y_sample)

	ei = EI(X_grid, X_sample, Y_sample, gp_predict)

	X_next = X_grid[np.argmax(ei)].reshape(1,-1)
	Y_next = f(X_next)

	X_sample = np.vstack((X_sample, X_next))
	Y_sample = np.vstack((Y_sample, Y_next))


	print(f"ITeration {i+1}: x_next = {X_next.flatten()[0]:.4f}, Y_next = {Y_next.flatten()[0]:.4f}")


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
mu, std = gp_predict(X_grid)
# mu, sigma = gp_predict(X_grid)
# plt.plot(X_grid, sigma)
# plt.title("Predicted Std Dev at each x")
# plt.show()

plt.fill_between(X_grid.ravel(), mu - 1.96*std, mu + 1.96*std, alpha=0.3)
plt.plot(X_grid, mu, 'r-', label='GP mean')
plt.scatter(X_sample, Y_sample, c='black', s=20, label='Samples')
plt.title(f"GP at iter {i+1}")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(X_grid, ei, label='EI')
plt.title("Expected Improvement")
plt.legend()
plt.tight_layout()
plt.show()
