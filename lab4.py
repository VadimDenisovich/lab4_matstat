import numpy as np 
import scipy.stats as stats 
import pandas as pd 
from statsmodels.stats.proportion import proportion_confint

SEED = 3 
SIZE_100 = 100
SIZE_1000 = 1000 

# .rvs() — «random variates» — случайные величины. Генерирует случайные выборки 
# из указанного распределения вероятностей. 
# loc — нижняя граница распределения (минимальное значение, которое может принять СВ)
# scale — ширина интервала распределения. (В таком случае loc + scale = 15, это верх. граница распределения)
#
# Равн. распределение на интервале [5, 15]
uniform_100 = stats.uniform.rvs(loc=5, scale=10, size=SIZE_100, random_state=SEED)

# Распределение Бернулли, вероятность p = 0.7 
bernoulli_100 = stats.bernoulli.rvs(p=0.7, size=SIZE_100, random_state=SEED)

# Биноминальное распределение с n=20 (число испытаний), p=0.6 (вероятность успеха в каждом испытании) 
binominal_100 = stats.binom.rvs(n=20, p=0.6, size=SIZE_100, random_state=SEED)

# Нормальное распределение с параметрами mu=10, sigma=2, 
# где mu — матожидание, а sigma^2 — дисперсия. 
# Поскольку sigma здесь это также стандартное отклонение, то большая часть значений
# будет находится в пределах +-2стандартных отклоения от среднего. 
normal_100 = stats.norm.rvs(loc=10, scale=2, size=SIZE_100, random_state=SEED)

uniform_1000 = stats.uniform.rvs(loc=5, scale=10, size=SIZE_1000, random_state=SEED)
bernoulli_1000 = stats.bernoulli.rvs(p=0.7, size=SIZE_1000, random_state=SEED)
binominal_1000 = stats.binom.rvs(n=20, p=0.6, size=SIZE_1000, random_state=SEED)
normal_1000 = stats.norm.rvs(loc=10, scale=2, size=SIZE_1000, random_state=SEED)

def bootstrap(sample, estimator, confidence_level=0.95, n_resamples=1000):
	"""
	sample — выборка 
	estimator — функция, вычисляющая состоятельную оценку
	confidence_level — уровень доверия 
	n_resamples (необязательно) — новые выборки, которые создаются из исходной выборки п
		с помощью случайного отбора элементов с возвращением (это значит, что один и тот же 
		элемент может быть выбран несколько раз)
		по умочанию 1000 выборок. 
	"""

	sample = np.array(sample) # переводим в Numpy массив, чтобы повысить эффективность

	bootstrap_estimates = []
	n = len(sample)

	for _ in range(n_resamples): 
		# replace = True позволяет реализовать выборку элементов с возвращением
		# т.е. элемент при случайно выборе возвращается в выборку и может быть выбран снова случайно 
		bootstrap_sample = np.random.choice(sample, size=n, replace=True)
		bootstrap_estimates.append(estimator(bootstrap_sample)) 
        
	alpha = 1 - confidence_level
	lower_percentile = alpha / 2 * 100 
	upper_percentile = (1 - alpha / 2) * 100 

	# np.percentile находит значение, ниже которого находится заданный процент наблюдений
	# например, если передать 25%, то будет выдано значение ниже которого находятся 25% 
	# bootstrap оценок. 
	lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
	upper_bound = np.percentile(bootstrap_estimates, upper_percentile)

	return lower_bound, upper_bound

def uniform_loc_estimator(sample):
  """
  Оценка для нижней границы распредления в равномерном распределении.
  """
  return np.min(sample)

def uniform_scale_estimator(sample):
  """
  Оценка для ширины распределения в равномерном распределении.
  """
  return np.max(sample) - np.min(sample)

def bernoulli_p_estimator(sample):
  """
  Оценка для p в распределении Бернулли. 
  """
  return np.mean(sample)

def binomial_p_estimator(sample):
  """
  Оценка для p в биноминальном распределении. 
  """
  return np.mean(sample) / 20 

def normal_loc_estimator(sample):
  """
  Оценка для mu в нормальном распределении. 
  """
  return np.mean(sample)

def normal_scale_estimator(sample):
  """
  Оценка для sigma в нормальном распределении. 
  """
  return np.std(sample, ddof=1)

titles = ['Название выборки', 'Оцениваемый параметр', 'Истинное значение', 
          'Нижняя граница Bootstrap', 'Верхняя граница Bootstrap', 
          'Нижняя граница SciPy', 'Верхняя граница SciPy']
result_df = pd.DataFrame(columns=titles) 

# Равномерное распределение
uniform_100_loc_my_bootstrap = bootstrap(uniform_100, uniform_loc_estimator)
uniform_1000_loc_my_bootstrap = bootstrap(uniform_1000, uniform_loc_estimator)
uniform_100_scale_my_bootstrap = bootstrap(uniform_100, uniform_scale_estimator)
uniform_1000_scale_my_bootstrap = bootstrap(uniform_1000, uniform_scale_estimator)

# Распределение Бернулли 
bernoulli_100_p_my_bootstrap = bootstrap(bernoulli_100, bernoulli_p_estimator)
bernoulli_1000_p_my_bootstrap = bootstrap(bernoulli_1000, bernoulli_p_estimator) 

# Биноминальное распределение
binomial_100_p_my_bootstrap = bootstrap(binominal_100, binomial_p_estimator)
binomial_1000_p_my_bootstrap = bootstrap(binominal_1000, binomial_p_estimator)

# Нормальное распределение 
normal_100_loc_my_bootstrap = bootstrap(normal_100, normal_loc_estimator)
normal_1000_loc_my_bootstrap = bootstrap(normal_1000, normal_loc_estimator)
normal_100_scale_my_bootstrap = bootstrap(normal_100, normal_scale_estimator)
normal_1000_scale_my_bootstrap = bootstrap(normal_1000, normal_scale_estimator)


bernoulli_100_scipy_ci = proportion_confint(count=np.sum(bernoulli_100), 
                                                nobs=len(bernoulli_100),
                                                alpha=0.05, method='wilson')

bernoulli_1000_scipy_ci = proportion_confint(count=np.sum(bernoulli_1000), 
                                                 nobs=len(bernoulli_1000),
                                                 alpha=0.05, method='wilson')

binomial_100_scipy_ci_p = proportion_confint(count=int(np.sum(binominal_100)/20), 
                                               nobs=len(binominal_100),
                                               alpha=0.05, method='wilson')
binomial_1000_scipy_ci_p = proportion_confint(count=int(np.sum(binominal_1000)/20), 
                                                nobs=len(binominal_1000),
                                                alpha=0.05, method='wilson')

normal_100_mean = np.mean(normal_100)
normal_100_std = np.std(normal_100, ddof=1)
normal_1000_mean = np.mean(normal_1000)
normal_1000_std = np.std(normal_1000, ddof=1)

normal_100_mean_ci = stats.t.interval(0.95, len(normal_100)-1, 
                                    loc=normal_100_mean, 
                                    scale=normal_100_std/np.sqrt(len(normal_100)))
normal_1000_mean_ci = stats.t.interval(0.95, len(normal_1000)-1, 
                                     loc=normal_1000_mean, 
                                     scale=normal_1000_std/np.sqrt(len(normal_1000)))

normal_100_var_ci = [(len(normal_100)-1) * normal_100_std**2 / stats.chi2.ppf(0.975, len(normal_100)-1),
                   (len(normal_100)-1) * normal_100_std**2 / stats.chi2.ppf(0.025, len(normal_100)-1)]
normal_100_std_ci = [np.sqrt(ci) for ci in normal_100_var_ci]

normal_1000_var_ci = [(len(normal_1000)-1) * normal_1000_std**2 / stats.chi2.ppf(0.975, len(normal_1000)-1),
                    (len(normal_1000)-1) * normal_1000_std**2 / stats.chi2.ppf(0.025, len(normal_1000)-1)]
normal_1000_std_ci = [np.sqrt(ci) for ci in normal_1000_var_ci]

def uniform_loc_ci(sample, confidence_level=0.95):
    n = len(sample)
    alpha = 1 - confidence_level
    
    lower_bound = np.min(sample) - np.max(sample) * stats.beta.ppf(1-alpha/2, 1, n) / n
    upper_bound = np.min(sample) - np.max(sample) * stats.beta.ppf(alpha/2, 1, n) / n
    
    return lower_bound, upper_bound

def uniform_scale_ci(sample, confidence_level=0.95):
    n = len(sample)
    alpha = 1 - confidence_level
    range_sample = np.max(sample) - np.min(sample)
    
    lower_bound = range_sample / stats.beta.ppf(1-alpha/2, n-1, 2)
    upper_bound = range_sample / stats.beta.ppf(alpha/2, n-1, 2)
    
    return lower_bound, upper_bound 

uniform_100_loc_scipy_ci = uniform_loc_ci(uniform_100)
uniform_1000_loc_scipy_ci = uniform_loc_ci(uniform_1000)
uniform_100_scale_scipy_ci = uniform_scale_ci(uniform_100)
uniform_1000_scale_scipy_ci = uniform_scale_ci(uniform_1000)

data = [
    # Uniform distribution
    ["Uniform 100", "loc", 5, uniform_100_loc_my_bootstrap[0], uniform_100_loc_my_bootstrap[1], 
     uniform_100_loc_scipy_ci[0], uniform_100_loc_scipy_ci[1]],
    ["Uniform 1000", "loc", 5, uniform_1000_loc_my_bootstrap[0], uniform_1000_loc_my_bootstrap[1], 
     uniform_1000_loc_scipy_ci[0], uniform_1000_loc_scipy_ci[1]],
    ["Uniform 100", "scale", 10, uniform_100_scale_my_bootstrap[0], uniform_100_scale_my_bootstrap[1], 
     uniform_100_scale_scipy_ci[0], uniform_100_scale_scipy_ci[1]],
    ["Uniform 1000", "scale", 10, uniform_1000_scale_my_bootstrap[0], uniform_1000_scale_my_bootstrap[1], 
     uniform_1000_scale_scipy_ci[0], uniform_1000_scale_scipy_ci[1]],
    
    # Bernoulli distribution
    ["Bernoulli 100", "p", 0.7, bernoulli_100_p_my_bootstrap[0], bernoulli_100_p_my_bootstrap[1], 
     bernoulli_100_scipy_ci[0], bernoulli_100_scipy_ci[1]],
    ["Bernoulli 1000", "p", 0.7, bernoulli_1000_p_my_bootstrap[0], bernoulli_1000_p_my_bootstrap[1], 
     bernoulli_1000_scipy_ci[0], bernoulli_1000_scipy_ci[1]],
    
    # Binomial distribution
    ["Binomial 100", "p", 0.6, binomial_100_p_my_bootstrap[0], binomial_100_p_my_bootstrap[1], 
     binomial_100_scipy_ci_p[0], binomial_100_scipy_ci_p[1]],
    ["Binomial 1000", "p", 0.6, binomial_1000_p_my_bootstrap[0], binomial_1000_p_my_bootstrap[1], 
     binomial_1000_scipy_ci_p[0], binomial_1000_scipy_ci_p[1]],
    
    # Normal distribution
    ["Normal 100", "loc", 10, normal_100_loc_my_bootstrap[0], normal_100_loc_my_bootstrap[1], 
     normal_100_mean_ci[0], normal_100_mean_ci[1]],
    ["Normal 1000", "loc", 10, normal_1000_loc_my_bootstrap[0], normal_1000_loc_my_bootstrap[1], 
     normal_1000_mean_ci[0], normal_1000_mean_ci[1]],
    ["Normal 100", "scale", 2, normal_100_scale_my_bootstrap[0], normal_100_scale_my_bootstrap[1], 
     normal_100_std_ci[0], normal_100_std_ci[1]],
    ["Normal 1000", "scale", 2, normal_1000_scale_my_bootstrap[0], normal_1000_scale_my_bootstrap[1], 
     normal_1000_std_ci[0], normal_1000_std_ci[1]]
]

for row in data:
    result_df = pd.concat([result_df, pd.DataFrame([row], columns=titles)], ignore_index=True)

result_df.to_csv('bootstrap_results.csv', index=False)