from scipy.stats import ttest_rel


a = [0.877, 0.883, 0.867, 0.903, 0.912, 0.905, 0.893, 0.922, 0.919, 0.903]

b = [0.962, 0.988, 0.959, 0.969, 0.978,1, 0.975, 0.988, 0.981, 0.97]
# Python paired sample t-test
print(ttest_rel(a, b))