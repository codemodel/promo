# promo
Probabilistic model 

Techniques used:
Correlation between column pairs (Pearson, Sperman correlation. Spearman is supposed to be better when underlying distribution is not known).

KDE - kernel density estimation. 

Polyfit, to find parameters a*x4 + b*x3 + c*x2 + d*x + e  that when given value x from column i, will produce (estimate) value at column j

Bins counting of values in data between each column pairs.


Libraries used: 
scipy.stats  for correlations, stats and poly

pickle for saving object to file and loading it

Numpy, numeric python for fast numeric tables that are implemented in C-language. (about 15 times faster than Python tables)

Pyplot (and matplotlib) plotting data

