This model optimizes curve-fitting by combining polynomial fitting, adaptive
sampling, and recursive iterating. The goal is to use as little compute as possible
to achieve only the necessary level of accuracy when solving ordinary
differential equations. This essentially means reaching ”convergence” in as
few iterations as possible. The required level of accuracy to achieve convergence
is set by the user, as they choose the maximum error for the model to
keep iterating. This model is unique because it only continues to sample in
regions where there are high residuals. This allows it to not only be efficient
with compute, but efficient with information. The model only samples new
points when it is necessary, hence making it very useful for large datasets.
The principles behind this model have broad implications beyond just differential
equations such as stock market trend analysis, derivatives trading,
and the optimization of neural nets by providing an efficient way to minimize
loss functions.
