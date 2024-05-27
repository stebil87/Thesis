# Thesis
Advancing Precision Agriculture: Machine Learning-based Early Detection of Potato Sprouting via Electrophysiological Signals

Last push:

- extended to all dataframes

- debugged anomaly detection

-  detrended wavelet linear and continous, plus keeped one dict with raw data

- featgen only with raw data, no windowing

- cleaned data after featgen (nans and inf in df12, replaced with previous OK value of the same feature)

- computed correlations wrt y
