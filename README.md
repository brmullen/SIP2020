# READ ME for SIP 2020

## TODO list

### Analysis/Exploration/Visualization

- [x] Find ways of displaying data categories (pairplots and histograms)
- [x] Run any machine learning classifier (use the code from the jupyter notebook and the pyWholeBrainClassify.py)
- [x] Investigate based on different age groups:
- [x] Compare between domain vs no-domain (x2 per other comparison)
- [x] Compare each age group to all the data together (total of 8 ML algorithms)
    Use randomForest for classification

- [ ] Needed plots 
    - [ ] Plot the two features from no-domain data to see how they separate
    - [ ] Plot metrics on top of t-SNE plot domain plot (including age and animal number)
    - [ ] Create plot to show the extent of the data we showing (how many eigenvectors, how many ages?)
    - [ ] Implement contours intstead of a bunch of scatter plots? (Use 2dhistogram and then countours)
    - [ ] Combine the no-domain and domain accuracy plots together (overall accuracy)

- [ ] New Data
    - [ ] Get new data for analysis, run through our model (Brian)
    - [ ] Run new data through model, compare to how the new data performs on the data


### Presentation

**Remember**: We probably won't show all the data/things we try, since we have limited amount of time. We need to create a good representation even if its not everything.

__Next practice presentation__: **Friday August 7, 2020 at 3PM**
    - Desi
    - Sydney

- [ ] Create and manage a script for the presentation/manage the organization of the slides.
    - [ ] Add informative slide titles.
    - [ ] Periodically make the slides connect to the bigger picture.
    - [ ] Make sure all plots have axis, labels, colorbars, legends, etc.
- [ ] Make an eigenvector slide that describes how metrics were measured.
    - [ ] Spend a lot of time developing the spatial and temporal features.
    - [ ] Fourier analysis description.
- [ ] Add human classification of artifacts vs signal 
    - [ ] show confussion matrix?
    - [ ] Make it more prominant/this is our anchor
- [ ] Add domain vs no-domain description (if we keep the sepparate).  We can combine the results into the same graph.

__FINAL PRESENTATION__: **Saturday August 15, 2020**


==================================================================

New list:

['temporal.autocorr',
'region.extent',
'region.majaxis',
'region.minaxis',
'mass.region',
'threshold.area',
'freq.rangesz',
'freq.maxsnr.freq',
'freq.avgsnr',
'temporal.max',
'age']

Note: I added 'temporal.autocorr' and 'region.extent' because they work well when I run the whole dataset through a random forest classifier with 18 branches and 5 max features per branch.

Age-wise list of best-performing features:

P1:
['temporal.min','temporal.autocorr','region.extent','region.majaxis','region.majmin.ratio',"region.minaxis","mass.region","threshold.area","mass.total","freq.rangesz", "freq.maxsnr.freq", "freq.avgsnr", "spatial.min","temporal.max"]

P2:
['freq.rangesz', 'freq.maxsnr.freq', 'freq.avgsnr','temporal.autocorr', 'temporal.min', 'temporal.max', 'mass.region', 'region.majaxis', 'region.minaxis', 'threshold.area', 'spatial.min','region.extent']

P3:
['mass.region', 'region.majaxis', 'region.minaxis','threshold.area', 'temporal.autocorr','spatial.min', 'temporal.min', 'region.extent','temporal.max', 'freq.maxsnr.freq']

P4:
['temporal.autocorr', "freq.rangesz",'region.extent', 'region.majaxis', 'region.minaxis', 'mass.region', 'threshold.area', 'freq.maxsnr.freq', 'freq.avgsnr', 'temporal.max']

P5:
['temporal.autocorr', "freq.rangesz",'region.extent', 'region.majaxis', 'region.minaxis', 'mass.region', 'threshold.area', 'freq.maxsnr.freq', 'freq.avgsnr', 'temporal.max']
                
P6:
['temporal.autocorr', "freq.rangesz",'region.extent', 'region.majaxis', 'region.minaxis', 'mass.region', 'threshold.area', 'freq.maxsnr.freq', 'freq.avgsnr', 'temporal.max']
                
P7:
['temporal.autocorr','temporal.min', 'region.extent', 'region.majmin.ratio', 'region.majaxis', 'region.minaxis', 'mass.region', 'threshold.area', 'freq.rangesz', 'temporal.max', 'spatial.min', 'mass.total']

P8:
['region.minaxis', 'threshold.area', 'mass.region','region.majmin.ratio', 'freq.rangesz', 'region.majaxis','temporal.autocorr', 'temporal.max']

P9:
['temporal.autocorr',"freq.rangesz",'region.extent', 'region.majaxis', 'region.minaxis', 'mass.region', 'threshold.area', 'freq.maxsnr.freq', 'temporal.max']

P10:
['temporal.autocorr','temporal.min','region.extent','region.majaxis','region.majmin.ratio','region.minaxis','mass.region','threshold.area','freq.rangesz', 'freq.maxsnr.freq', 'freq.avgsnr']

P11:
['temporal.autocorr', "freq.rangesz",'region.extent', 'region.majaxis', 'region.minaxis', 'mass.region', 'threshold.area', 'freq.maxsnr.freq', 'freq.avgsnr', 'temporal.max']

P12:
['spatial.min', 'region.minaxis', 'freq.rangesz','temporal.autocorr', 'threshold.area', 'region.majaxis','mass.region', 'region.majmin.ratio', 'region.extent', 'temporal.max', 'temporal.min']

P13:
['temporal.autocorr', 'region.extent', 'region.majmin.ratio', 'freq.maxsnr.freq', 'region.minaxis', 'threshold.area', 'spatial.min', 'mass.region','region.majaxis', 'freq.rangesz']

P14:
['temporal.autocorr', 'freq.rangesz','region.extent', 'region.majaxis', 'region.minaxis', 'mass.region', 'threshold.area', 'freq.maxsnr.freq', 'freq.avgsnr', 'temporal.max']


Feature list for "No Domain" data:

*These features separate signal and artifact perfectly, so they can be used individually and the accuracy will still be at 100%

- 'region.minaxis'
- 'region.majmin.ratio'
- 'region.majaxis'
- 'mass.perc'
- 'mass.region'
- 'region.orient'
- 'mass.total'
- 'region.extent'
- 'region.centroid.1'
- 'region.centroid.0'
- 'region.eccentricity'

*If they overfit for other data and do no generalize well, the feature list below can be used to get a better model. The features above were excluded from the full list while generating the list below. 

['temporal.std',
 'spatial.min',
 'freq.rangesz',
 'temporal.max',
 'temporal.autocorr',
 'length',
 'freq.range.high',
 'freq.integrate',
 'freq.range.low',
 'freq.avgsnr',
 'freq.maxsnr.freq',
 'temporal.min']

Model used: Random Forest
Hyperparameters:
- n_estimators = 18
- max_features = 5

Performance with this list:
Accuracy: 92.1%
Precision: 89.6%
Recall: 71.1%