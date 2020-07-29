# READ ME for SIP 2020

## TODO list

- [x] Find ways of displaying data categories (pairplots and histograms)
- [x] Run any machine learning classifier (use the code from the jupyter notebook and the pyWholeBrainClassify.py)
- [x] Investigate based on different age groups:

__BEST FEATURES BY AGE GROUP__

Saathvik (Age group: P1-P5):

    1. region.minaxis
    2. freq.rangesz
    3. mass.region/mass.perc
    4. threshold.area
    5. mass.total
    6. region.majaxis
    7. freq.maxsnr.freq
    8. freq.avgsnr

Anna (Age group: P6-10):

    1. region.majaxis
    2. region.minaxis
    3. spatial.min
    4. temporal.max
    5. temporal.min

Giovanni (Age group 11-14):

    1. freq.integrate
    2. freq.rangesz
    3. mass.region/mass.perc
    4. region.minaxis
    5. threshold.area


Features that don't have any meaningful relavance:

    1. length (this is the length of values that define the freq.rangesz, which their values are not evenly spaced)
    2. spatial.avg (these eigenvectors are zero centered to begin with, there are only slight variations from zero, most likely noise)
    3. temporal.avg (see spatial.avg)

Current list that we will use to compare models (3 regoin, 3 spatial, 3 freq, 1 temporal; July 16,2020):

    1. region.minaxis
    2. region.majaxis
    3. threshold.area
    4. mass.total
    5. mass.region/mass.perc
    6. spatial.min
    7. freq.rangesz
    8. freq.maxsnr.freq
    9. freq.avgsnr
    10. temporal.max
    11. age (not scaled?/scaled?)


- [ ] Compare between domain vs no-domain (x2 per other comparison)
- [ ] Compare each age group to all the data together (total of 8 ML algorithms)
    Use randomForest for classification

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