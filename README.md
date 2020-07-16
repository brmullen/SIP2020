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



