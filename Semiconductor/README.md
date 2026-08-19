TeamRAV -- Problem D: Machine Learning for Static Voltage Drop Prediction
Semiconductor Solutions Challenge 2026, Arizona State University


OVERVIEW

This repository contains two trained machine learning models for predicting
static IR drop in VLSI power delivery networks. Both models predict a
per-pixel voltage drop map from three layout-derived input maps:

  current_map.csv    cell current demand at each pixel
  pdn_density.csv    local PDN wire density (resistance proxy)
  eff_dist_map.csv   effective distance to the nearest supply pad

The primary submission model is the physics-informed WLS model.


MODELS

1. Physics-informed WLS model  (src/physics_model.py)
   A 31-feature weighted least squares model derived from Ohm's law.
   Features span isotropic and anisotropic Gaussian blurs of the current
   map, PDN interaction terms, and difference-of-Gaussians contrast features.
   All coefficients are non-negative (BVLS constraint), which enforces the
   physically correct direction of influence for each term.

   Results on 10 hidden real-circuit test cases:
     MAE  : 0.227 mV
     F1   : 0.704  (threshold = 90th percentile of ground-truth IR drop)

2. XGBoost model  (src/xgboost_model.py)
   A 26-feature gradient-boosted tree model using the same physics-inspired
   features. Requires GPU for practical training time. The saved checkpoint
   xgb_model.json can be loaded to skip retraining.

   Results on 10 hidden real-circuit test cases:
     MAE  : 0.243 mV
     F1   : 0.619


F1 SCORE DEFINITION

The F1 score reported throughout the code uses the 90th percentile of the
ground-truth IR drop distribution as the hotspot threshold. Pixels with IR
drop above this threshold are labelled as hotspots. F1 measures the balance
between precision and recall in identifying these high-severity pixels.

Two additional reference metrics are printed but are not the primary metric:
  F1_comp  -- threshold at 90% of the maximum ground-truth value
  F1_pct   -- separate 90th percentile thresholds for GT and prediction


REPOSITORY LAYOUT

  benchmarks/
      real-circuit-data/          10 real training circuits
      fake-circuit-data/          100 synthetic BeGAN circuits
      hidden-real-circuit-data/   10 hidden test circuits
  src/
      physics_model.py            physics-informed WLS model
      xgboost_model.py            XGBoost model
  predictions_final/              predicted CSV files and diagnostic figures
  xgb_model.json                  saved XGBoost checkpoint
  IR_Drop_Prediction.ipynb        Jupyter notebook (run to replicate results)
  requirements.txt                Python dependencies
  README.md                       this file


SETUP

Python 3.8 or later is required.

  pip install -r requirements.txt

GPU training (XGBoost only): an NVIDIA GPU with CUDA support is detected
automatically. If no GPU is found the model falls back to CPU, which takes
approximately 90 minutes for 500 trees on the full dataset.

Minimum RAM: 16 GB recommended for full-data physics model training.
On machines with less than 12 GB, open the notebook and pass subsample=0.2
to the run_all call (see Section 2 of the notebook).


RUNNING THE CODE

Open IR_Drop_Prediction.ipynb in JupyterLab or Jupyter Notebook and run the
cells in order.

  Section 1  setup and imports
  Section 2  train the physics model
  Section 3  predict a new map with the physics model
  Section 4  evaluate predictions against ground truth
  Section 5  train the XGBoost model (or load saved checkpoint)
  Section 6  predict a new map with the XGBoost model
  Section 7  combined results summary

Predicted IR drop maps are saved to predictions_final/ as CSV files in the
required format (one value per pixel, comma-delimited, same dimensions as
the input maps).


PREDICTING A NEW MAP

Physics model (from Python script or notebook cell):

  import sys
  sys.path.insert(0, 'src')
  import numpy as np
  import physics_model as pm

  current  = pm.load_csv('path/to/current_map.csv')
  pdn      = pm.load_csv('path/to/pdn_density.csv')
  eff_dist = pm.load_csv('path/to/eff_dist_map.csv')

  # params is returned by pm.run_all() or loaded from a saved checkpoint dict.
  pred = pm.predict_new(current, pdn, eff_dist, params)
  pred = pred * params['global_scale']

  np.savetxt('ir_drop_map.csv', pred, delimiter=',', fmt='%.6e')

XGBoost model (load saved checkpoint):

  import sys
  sys.path.insert(0, 'src')
  import numpy as np
  import xgboost as xgb
  import xgboost_model as xm

  model = xgb.XGBRegressor()
  model.load_model('xgb_model.json')

  current  = xm.load_csv('path/to/current_map.csv')
  pdn      = xm.load_csv('path/to/pdn_density.csv')
  eff_dist = xm.load_csv('path/to/eff_dist_map.csv')

  pred = xm.predict_map(model, current, pdn, eff_dist)
  np.savetxt('ir_drop_map.csv', pred, delimiter=',', fmt='%.6e')


OUTPUT FORMAT

Predicted IR drop maps are plain CSV files. Each row corresponds to one row
of the input map, and each value is the predicted voltage drop in volts.
The file has the same number of rows and columns as the input maps.


CONTACT

Ridwan Olabiyi
rolabiyi@asu.edu
School of Computing and Augmented Intelligence
Arizona State University

