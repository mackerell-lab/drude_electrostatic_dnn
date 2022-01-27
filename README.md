# drude_electrostatic_dnn

This repository contains data related to

## Deep Neural Network model to predict the electrostatic parameters in the polarizable classical Drude oscillator force field
Anmol Kumar, Poonam Pandey, Payal Chatterjee and Alexander D. MacKerell Jr.

Copyright (c) 2022, University of Maryland Baltimore

This release includes testing and training data for generating Drude Electrostatic model published in Journal of Chemical Theory and Computation.
The Drude Electrostatic model created using Deep Neural Network can predict atomic partial charges and atomic polariizability values in Drude force-field format.

## Contents

1. The dgenff_dataset.2021.tar.gz holds the feature vector and target (charge, alpha and thole) for the entire data in pickle format, split in test and train sets.

2. The dgenff_dnn_model holds the charge, alpha and thole model which can be used to predict electrostatic parameters.

3. DNN_charge_pol_pred.2021.py file allows one to use the data files in pickle format to predict charge, alpha and thole values. 
 
4. dnn_predicted_alphathole.csv and dnn_predicted_charges.csv contains target and predicted values for testing dataset.
