# drude_electrostatic_dnn

Copyright (c) 2022, University of Maryland Baltimore

This release includes testing and training data for generating Drude Electrostatic model published in Journal of Chemical Theory and Computation.
The Drude Electrostatic model created using Deep Neural Network can predict atomic partial charges and atomic polariizability values in Drude force-field format.

The dgenff_dataset.2021.tar.gz holds the feature vector and target (charge, alpha and thole) for the entire data in pickle format, split in test and train sets.

The dgenff_dnn_model.2021.tar.gz holds the charge, alpha and thole model which can be used to predict electrostatic parameters.

DNN_charge_pol_pred.2021.py file allows one to use the data files in pickle format to predict charge, alpha and thole values. 
