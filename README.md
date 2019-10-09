# BGP-SerialHijackers
Additional material for paper Profiling BGP Serial Hijackers: Capturing Persistent Misbehavior in the Global Routing Table, IMC ’19

File description:
---------
Feature_description.txt : Text file with description of all features.

groundtruth_dataset.csv /.pkl : csv/pandas pickle file with all features of ground truth ASes including their class (0: legitimate ASes, 1: serial hijacker ASes).

prediction_set_with_class.csv /.pkl: csv/pandas pickle file with all features of ASes originating 10 or more prefixes in the 5 year data including the prediction result of the classifier (0: non-flagged, 1: flagged as having similar behavior to serial hijackers).

treeClassifierEnsamble_CR.py: Code to train the voting ensemble of Extra-Trees classifiers from the ground truth to output a prediction for the prediction set.
