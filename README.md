# BGP-SerialHijackers
Additional material for paper Profiling BGP Serial Hijackers: Capturing Persistent Misbehavior in the Global Routing Table, IMC ’19.
https://people.csail.mit.edu/ctestart/publications/BGPserialHijackers.pdf

File description:
---------
Feature_description.txt : Text file with description of all features.

groundtruth_dataset.csv /.pkl : csv/pandas pickle file with all features of ground truth ASes including their class (0: legitimate ASes, 1: serial hijacker ASes) and AS number ('ASN' column).

prediction_set_with_class.csv /.pkl: csv/pandas pickle file with all features of ASes originating 10 or more prefixes in the 5 year data including the prediction result of the classifier ('HardVotePred' column, 0: non-flagged, 1: flagged as having similar behavior to serial hijackers)  and AS number ('ASN' column).

treeClassifierEnsamble_CR.py: Code to train the voting ensemble of Extra-Trees classifiers from the ground truth to output a prediction for the prediction set.

flagged_networks.csv/.pkl: csv/pandas pickle file with all features of networks flagger by the classifier with the additional classification described in the paper (Blacklisted ASN, blacklisted prefixes, private AS number, DDoS protection provider, top 100, top 500, top 1000 CAIDA AS rank).
