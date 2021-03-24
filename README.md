# BGP-SerialHijackers
Additional material for paper Profiling BGP Serial Hijackers: Capturing Persistent Misbehavior in the Global Routing Table, IMC ’19.
https://people.csail.mit.edu/ctestart/publications/BGPserialHijackers.pdf

File description:
---------
Feature_description.txt : Text file with description of all features.

groundtruth_dataset.csv /.pkl : csv/pandas pickle file with all features of ground truth ASes including their class (0: legitimate ASes, 1: serial hijacker ASes) and AS number ('ASN' column).

prediction_set_with_class.csv /.pkl: csv/pandas pickle file with all features of ASes originating 10 or more prefixes in the 5 year data including the prediction result of the classifier ('HardVotePred' column, 0: non-flagged, 1: flagged as having similar behavior to serial hijackers)  and AS number ('ASN' column).

treeClassifierEnsamble_CR.py: Code to train the voting ensemble of Extra-Trees classifiers from the ground truth to output a prediction for the prediction set.

flagged_networks.csv/.pkl: csv/pandas pickle file with all features of networks flagged by the classifier with the additional classification described in the paper (Blacklisted ASN, blacklisted prefixes, private AS number, DDoS protection provider, top 100, top 500, top 1000 CAIDA AS rank).

**How to run classifier:**
1- Download treeClassifierEnsamble_CR.py, create a folder named 'MA' in the same place and download grountruth_dataset.pkl and prediction_set_with_class.pkl to the MA floder.
2- Run treeClassifierEnsamble_CR.py. If not modifying the code, it will output predictions_54f_500t_34e.pkl and predictions_54f_500t_34e.csv in the MA folder.
3- Enjoy your new prediction and compare with the old ones in the prediction_set_with_class!
4- Want to try your own classifier? Change the number of ensemble, the balancing techniques used, the size of the forest and run again! If you want to study OOB prediction and the accuraty of the classifier, create an 'OOBpred' folder and a 'MA' folder and to save the results of the OOB prediction and accuracy of the model. From the OOB prediction files you can compute precision, recall and other parts of the confusion matrix by using a hard vote majority to predict the class of samples.
