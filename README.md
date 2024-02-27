API changes:

1. DataManager.get_torch_iterator() get as argument [TRAIN, TEST, VAL, RARE, NEGATED_POLARITY] And not only train/val/test. Its seem like the natural place to get those subsets
2. binary_accuracy() is not used - we did another implementation to compute loss/acc. Since could be situation that last batch size nto equal the rest of the batches.
3. train_log_linear_with_one_hot(), train_log_linear_with_w2v() and train_lstm_with_w2v() returns tuple of the trained model and the dataManager such that we could save and run the model on another scenarios (test set and the special sets) more easily - and use less code by using one function named answer() to get the trained models and answer the questions






