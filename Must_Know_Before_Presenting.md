# Must Know Before Presenting

## The 30-second explanation

“We classify seven dry bean varieties from sixteen measured shape features using small MLPs. We remove exact duplicates, standardize using training data only, compare six configurations across three seeds and select by validation macro-F1. The selected configuration is MLP-high-lr, with test macro-F1 0.9373 and accuracy 92.44%. We also report a simple logistic reference and acknowledge that one random split does not establish performance on new acquisition conditions.”

## Facts every member must know

| Item | Answer |
|---|---|
| Dataset | UCI Dry Bean, DOI 10.24432/C50S4B |
| Raw / duplicate / clean rows | 13,611 / 68 / 13,543 |
| Inputs / outputs | 16 numeric features / 7 varieties |
| Train / validation / test | 9,480 / 2,031 / 2,032 |
| Split | Approximately 70/15/15, stratified, seed 42 |
| Training repetitions | Seeds 7, 21, 42 on the same split |
| Neural configurations | Six, giving eighteen runs |
| Optimizer / batch / epoch cap | Adam / 256 / 40 |
| Early stopping | Unweighted validation loss, patience 8, minimum improvement 0.0001 |
| Configuration selection | Highest mean validation macro-F1 |
| Small / larger MLP parameters | 775 / 3,399 |
| Selected configuration | MLP-high-lr |
| Mean selected test macro-F1 | 0.9373 ± 0.0019 |
| Mean selected test accuracy | 92.44% ± 0.15 percentage points |
| Logistic reference test macro-F1 | 0.9349, one fit |
| Selected seed-7 errors | 153 / 2,032 |

## Essential concepts, explained plainly

**Supervised multiclass classification.** Training rows have known variety labels. The model learns to assign a new row to one of seven categories. A numeric label such as 0 or 1 is an index, not a ranking of bean quality.

**Feature versus target.** Features are inputs such as area and compactness. The target is the variety. Including Class among input columns would leak the answer. `source_row` is for traceability and is also excluded from features.

**Pre-extracted features.** The data creators measured shape from segmented images. Our network learns a nonlinear decision function from those measurements. It does not learn an image feature extractor.

**Parameters and hyperparameters.** Network weights and biases are learned parameters. Layer sizes, learning rate, dropout, class weights and epoch limits are hyperparameters or training choices.

**Standardization.** For each feature, subtract its training mean and divide by its training standard deviation. This stops large numerical scales such as area from dominating optimization relative to small ratios. Standardization is not a guarantee of a normal distribution. Fit on training only and reuse those statistics.

**Stratification.** Keep class proportions approximately similar across partitions. It does not create equal class counts or guarantee separation of farms, cameras or acquisition batches.

**Exact duplicate removal.** Remove identical rows before splitting so a copied row cannot appear in both training and test. This checks one concrete leakage risk. Near-duplicates and hidden group relationships can remain.

**Dense layer and ReLU.** A dense layer computes a weighted sum plus bias. ReLU gives max(0,x), introducing nonlinearity. Without nonlinear operations, a stack of linear layers is still a linear map.

**Forward pass, loss, backward pass, optimizer.** The forward pass produces logits. Loss scores how wrong the predictions are. Backpropagation uses the chain rule to calculate gradients. Adam uses those gradients to update weights. Backpropagation and optimization are related but different steps.

**Logits versus probabilities.** Logits are raw scores. Softmax converts them to values between zero and one that sum to one. Argmax gives the same winning class before and after softmax. PyTorch cross-entropy expects logits, so a separate softmax before this loss is unnecessary and inappropriate.

**Epoch and batch.** An epoch is one pass over training rows. One batch supplies one gradient update. With 9,480 training rows and batch size 256 there are 38 updates per epoch, including a smaller last batch.

**Learning rate.** It controls update scale. Too small can learn slowly. Too large can make training unstable. Adam adapts updates but still needs a learning rate. Comparisons with a fixed epoch budget also reflect convergence speed.

**Dropout.** It randomly disables hidden activations during training. At evaluation it turns off. The tested rate is 0.20 after each of the two hidden activations. Dropout adds no trainable parameters and is not guaranteed to improve every dataset.

**Class-weighted loss.** A rare class gets more influence in training. Here weight is N/(7×class count), using training labels. BOMBAY receives a larger weight than DERMASON. The weighted variant is an experiment, not an assumption that weighting must help. All validation losses use the same unweighted criterion.

**Early stopping.** Monitor validation loss and restore its best checkpoint. Patience tolerates temporary fluctuations. Reaching forty epochs means the cap ended that run, not necessarily that early stopping triggered or convergence was complete.

**Overfitting and generalization.** Overfitting means fitting training-specific patterns that do not transfer well. Look for a growing training/validation loss gap, but do not infer it from one fluctuating point. Generalization is performance on examples that did not guide weight updates or model choice.

## Architecture arithmetic

Small model: input 16, hidden 32, output 7. Parameters = (16×32+32)+(32×7+7)=544+231=775.

Two-hidden-layer model: input 16, hidden 64, hidden 32, output 7. Parameters = (16×64+64)+(64×32+32)+(32×7+7)=1,088+2,080+231=3,399. ReLU and dropout have no learned parameters. A batch of N examples has shapes N×16, N×64, N×32 and N×7. MLP-32 means 32 hidden units, not 32 hidden layers.

## Metrics without confusion

- Accuracy = correct predictions / all predictions. Common classes contribute more examples.
- For one class, precision = TP/(TP+FP), and recall = TP/(TP+FN). Treat that variety as positive and all other varieties as negative.
- F1 = 2×precision×recall/(precision+recall). It balances precision and recall.
- Macro-F1 averages the seven class F1 scores with equal weight. It is generally not the F1 of averaged precision and recall.
- A confusion matrix has true labels in rows and predicted labels in columns in this notebook. Diagonal entries are correct. Support means the number of true examples in a class.
- Mean ± sample SD summarizes three training-seed results. Sample SD uses n−1 in its denominator. It is not a confidence interval and does not measure variation across splits.
- The confusion matrix uses seed 7 only. It is not an average of the three models.
- Macro-F1 here is reported on a 0–1 scale. Accuracy is shown in percent. A 90% to 92% accuracy change is two percentage points, not automatically a two-percent relative increase.

## Questions the lecturer may ask

**Why use this dataset?** It is public, small enough for complete CPU experiments and easy to explain. Its unequal class counts and overlapping shape measurements create useful questions about metrics and regularization.

**Why use an MLP rather than a CNN?** The inputs are sixteen named measurements. Their column order is not a natural two-dimensional image grid. An MLP matches the supplied data without inventing spatial structure.

**Is this really an advanced deep learning project?** The assignment allows neural networks and/or deep learning. We compare one and two hidden layers and examine optimization, regularization, imbalance, checkpoint selection and reproducibility. We do not claim a very deep architecture or end-to-end representation learning.

**Where are the original bean photographs?** The archive supplies numerical features. We do not pretend generated images are dataset examples. The code classifies the provided measurements.

**Why remove duplicates before splitting?** Otherwise identical rows could appear in different partitions, making evaluation easier. We removed 68 duplicates while preserving one copy of each row and the source-row mapping.

**Did the cleaning use test outcomes?** No. Exact-row deduplication is a prespecified data-quality step before partition creation. We also check identical features for conflicting labels. No model predictions or test scores determine the cleaning.

**Why fit the scaler only on training?** Validation and test statistics should not affect the learned preprocessing. Their feature distributions are part of what we want to evaluate on held-out data.

**Did you use k-fold?** No. It is one fixed stratified holdout with three training seeds. Seeds change initialization and shuffling, not the data split. Repeated or grouped splits would strengthen future analysis.

**Why select by macro-F1?** Class counts differ substantially. Equal class weighting in macro-F1 makes minority-variety performance visible. Accuracy remains useful and is also reported.

**Why checkpoint by loss but choose configurations by F1?** Loss responds continuously to confidence and is useful for stopping. Macro-F1 matches the class-balanced comparison objective. Both rules are fixed before testing and openly documented.

**Does class weighting always help?** No. It can improve minority recall while changing precision or majority performance. That is why it is a separate validation experiment. It does not force the class predictions to be equally frequent.

**What is the logistic reference for?** To check whether nonlinear neural capacity adds practical value over a simpler linear classifier. C=1.0 is fixed without tuning. We report it honestly even if its performance is close to the neural model.

**Why does logistic regression have no SD?** It is a single deterministic fit with the chosen solver and fixed data. We do not invent three repeated independent results for it. Neural initialization creates the seed variation we summarize.

**Are you comparing only depth?** No. The small and base models change both width and depth and therefore parameter count. Other variants change one factor relative to the base. A width-matched study would be needed to isolate depth more carefully.

**Why not train longer?** A fixed forty-epoch budget keeps experiments compact. We report which runs reach it. Slower learning rates may benefit from more epochs, so our conclusion is conditional on the training budget.

**Why keep validation out of final training?** We preserve the exact selected checkpoints and stopping rule. Refitting on training plus validation would require a separate, prespecified stopping strategy. It is not necessary for the study's question.

**Why seed 7 for errors and model saving?** It is chosen in advance for a reproducible illustration. We did not choose the seed with the best test score. The final summary retains all three seeds.

**Is the improvement significant?** We did not perform a significance test. We report the observed differences and seed variation. All models use the same test rows, and one split with three seeds does not establish universal superiority.

**Can you deploy this at a farm?** Not from these results alone. We need consistent feature extraction and external evaluation across acquisition conditions. No farm or camera identifiers are available for a grouped test in this file.

**What gets saved besides weights?** Architecture, feature order, training mean and scale, class labels and the demonstration seed. Preprocessing and label mapping are essential to interpret the outputs correctly.

**What if a rerun changes the numbers?** Check versions, seeds, split and dataset checksum. Small numerical differences can occur across platforms. Report the actual rerun and update all deliverables consistently rather than mixing values.

## Code walkthrough every member should practice

1. Explain how the archive loads and why Class and source_row are excluded from features.
2. Show duplicate removal before splitting and scaler fitting on training only.
3. Read CONFIGS and say which factor each comparison changes.
4. Walk through make_model and calculate the output shape and parameter count.
5. Explain zero_grad, forward loss, backward, step, eval mode and saved checkpoints.
6. Show selected_name uses validation macro-F1 before test tensors are created.
7. Distinguish all-seed test summaries from the seed-7 confusion matrix.
8. Show that model reloading also uses stored preprocessing and verifies predictions.

## Before walking into the classroom

Fill the four member identities. Agree on actual duties. Upload and run the notebook in Colab, save it, share view access, and replace slide 11's URL field. Read the report, then rehearse the script with a timer. Each member should answer the split and leakage questions without notes. Keep the deck and executed notebook offline. Do not claim an external test, statistical significance, image feature learning or a Colab run that did not happen.
