# Dry Bean Classification: Complete Project Report

## Executive summary

This project predicts seven dry bean varieties from sixteen numerical size and shape measurements. It uses small fully connected neural networks and a fixed logistic-regression reference. Six neural configurations run with three seeds, producing eighteen experiments. The primary selection measure is validation macro-F1 because class counts are unequal.

The validation-selected neural configuration is **MLP-high-lr**. On the held-out test set its mean macro-F1 is **0.9373 ± 0.0019**, and mean accuracy is **92.44% ± 0.15 percentage points**. The small MLP reference has macro-F1 0.9331, while fixed logistic regression has 0.9349. All numbers come from actual local CPU execution of the supplied notebook. They are descriptive results from one split, not a claim of statistical significance.

## 1. Assignment and deliverables

The supplied assignment specifies 3–4 group members, a secondary dataset, neural network/deep learning models, comprehensive experiments, exactly twelve slides, a 10–15 minute presentation and 5 minutes of Q&A on Friday, 16 October 2026. The user requested four members and English. Identity fields remain editable for the group to complete.

| Slide | Required topic | Project content |
|---|---|---|
| 1 | Member identities | Four name and student-ID fields |
| 2 | Job descriptions | Proposed duties for four members |
| 3 | Problem case | Seven-class dry bean classification |
| 4 | Data and splitting | Provenance, duplicate removal, stratified holdout |
| 5 | Architecture | One- and two-hidden-layer MLPs |
| 6 | Learning and tuning | Six configurations, three seeds, explicit selection rules |
| 7 | Results | Complete validation comparison |
| 8 | Results | Frozen-model test results and training curve |
| 9 | Results | Confusion matrix and class-level errors |
| 10 | Conclusion | Measured result and practical limitations |
| 11 | Code URL | Field for the group's shared Colab link |
| 12 | References | UCI, course materials and official documentation |

The notebook includes explanatory Markdown, executed outputs and exportable results. The script supplies spoken detail for the slides. The preparation guide covers concepts and likely questions. This report records the full methodology so the deck stays readable.

## 2. Problem formulation

Each example is a bean represented by measured size/shape values. The class is one of BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER or SIRA. Predicting that class is supervised multiclass classification. The Class column is excluded from the inputs. We use the spelling DERMASON from the actual file, even though descriptive repository prose can contain spelling variants.

An MLP is appropriate because the inputs are named numerical features. These are already extracted from images, so this project does not learn an end-to-end image representation. A CNN would impose a spatial arrangement that these columns do not naturally possess. We study whether nonlinear capacity and learning choices improve prediction while keeping the solution understandable.

A possible application is supporting variety sorting from consistent measurements. This experiment does not implement segmentation, mechanical sorting, deployment monitoring or new-camera calibration.

## 3. Dataset, license and cleaning

**Dry Bean (2020), UCI Machine Learning Repository**, [DOI 10.24432/C50S4B](https://doi.org/10.24432/C50S4B). [Repository page](https://archive.ics.uci.edu/dataset/602/dry+bean). License: CC BY 4.0. The repository describes the work of M. Koklu and I. A. Ozkan on multiclass dry bean classification.

The downloaded ARFF file has **13,611 rows**, **16 numeric features** and **7 labels**. We verified zero missing values, finite features and no identical feature vectors with conflicting labels. There are **68 exact duplicate rows**. We remove them before the split, retaining **13,543 unique rows**. Removing exact duplicates avoids distributing identical rows between training and test. It does not remove every possible near-duplicate or shared acquisition condition. We preserve a source-row mapping for traceability.

The feature names in file order are: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4. `AspectRation` is the source file's spelling and is retained consistently. Area and perimeter describe size, axis lengths describe elongation, and compactness/shape factors describe geometry. Some features are derived from others, so they can be correlated. We do not claim all sixteen are independent.

## 4. Split and preprocessing

Two stratified splits with seed 42 create approximately 70/15/15 partitions: **9,480 training**, **2,031 validation**, **2,032 test**. The first separates training from a 30% holdout. The second halves that holdout. Integer rounding explains the one-row difference. The original UCI release does not provide the same sort of predefined train/test boundary as some vision benchmarks, so this is our own documented split.

The scaler learns each feature's mean and standard deviation from training rows only. It then applies `(x - mean_train) / std_train` to every partition. LabelEncoder is fit on training labels. All classes appear in each stratified partition. The checkpoint stores feature order, scaler statistics and label mapping so inference matches training.

We use a holdout rather than k-fold validation for simplicity and runtime. Seeds 7, 21 and 42 repeat training on identical partitions. They measure initialization and shuffle variability, not sensitivity to split choice. Group identifiers for farm, acquisition session or camera are unavailable in the supplied feature file. This random-split estimate therefore concerns a similar distribution, not unseen acquisition conditions.

## 5. Architectures

| Network | Hidden layers | Parameters |
|---|---|---:|
| MLP-32 | 16 inputs, Dense32, ReLU, Dense7 | 775 |
| MLP-64-32 | 16 inputs, Dense64, ReLU, Dense32, ReLU, Dense7 | 3,399 |

For a dense layer, count `input width × output width + output biases`. The small MLP has (16×32+32)+(32×7+7)=775. The larger has (16×64+64)+(64×32+32)+(32×7+7)=1,088+2,080+231=3,399. ReLU and dropout add no trainable parameters.

The deeper variants use two hidden layers. Dropout, where enabled, follows both hidden activations with probability 0.20. Output scores are logits, not probabilities. For probabilities, softmax gives exp(z_c)/sum_j exp(z_j). Cross-entropy uses the negative log probability assigned to the true class. PyTorch combines log-softmax with the classification loss internally, so the training network ends in a linear output layer.

We also use multinomial logistic regression with fixed C=1.0 and max_iter=2000. It uses the same standardized training data and no validation tuning. This reference offers a simple linear decision model and does not claim to represent the best non-neural method.

## 6. Experiment design and learning strategy

| Name | Hidden widths | Learning rate | Dropout | Class weights |
|---|---|---|---|---|
| MLP-32 | 32 | 0.001 | 0 | No |
| MLP-base | 64, 32 | 0.001 | 0 | No |
| MLP-low-lr | 64, 32 | 0.0003 | 0 | No |
| MLP-high-lr | 64, 32 | 0.003 | 0 | No |
| MLP-dropout | 64, 32 | 0.001 | 0.20 | No |
| MLP-weighted | 64, 32 | 0.001 | 0 | Yes |

Each configuration uses seeds 7, 21 and 42, Adam, batch size 256 and a 40-epoch cap. Early stopping has patience 8 and minimum improvement 0.0001 in **unweighted validation cross-entropy**. Each run restores its best checkpoint. Training and validation curves use evaluation mode after each epoch so dropout does not distort the comparison.

The weighted variant sets class c's weight to `N_train / (7 × n_c_train)` using training labels only. It gives rarer classes more influence in the loss. It changes the training objective, while all validation losses and reported metrics stay unweighted and comparable. We do not use resampling or SMOTE.

After all eighteen runs, rank configurations by **mean validation macro-F1**, then lower mean validation loss, fewer parameters and name. The test set plays no role in the ranking. A checkpoint criterion and a configuration criterion can differ: validation loss selects a state within one run, while macro-F1 compares the resulting configurations. We keep the selected checkpoints rather than refit on training plus validation.

The final test stage evaluates only MLP-32, the selected neural configuration and the fixed logistic reference. All three neural seeds remain in the reported mean. There is no ensemble and no best-test-seed selection. Seed 7 is fixed in advance for error examples and saving. Testing the prespecified references is contextual comparison, not permission to revisit neural tuning.

## 7. Validation results

| Configuration | Macro-F1 mean ± sample SD | Accuracy mean (%) | Parameters |
|---|---:|---:|---:|
| MLP-32 | 0.9300 ± 0.0009 | 91.97 | 775 |
| MLP-base | 0.9355 ± 0.0020 | 92.65 | 3,399 |
| MLP-low-lr | 0.9306 ± 0.0002 | 92.07 | 3,399 |
| MLP-high-lr | 0.9367 ± 0.0015 | 92.78 | 3,399 |
| MLP-dropout | 0.9342 ± 0.0027 | 92.45 | 3,399 |
| MLP-weighted | 0.9352 ± 0.0011 | 92.60 | 3,399 |

**Selected neural configuration: MLP-high-lr.** Logistic reference validation macro-F1 is 0.9314. The small/base comparison changes both width and depth. Other variants change one factor relative to MLP-base. This is a targeted search, not an exhaustive interaction study. 9/18 runs reached the epoch cap, so convergence speed can affect the comparison. The validation score is optimistic to some extent because it also guides checkpointing and configuration choice.

The capacity comparison raises validation macro-F1 from 0.9300 for MLP-32 to 0.9355 for MLP-base. Under this budget, reducing the base learning rate to 0.0003 lowers the mean to 0.9306, while raising it to 0.003 gives 0.9367. Dropout at 0.20 gives 0.9342, and class weighting gives 0.9352. Neither tested regularization change exceeds the unmodified base's mean. These small differences do not prove that dropout or weighting are generally unhelpful. The search identifies the best tested setting, not a universal rule.

## 8. Test results and interpretation

| Model | Macro-F1 mean ± sample SD | Accuracy (%) mean ± sample SD |
|---|---:|---:|
| MLP-32 | 0.9331 ± 0.0011 | 92.03 ± 0.13 |
| MLP-high-lr | 0.9373 ± 0.0019 | 92.44 ± 0.15 |
| Logistic regression (one fit) | 0.9349 | 92.18 |

The selected neural model's mean macro-F1 differs from MLP-32 by **+0.0043** and from logistic regression by **+0.0024**. The logistic row is one deterministic fit and therefore has no seed SD. Neural results are averages of three individual models, not ensemble predictions.

Macro-F1 averages class-specific F1 values, treating rare and common varieties equally. Accuracy gives each example equal weight and can be more influenced by common classes. Both metrics matter. Sample SD is computed with denominator n−1 across the three training seeds. It is not a confidence interval or a significance test. All seeds share one test set.

For the selected seed-7 checkpoint, accuracy is 92.47% and macro-F1 0.9377, with **153 errors out of 2,032**. The largest confusions are **DERMASON as SIRA (45), SIRA as DERMASON (34), CALI as BARBUNYA (12)**. The confusion matrix has true labels in rows, predictions in columns. Large raw error counts can partly reflect class frequency, so the notebook includes per-class precision, recall, F1 and support. Error examples are the first ten errors in test-partition order, not a cherry-picked sample. They include raw feature measurements, because original bean images are not part of this input table.

## 9. Limitations

The data are pre-extracted measurements, so the experiment does not validate image acquisition or feature extraction. A different camera, farm, harvest or segmentation method can shift the feature distribution. Acquisition-group identities are unavailable. Exact duplicate removal does not establish complete statistical independence. One split and three seeds are a limited robustness check.

The comparison does not isolate depth alone because layer widths and parameter counts also differ. The fixed epoch cap can disadvantage slower learning rates. Dropout and class weighting have only one setting each. Logistic regression has fixed regularization, so the study does not identify the strongest possible classical baseline. No significance test or deployment evaluation was performed.

Useful extensions are repeated development splits, group-based validation if identifiers become available, and an external acquisition-session test. Once the current test scores have been inspected, additional tuning should be evaluated on new independent data for a fresh final claim.

## 10. Reproducibility and output files

Upload `Dry_Bean_Project.ipynb` to Colab and select Run all on CPU. The public archive downloads automatically if absent. Libraries are installed only when missing. The supplied notebook has completed local execution outputs. No Colab-hosted execution or upload is claimed. Small numerical differences are possible across library versions and platforms.

Local versions: `{"python": "3.12.10", "torch": "2.10.0+cpu", "numpy": "2.3.5", "pandas": "2.3.3", "sklearn": "1.8.0", "scipy": "1.16.3", "device": "cpu"}`. Neural execution uses two CPU threads and deterministic operations. The dataset SHA256 is `0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f`.

The results folder contains individual validation/test runs, summaries, class counts, split indices, cleaned-to-source row mapping, all epoch histories, predictions, per-class report, error rows, plots and experiment metadata. The saved seed-7 model contains weights, architecture, feature order, scaler mean/scale and class names. The notebook reloads it, preprocesses raw examples using saved statistics, and verifies prediction logits against the original model.

## 11. Submission tasks for the group

1. Fill four names and student IDs on slide 1 and in the notebook heading.
2. Adjust slide 2's proposed duties to match actual work.
3. Run and review the notebook in Colab. Save to Drive and share view access.
4. Paste the actual shared notebook URL into slide 11 and test viewer access. A generic Colab homepage is not a link to your code.
5. Rehearse using the script, keeping the talk within 10–15 minutes plus 5 minutes of questions.
6. If changing any settings, update the slide numbers and written explanations from the new run. Keep these reference results separate.

## References

- Dry Bean (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C50S4B . CC BY 4.0.
- Afiahayati. *4.1. MLP to Deep Learning*. Supplied course material.
- *4.2. Group Mini Project*, page 1. Supplied assignment.
- PyTorch. CrossEntropyLoss. https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- PyTorch. Reproducibility. https://docs.pytorch.org/docs/stable/notes/randomness.html
- scikit-learn. StandardScaler. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- scikit-learn. LogisticRegression. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
