# Presentation script

**Language:** English. **Four speakers:** Member 1 slides 1–4, Member 2 slides 5–6, Member 3 slides 7–9, Member 4 slides 10–12. **Timing:** 840 seconds, about 14 minutes, with Q&A separate. Rehearse aloud within the required 10–15 minutes. Replace names, IDs, proposed duties and the shared code URL before submission.

## Slide 1. Group identity

**Member 1 • About 35 seconds**

Good morning. We are [introduce all four names]. Our project classifies dry bean varieties using small neural networks. The input is a set of measured size and shape features, and the output is one of seven varieties. We use a public dataset and compare a small MLP with a two-hidden-layer MLP under a compact set of training choices. Our focus is a clear experiment that we can reproduce and explain, rather than a large architecture.

## Slide 2. Responsibilities

**Member 1 • About 35 seconds**

Our proposed work allocation has four parts. Member one handles the problem, dataset and preprocessing. Member two handles model architecture and training. Member three handles experiments and error analysis. Member four handles documentation, presentation and reproducibility. We will adjust these entries to reflect our actual contributions. All members should run the notebook and understand the model-selection rule, because the evaluation belongs to the whole group.

## Slide 3. Problem case

**Member 1 • About 65 seconds**

The problem is supervised classification of seven dry bean varieties. The data creators measured sixteen features from segmented bean images. Examples include area, perimeter, axis lengths and compactness. Similar varieties can overlap in these measurements, which makes classification more interesting than applying one simple threshold. A classifier could support a sorting workflow, but we do not claim to build a complete sorting machine. We receive numerical features rather than original photographs, so an MLP is a natural model. A CNN would add an unnecessary spatial assumption to these named measurements. We ask whether a second hidden layer and changes in learning rate, dropout or class weighting improve classification. A logistic-regression reference checks how much value the neural models add over a simpler method.

## Slide 4. Dataset and split

**Member 1 • About 95 seconds**

The public UCI Dry Bean dataset contains 13,611 rows, sixteen features and seven class labels. It has no missing values in the downloaded file. We found 68 exact duplicate rows and removed them before splitting, leaving 13,543 unique rows. The split is approximately seventy percent training, fifteen percent validation and fifteen percent test: 9,480, 2,031 and 2,032 rows respectively. Both splitting steps are stratified, so class proportions remain similar. We fit the standard scaler only on training data and reuse its means and standard deviations for validation and test. The test set does not select models. We use a fixed holdout rather than k-fold validation to keep the experiment small. The three training seeds are not three folds. We do not have acquisition-group identifiers, so this random split evaluates similar-distribution examples rather than new farms or cameras.

## Slide 5. Architectures

**Member 2 • About 85 seconds**

The small MLP takes sixteen standardized inputs, uses a hidden layer of thirty-two neurons with ReLU, then produces seven output logits. It has 775 trainable parameters. The larger MLP uses hidden layers of sixty-four and thirty-two neurons and has 3,399 parameters. The parameter count of a dense layer is input width times output width, plus one bias per output. ReLU introduces nonlinearity. Without it, stacked linear layers would still represent a linear transformation. The dropout variant places dropout after both hidden activations at a rate of 0.20. Dropout adds no trainable parameters and turns off during evaluation. The output consists of raw logits. Cross-entropy already performs the required log-softmax calculation, so we do not apply softmax before training loss. We use softmax only when displaying prediction confidence.

## Slide 6. Learning strategy

**Member 2 • About 100 seconds**

We compare six neural configurations with seeds seven, twenty-one and forty-two, giving eighteen runs. The small MLP is the baseline. The two-layer base model uses learning rate 0.001. Its variants use 0.0003, 0.003, dropout of 0.20, or class-weighted loss. Each variant changes one setting relative to the base. All use Adam, batches of 256 and up to forty epochs. We restore the checkpoint with the lowest unweighted validation cross-entropy and stop after eight epochs without sufficient improvement. The class-weighted variant changes only training loss. We calculate its weights from training counts, giving more influence to rare classes. Because the classes are imbalanced, we choose the neural configuration with the highest mean validation macro-F1 across seeds. Macro-F1 treats each variety equally. The fixed logistic regression uses C equal to one with no tuning and provides a separate reference.

## Slide 7. Validation results

**Member 3 • About 90 seconds**

This table reports every neural configuration. Macro-F1 is the primary comparison. The selected configuration is MLP-high-lr, with mean validation macro-F1 of 0.9367 and sample standard deviation 0.0015. Accuracy is also shown, but it is not our selection criterion. We compare the small and base MLPs for capacity, the learning-rate variants against the base for optimization, dropout against the base for regularization, and class weighting against the base for imbalance handling. This is a limited search on one split. The standard deviation describes variation from initialization and shuffling, not uncertainty across data splits. We have not established statistical significance. 9 of eighteen runs reached the forty-epoch cap, so the budget can affect the ranking. The logistic reference has validation macro-F1 0.9314, which provides useful context.

## Slide 8. Final test results

**Member 3 • About 90 seconds**

After selecting the neural configuration, we evaluated it, the small MLP and the fixed logistic reference on the untouched test partition. The selected configuration achieved mean macro-F1 0.9373 and mean accuracy 92.44 percent. The small MLP achieved macro-F1 0.9331. The difference is +0.0043 macro-F1. The logistic reference achieved 0.9349, so the selected neural model's difference from that simpler method is +0.0024. We report this comparison even if the difference is small. The neural summaries describe three separate models, not an ensemble. The learning curve shows the predetermined seed-seven run, with the checkpoint restored from epoch 26. We use unweighted evaluation loss for both curves, including when training used class weights. We do not change hyperparameters after seeing these test scores.

## Slide 9. Error analysis

**Member 3 • About 85 seconds**

This confusion matrix is for the selected neural configuration with seed seven only. Rows show actual classes and columns show predicted classes. The diagonal is correct classification. The model makes 153 mistakes out of 2,032 test rows. The largest off-diagonal counts include DERMASON as SIRA (45), SIRA as DERMASON (34), CALI as BARBUNYA (12). Raw counts can be larger for more common classes, so the notebook also reports per-class recall and F1. Recall asks how many true members of a variety were found. Precision asks how many predictions of that variety were correct. Macro-F1 averages the seven class F1 values equally. The first ten error rows include area, aspect ratio and compactness for inspection. We cannot show original photographs because the supplied archive contains feature tables. Similar measurements are a possible interpretation of confusions, not proof of the model's causal reasoning.

## Slide 10. Conclusion

**Member 4 • About 70 seconds**

Our selected neural configuration is MLP-high-lr, with mean test macro-F1 0.9373 and accuracy 92.44 percent. Its macro-F1 differs from the small MLP by +0.0043 and from the fixed logistic reference by +0.0024. The project shows how to compare capacity and learning choices without making the code unnecessarily complicated. We should keep the scope clear. This is a classifier of pre-extracted features, not an end-to-end image system. We used one random split and three seeds, and acquisition groups are unavailable. We have not established transfer to new farms, harvests or cameras. A sensible next experiment is repeated development splits and an independent acquisition-session test. Greater model depth should be motivated by evidence rather than assumed to be better.

## Slide 11. Code and reproducibility

**Member 4 • About 65 seconds**

The shared Colab notebook contains the complete code and explanations. It downloads the public archive, cleans duplicates, creates the splits, standardizes features, trains all configurations and evaluates the frozen choice. CPU is sufficient. We save split indices, original-row mapping, software versions, dataset checksum and every run's results. The model checkpoint includes the feature order, scaler means and scales, class mapping and architecture settings, as well as learned weights. We verify that reloading it reproduces predictions. The delivered outputs were executed locally, so a Colab rerun may differ slightly with platform or version changes. We will add our actual viewable notebook link to this slide after uploading. The saved outputs are also available if live internet access fails during the presentation.

## Slide 12. References

**Member 4 • About 25 seconds**

The data come from the UCI Machine Learning Repository under a Creative Commons attribution license. We also use the course materials and official PyTorch and scikit-learn documentation for implementation details. These sources support the dataset description, loss function, scaling and reproducibility choices. Thank you. We are ready for questions.

## Rehearsal and optional demo

Use the executed notebook rather than waiting for eighteen training runs in front of the audience. Show the validation summary and saved-model inference cell if time permits. Keep the deck open as a fallback. A live rerun is optional and is not included in the 14-minute script.

After slide 4: “Member two will explain the neural architectures and learning strategy.” After slide 6: “Member three will show how the experiments selected a model and how it performed.” After slide 9: “Member four will summarize the result and reproducibility.”

If time is short, shorten the layer-by-layer explanation on slide 5 and the experiment enumeration on slide 6. Preserve the split, selection rule, test result and limitations. Do not claim that the neural model is dramatically better when the measured reference difference is small.
