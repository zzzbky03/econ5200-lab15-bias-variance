# Bias, Variance and an Unbiased Cross-Validation Score

## Objective
I wanted to see how far off a cross-validation score is when the same score was used to choose the model.

## Methodology
- Simulated 80 points from a known curve with noise.
- Cross-validated polynomials of degree 1 to 12 and picked the one with the lowest CV error.
- Scored that fit on 20,000 new points from the same simulation to get its true error.
- Ran nested cross-validation, which puts the whole degree search inside an outer loop of folds.
- Split the expected error into bias squared, variance and noise by simulation.
- Wrote my own k-fold cross-validation function for scikit-learn models and checked it against the lab's numbers.

## Key Findings
- The search chose degree 5, with a CV error of 0.1853.
- Its true error on 20,000 new points was 0.2330, so the reported score was too optimistic.
- Nested cross-validation estimated 0.1910. That is closer to the truth, but still low, because these 80 points happened to be less noisy than usual.
- The noise alone was 0.2025. No model can get below that, so the 0.1853 score was too good to be true.
- My k-fold function gave 0.1921 for degree 5 with 10 folds.
