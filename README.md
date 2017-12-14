# distributional_bootstrapping
Code to run the simulations about distributional bootstrapping

There are two main functions, both can be called from command line:
 - context_analysis.py can be used to run experiments where the goal is to compute (i) the usefulness of distributional contexts and the effect that distributional factors like frequency and lexical diversity have on contexts' usefulness; and (ii) the categorization accuracy in a PoS tagging experiment that uses co-occurrence counts with distributional contexts as features, and again the effect that distributional predictors have on how easy it is to categorize a word based on distributional information
 - cumulative_learning.py implements some of the finding of the previous analysis by focusing attention on only few contexts out of all the possible ones. Contexts' salience is defined as a combination of frequency, diversity, and predictability - which were found to all have positive effects on usefulness. The usefulness of the selected contexts is evaluated by running PoS tagging experiments and comparing several different models, varying the granularity of the context (bigrams, trigrams, or both), and the presence/absence of utterance boundaries. Other options are avaliable to run different experiments: the possibilities are documented in the function's docstring.
 
The folder context_utils contains several modules that are called in the two main functions.

The file Eng_POSmapping.txt contains the mapping from CHILDES PoS tags to custom tags that I devised. The functions in this repo can be run without it, but to reproduce my experiments it has to be passed as an argument to both functions.

The code is in Python 3, makes use of standard scientific packages such as numpy, matplotlib, sklearn plus other general purpose packages (os, re, ...). Importantly, the context_analysis.py function depends on TiMBL: it cannot be run if TiMBL is not installed. Cumulative_learning.py can also be asked to use TiMBL functionalities, but it is not mandatory: with the appropriate function call it can run also when TiMBL is not installed.

Code to perform statistical analysis of the data generated by the simulations is not provided here.

For any further detail, mailto: giovanni DOT cassani AT uantwerpen DOT be
