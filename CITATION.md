# How to cite

## The paper

Please cite the paper when you use the code or data of this repository:

Simpson, D. D., Wixted, J. T., & Brady, T. F. (2026). *Memorability as an emergent property: How
representational structure creates apparent intrinsic effects* [Manuscript submitted for publication].
Department of Psychology, University of California, San Diego.

## Data and models of others that the analyses use

Cite these when you use the parts of the repository that rest on them.

| Used in this repository | Reference |
|---|---|
| THINGS concepts and image names (`shared/things/`) | Hebart, M. N., Dickter, A. H., Kidder, A., Kwok, W. Y., Corriveau, A., Van Wicklin, C., & Baker, C. I. (2019). THINGS: A database of 1,854 object concepts and more than 26,000 naturalistic object images. *PLOS ONE*, 14(10), e0223792. https://doi.org/10.1371/journal.pone.0223792 |
| SPoSE embedding of the 1,854 concepts (49 dimensions) | Hebart, M. N., Zheng, C. Y., Pereira, F., & Baker, C. I. (2020). Revealing the multidimensional mental representations of natural objects underlying human similarity judgements. *Nature Human Behaviour*, 4(11), 1173-1185. https://doi.org/10.1038/s41562-020-00951-3 |
| SPoSE files and triplets as distributed (osf.io/f5rn6); the refits of SI Appendix, Fig. S16 | Hebart, M. N., Contier, O., Teichmann, L., Dickter, A. H., Zheng, C. Y., Kidder, A., Corriveau, A., Vaziri-Pashkam, M., & Baker, C. I. (2023). THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior. *eLife*, 12, e82580. https://doi.org/10.7554/eLife.82580 |
| SPoSE model | Zheng, C. Y., Pereira, F., Baker, C. I., & Hebart, M. N. (2019). Revealing interpretable object representations from human behavior [Conference paper]. *International Conference on Learning Representations (ICLR 2019)*. https://doi.org/10.48550/arXiv.1901.02915 |
| THINGS memorability scores and categories (osf.io/5a7z6); the trial-level analyses of SI Appendix, Sections 1 to 5 | Kramer, M. A., Hebart, M. N., Baker, C. I., & Bainbridge, W. A. (2023). The features underlying the memorability of objects. *Science Advances*, 9(17), eadd2981. https://doi.org/10.1126/sciadv.add2981 |
| Monkey IT component scores (`shared/things/monkey_features.csv`) | Papale, P., Wang, F., Self, M. W., & Roelfsema, P. R. (2025). An extensive dataset of spiking activity to reveal the syntax of the ventral stream. *Neuron*, 113(4), 539-553.e5. https://doi.org/10.1016/j.neuron.2024.12.003 |
| Face memorability database (SI Appendix, Section 6; not included) | Bainbridge, W. A., Isola, P., & Oliva, A. (2013). The intrinsic memorability of face photographs. *Journal of Experimental Psychology: General*, 142(4), 1323-1334. https://doi.org/10.1037/a0033872 |
| Scene memorability data (SI Appendix, Section 6) | Isola, P., Xiao, J., Torralba, A., & Oliva, A. (2011). What makes an image memorable? In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 145-152). https://doi.org/10.1109/CVPR.2011.5995721 |
| REM model (modules 01, 02, A2a to A2e, A3a and A3c) | Shiffrin, R. M., & Steyvers, M. (1997). A model for recognition memory: REM, retrieving effectively from memory. *Psychonomic Bulletin & Review*, 4(2), 145-166. https://doi.org/10.3758/BF03209391 |
| Context experiment simulated in Fig. 2 | Bylinskii, Z., Isola, P., Bainbridge, C., Torralba, A., & Oliva, A. (2015). Intrinsic and extrinsic effects on image memorability. *Vision Research*, 116, 165-178. https://doi.org/10.1016/j.visres.2015.03.005 |
| Hybrid-similarity exemplar model (module A2f) | Nosofsky, R. M., & Zaki, S. R. (2003). A hybrid-similarity exemplar model for predicting distinctiveness effects in perceptual old-new recognition. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 29(6), 1194-1209. https://doi.org/10.1037/0278-7393.29.6.1194 |
| Hybrid-similarity exemplar model (module A2f) | Knapp, B. R., Nosofsky, R. M., & Busey, T. A. (2006). Recognizing distinctive faces: A hybrid-similarity exemplar model account. *Memory & Cognition*, 34(4), 877-889. https://doi.org/10.3758/BF03193434 |
| Generalized context model, the special case of the hybrid model in module A2f | Nosofsky, R. M. (1986). Attention, similarity, and the identification-categorization relationship. *Journal of Experimental Psychology: General*, 115(1), 39-57. https://doi.org/10.1037/0096-3445.115.1.39 |
| d′ and d_a (module A1 and the signal-detection code in `shared/`) | Macmillan, N. A., & Creelman, C. D. (2004). *Detection theory: A user's guide* (2nd ed.). Psychology Press. https://doi.org/10.4324/9781410611147 |
| The fixed z-ROC slope of 0.80 used for d_a | Didi-Barnea, C., Peremen, Z., & Goshen-Gottstein, Y. (2016). The unitary zROC slope in amnesics does not reflect the absence of recollection: Critical simulations in healthy participants of the zROC slope. *Neuropsychologia*, 90, 94-109. https://doi.org/10.1016/j.neuropsychologia.2016.05.026 |
| The log-linear correction of hit and false alarm rates, (k + 0.5) / (n + 1) (`shared/python/sdt.py`, `shared/matlab/sdt_rates.m`) | Hautus, M. J. (1995). Corrections for extreme proportions and their biasing effects on estimated values of d′. *Behavior Research Methods, Instruments, & Computers*, 27(1), 46-51. https://doi.org/10.3758/BF03203619 |
| The standard error of the Fisher z interval of a Spearman ρ (modules A4a, A4b and A6a) | Bonett, D. G., & Wright, T. A. (2000). Sample size requirements for estimating Pearson, Kendall and Spearman correlations. *Psychometrika*, 65(1), 23-28. https://doi.org/10.1007/BF02294183 |
| The embedding pipeline of the 14 deep-network spaces (`shared/embedding_extraction/`; SI Appendix, Sections 4 to 6) | Brady, T. F., & Robinson, M. M. (2026). *Predicting visual working memory errors with a priori measures of representational geometry from human and artificial systems*. PsyArXiv. https://doi.org/10.31234/osf.io/k3xu6_v1 |

The deep-network embeddings (AlexNet, VGG16-BN, ResNet18, InceptionV3, CORnet-S, DINOv3 ViT-B/16 and CLIP
ViT-B/32) are computed with the publicly released weights listed in `shared/embedding_extraction/README.md`;
cite the original model papers when you reuse them.
