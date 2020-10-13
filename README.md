# baby-brains
Code supporting Ball et al (2020) biorXiv
Cortical morphology at birth reflects spatio-temporal patterns of gene expression in the fetal brain  

![babyBrains](/img/brains.png)

### Requirements
_Python 3.7.3_  
Required packages include: `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `seaborn`

All installed packages are shown in req.txt
To clone environment try: `conda create -n new environment --file req.txt`

_R 3.6.2_  
Required libraries: `mgcv`, `tidyverse`, `WCGNA`, `RColorBrewer`, `vegan`

### Data sources
_Neuroimaging_  
Neuroimaging data can be accessed via the [Developing Human Connectome Project](http://www.developingconnectome.org/second-data-release/)  

_RNA-Seq_  
Processed transcriptomic data can be accessed from [development.psychencode.org](http://development.psychencode.org/)    

GEO accession for other datasets:  
BrainCloud [GSE30272](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE30272)  
Mouse data [GSE89998](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE89998)  
Single-cell RNA [GSE103723](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE103723)  


### Analysis
Preprocessed imaging, transcriptomic and validation data can be found in `data/`.

Python scripts should be run in order, i.e.:  
`python A__principal_components.py`  
`python B__run_gene_models.py` etc  

Apart from some larger files that will need to be regenerated, the output of most scripts are already in `results/`

### Figures
Jupyter notebooks to reproduce main and supplemental figures are in `figures/` with the exception of network visualisations. For these, one can load the appropriate `*.graphml` files in Cytoscape.

### Paper
For further details on preprocessing and analysis, please see:  

G. Ball, J. Seidlitz, J. O’Muircheartaigh, R. Dimitrova, D. Fenchel, A. Makropoulos, D. Christiaens, A. Schuh, J. Passerat-Palmbach, J. Hutter, L. Cordero-Grande, E. Hughes, A. Price, J.V. Hajnal, D. Rueckert, E.C. Robinson, A.D. Edwards [Cortical morphology at birth reflects spatio-temporal patterns of gene expression in the fetal human brain](https://www.biorxiv.org/content/10.1101/2020.01.28.922849v3). _biorxiv_. 2020.
