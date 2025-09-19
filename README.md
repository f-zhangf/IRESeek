# IRESeek

## Folder Usage Instructions


### 'ProcessData/IRES_BPE_matrix/' : This directory is used to store the generated BPE `.npy` files.

### 'ProcessData/IRES_BPP_matrix/' : This directory is used to store the generated BPP `.npy` files.

### 'ProcessData/IRES_pkl/' : This directory is used to store the `.pkl` files used for prediction.

### 'ProcessData/IRES_Predict/': This directory is used to store the prediction result files.

### 'ProcessData/IRES_Seq/ : This directory is used to store the sequence files (.fa or .fasta) used for prediction.

## Installation of IRESeek and environment
Download the repository and create corresponding environment.
### Environment Preparation

* linux(eg. CentOS)

* python 3.9 +

* Anaconda

### BPfold Installation
* Use base pair motif library
```shell
pip3 install BPfold
```

### RNAfold Installation

#### Installation Method 1
RNAfold is part of the ViennaRNA package. Download the source code from the official website or GitHub
```shell
wget https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_7_x/ViennaRNA-2.7.0.tar.gz
tar -zxvf ViennaRNA-2.7.0.tar.gz
./configure
make
make install
```
#### Installation Method 2
ViennaRNA is packaged in conda-forge and can be installed directly in a conda environment.
```shell
conda install -c conda-forge viennarna -y
```

### IRESeek Installation

```shell
git clone https://github.com/f-zhangf/IRESeek.git
cd ./IRESeek
conda env create -f IreSeek.yml
```
Then activate virtual enviroment

```shell
conda activate IRESeek
```
### Seqkit installation
Install SeqKit from bioconda

```shell
conda install -c bioconda seqkit -y
```

## USAGE

Run IreSeek.py file to predict IRES

```shell
python IreSeek.py -i input_file 
```

usage: IreSeek.py [-i] [-s] [-w] [-g]
```
optional arguments:
    -i, --input_file      Input seq file (.fasta or .fa file).
    -s, --split           True: the input sequences will be split. Fasle: the input sequences will be not split.
    -w, --window          If split = True,this para will be seted.
    -g, --interval        If split = True,this para also will be seted.
```

## Example

```
When no splitting is performed, the command is as follows:
    python IreSeek.py -i example.fa

When splitting is performed, the command is as follows:
    python IreSeek.py -i example.fa -s/--split -w/--window 174  -g/--interval 10
```

## Result
```
The obtained results will be saved as a CSV file and stored in 'ProcessData/IRES_Predict/'.

The prediction results consist of four columns: sequence ID, sequence, predicted label, and prediction score. The ID column also contains the predicted region.
```
