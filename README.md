# ERxn

## Introduction
The enzyme reaction dataset is desgined for 
machine learning applications using enzymes
based on the [ENZYME](https://enzyme.expasy.org/) 
repository of information relative to the 
nomenclature of enzymes. It is 
primarily based on the recommendations of the 
Nomenclature Committee of the International 
Union of Biochemistry and Molecular Biology 
(IUBMB) and it describes each type of 
characterized enzyme for which an EC (Enzyme 
Commission) number has been provided 
[[More details](https://enzyme.expasy.org/enzyme_details.html) 
/ [References](https://enzyme.expasy.org/enzyme_ref.html)].

The ERxn dataset contains all enzymes which
have available fasta sequences and their 
corresponding EC numbers, reactions in
form of smiles and co-factors.

## Usage
___

### download raw data
downloads raw data from Expasy ENZYME database and places
it in data/raw/enzyme.dat

``` 
python 00_get_raw_data.py
```

### parse uniprot IDs and EC numbers
parses uniprot IDs and EC numbers from enzyme.dat and saves
them in a csv file under 
data/processed/01_uniprotID_and_EC_reduced.csv

``` 
python 01_parse_uniprotIDs_and_ECs.py
```
### download fasta files

downloads fasta files from 
http://www.uniprot.org/uniprot/ and places them in
data/fastas/<UNIPROTID>.fasta

``` 
python 02_get_fastas.py
```
