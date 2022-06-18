# Easy-to-Read Sentence Generation for the Intellectual Disability

CS376 Team 4 Code Repository ([GITHUB](https://github.com/Nayoung-Oh/Easy_to_Read))

## Prepare wikilarge or wikismall dataset

1. Download the raw datasaet from DRESS repository ([DRESS_REPO](https://github.com/XingxingZhang/dress))
2. Unzip the file to wikilarge or wikismall folder
3. Run <code> python preprocess_data.py --data wikilarge </code> for wikilarge, <code> python preprocess_data.py --data wikismall </code> for wikismall

## Set up environment

1. Run <code> conda create --name <env> --file requirements.txt python==3.7 </code>
2. If you want to run the demo website, install Flask, too.
3. If you want to generate EASSE reports, follow the EASSE repository ([EASSE_REPO](https://github.com/feralvam/easse))

## How to train the baseline model (naive transformer)

ex) with wikilarge dataset

<code> python train.py --data wikilarge --model naive --loss none </code>

## How to train our own model (feature-based transformer)

ex) with wikilarge dataset, without weighted loss

<code> python train.py --data wikilarge --model feature --loss none </code>

ex) with wikilarge dataset, with weighted loss

<code> python train.py --data wikilarge --model feature --loss weighted </code>

## How to test the trained model (calculate cosine similarity)

ex) with wikilarge dataset, with weighted loss

<code> python test.py --data wikilarge --model feature --loss weighted --path PATH</code>

if you want to check the output of a specific sentence, use `simplify` function

If you have any questions, use **Issues**
