# ArMATH dataset

this dataset is the first large-scale dataset for solving Arabic Math Word Problem. There are 6000 samples and 883 templates. A template is an equation once the variables have been replaced with ordered placeholders. 

For the ease of use and uniform evaluation across future work, the dataset was randomly split into 5 folds; 1,200 samples in each. 

There are at most 15 variables and 10 constants. Constants are numbers that do not appear in the question body but the equation in at least five samples. These 10 constants are categorized as follows:

* Geometry: 3.14 and 0.5 
* Time: 12, 7, 60
* 0-4: numbers used in geometry, counting and facts.

The top 10 templates account for half of the samples. The table below shows the top templates and their freuqencies

| **Template** | **Frequency** |
| :---: | :---: |
| N0 / N1 | 631 |
| N0 - N1 | 491 |
| N0 * N1 | 481 |
| N1 * N0 | 361 |
| N0 + N1 | 254 |
| N1 / N0 | 245 |
| (N0 * N1) - N2 | 175 |
| N1 + N0 | 162 |
| (N0 / N1) - N2 | 123 |
| (N0 - N1) + N2 | 80 |


# Citations

comping up soon!
