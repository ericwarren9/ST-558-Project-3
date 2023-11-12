# ST-558-Project-3

The purpose of this repo is to house all of the `files` and documents used in creating `Project 3`, which analyzes the `Diabetes Health Indicators` Dataset. A more detailed description is provided in the `Warren_Ellsworth_ST 558 Project 3.Rmd` file. 

The main thing to note is all analysis and modeling is done for each `education level`, which is itself a feature of the data set. That means there will be multiple reports for each education level. Along with the `.Rmd` file, which includes the code for the project, there are `5 .md` documents, one for each education level in the data set. Each of these documents were rendered as a `github_document` to produce the `.md` file. There is also a `.yml` file that gives a line of code corresponding to the `theme` of our github pages site, a `.csv` file for the data set used, and a `.Rproj` file to house the project in R Studio.

In conducting our analyis and modeling, various packages were used that include: `rmarkdown` (to make the automated reports), `tidyverse` (for a variety of data manipulation and visualization techniques), `scales` (formatting on our visualizations), `plyr` (to transform contingency tables), `GGally` (a way to make our plots more interactive -- in our case we did this to get the percentages on the grpahs to update with each facet), `caret` (to do our advance models), `MASS` (helps create and tune our Linear Discriminant Analysis model), and `e1071` (helps tune our Support Vector Machines model).

We also wanted to include the render code here for people to be able to access and use when they want to make their automated reports like we did. The R code is below.

educationIDs <- unique(diabetes$Education) #get unique education

output_file <- paste0(educationIDs, ".md") #create filenames

params = lapply(educationIDs, FUN = function(x){list(Education = x)}) #create a list for each team with just the education name parameter

reports <- tibble(output_file, params) #put into a data frame 

library(rmarkdown) # Import library

apply(reports, MARGIN = 1,FUN = function(x){render(input = "Warren_Ellsworth_ST 558 Project 3.Rmd", output_file = x[[1]], params = x[[2]], output_format = "github_document", output_options = list(html_preview = FALSE, toc = TRUE, toc_depth = 3, toc_float = TRUE))}) # Make a function to render the md files

Lastly, below are `links` to each `.html` file of the generated analyses for each education level.  

* Analysis for [College 4 years or more -- College graduate](College 4 years or more -- College graduate.html)
* Analysis for [College 1 year to 3 years -- Some college or technical school](College 1 year to 3 years -- Some college or technical school.html)  
* Analysis for [Grade 12 or GED -- High school graduate](Grade 12 or GED -- High school graduate.html)
* Analysis for [Grades 9 through 11](Grades 9 through 11.html)
* Analysis for [Never attended school or only kindergarten or Grades 1 through 8](Never attended school or only kindergarten or Grades 1 through 8.html)  
