# METHODOLOGY: Introducing LEBRON: Longevity Estimate Based on Recurrent Optimized Network

[Link to blog post.](https://dribbleanalytics.blog/2020/06/lebron-longevity-metric/)

## All-NBA prediction

LEBRON uses an LSTM to model a player's All-NBA probability throughout his career. So, the first step was to generate All-NBA probabilities for each historical player.

Our data set consists of all players whose career started on or after the 1979-1980 season (introduction of the 3-point line). We did not include data from the 2019-20 season, so our final season is the 2018-19 season.

With this data, we created 3 tree-based models (random forest, gradient boost, and extreme gradient boost) to predict All-NBA probabilities. The models took the following inputs: games played, minutes played, PPG, RPG, AST, VORP, and WS. First, we tested the models like traditional models by splitting the data into train/test sets to evaluate performance. Then, once we found that the models performed well, we performed a type of leave-one-out cross-validation to generate predictions for all historical players. For each year X, we trained the models on data from all years except for year X, then used those models to generate All-NBA probabilities for year X. We repeated this process for each year. Our final output was the average probability of the 3 models.

The All-NBA models could probably be improved slightly with different features, hyperparameter tuning, etc. However, these models performed very well, so we chose not to pursue this.

## LEBRON creation

We excluded players who played fewer than 4 years as of the 2018-19 season (so no Embiid, Doncic, etc.).

After training the model, we used it to predict next year's All-NBA probability for a player given their entire career sequence of All-NBA probabilities. We repeated this process for each year until each player reached the maximum career length of 21 seasons.

## Results

For a more detailed discussion of the results, view the blog post (link at the top). A Google Sheet containing all the results can be found [here](https://docs.google.com/spreadsheets/d/1fEZt05UYflNOvSE-2UDipWR1PN61dtVmbrlqJ-nz54o/edit#gid=0). Furthermore, an R Shiny app that helps visualize year-by-year changes in LEBRON can be found [here](https://dribbleanalytics.shinyapps.io/lebron-longevity-metric/).
