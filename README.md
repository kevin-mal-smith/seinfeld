# Seinfeld
--- 
## "No BeautifulSoup for you!"

How could anyone not like Seinfeld? It seems impossible. I for one discovered Seinfeld a little later than most. I was 10 when the final episode aired. So, I didn't start falling in love with the show until the seasons started coming out on DVD a few years later. Once I bought the first two seasons, I was hooked. 

I have seen every episode probably 10 times over, and I still laugh out loud with each rewatch. I just can't seem to get enough. Which brings me to my project. 

By now you have surely seen people who utilize machine learning to generate episodes of beloved TV shows, Seinfeld included. Usually these are funny because they are very clunky, and dont make much sense while still catching some of the familiar patterns of the shows in question. What I am setting out to do is generate an episode that I would actually enjoy based on my personal rankings of each episode. 

## Acquisition
---
### The Non-Fat Data

The first thing I need to do is generate data about my own personal preferences. I did this by grading each episode on three categories. 
| *Category* | *Explanation* |
| :---: | :---: |
|  Plot | How well does the story come together? |
| Quotability/Cultural Impact | Is there an element that I quote a lot, or one that became a cultural phenomenon outside of the show ie "yada yada" |
| laughs | This is as nerdy as it sounds. I counted my laughs. a short chuckle was .5, an out loud laugh was 1, and anything that made me laugh for longer than 2-3 seconds counts as 1.5 |
| Total Score | (plot * quotability) * (laughs / 23(run time)) |

For the sake of reproducability I will be including the scores that I generated in csv format.

## Prep
---
### "There's good data and there's bad data"

I cleaned up and prepped the **scripts.csv** & **episode_info.csv** files then joined them with the data that I generated to create a main database. Luckily the data was relatively clean, but I did have to drop some clearly erroneous data. for instance, if a line was clearly not dialogue, or was not associated with a character, it got dropped. Other than that I changed "Kessler" to "Kramer" because in the pilot episode the character was called Kessler, but it was changed to Kramer by the second epsiode. Now you can impress your friends with that bt of Seinfeld trivia, and we can move on to the next step which is creating a model to predict how well I will like an episode based on its script.

## Preference Modeling
---
### "Your bag of words are sublime..."

The first thing we need to do is create a new dataframe using the bag of words library. Once that is complete its realtively straightforward. I just finished watching season 5. So, I am treating the first 5 seasons as my test data, then I will treat 6-7 as my validate, and 8-9 as my test data. 

I used SciKit-Learn's MLP regressor to predict my total score with an RMSE of .0311. I have actually gotten much better results with different parameters in my model, but I am opting for lower performance on the training data in hopes that it translates to better performance on out of sample data.


