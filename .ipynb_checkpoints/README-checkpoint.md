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
