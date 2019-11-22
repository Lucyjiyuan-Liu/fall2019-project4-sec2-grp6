# Project 4: Algorithm implementation and evaluation: Collaborative Filtering

Term: Fall 2019

+ Team: Section 2 Group 6
+ Project title: Collaborative Filtering: A Comparative Analysis

![image](figs/netflix0.jpg)

+ Team members
	+ Thomson Batidzirai
	+ Sen Dai
	+ Jie Jin 
	+ Jiyuan Liu
	+ Qiuyu Ruan
	+ Xiyi Yan
	
### [Project Description](doc/project4_desc.md)

Have you ever wondered why Netflix continues to dominate the movie and TV streaming industry since it's inception in the late 90's? Most people think that Netflix is able to keep most of it's competition at bay because of it's exceptionally large collection of original content which competitors cannot match. Indeed, this is true. However, we would like to also highlight Netflix's marketing strategy as another of it's major strengths. Netflix makes use of targeted marketing that makes full use of collaborative filtering techniques to satisfy clients by availing tailored content that match clients' tastes. This boasts customer loyalty and also easily entices new users; hence Netflix's continued dominance.

	
**Project summary:** 

We are assigned the task of ALS algorithm with temporal dynamics, along with KNN and KRR post processing using the data set on ratings of movie.We reduced the number of movies by deleting the movies with few ratings. First we implemented ALS algorithm with temporal dynamics and tune lambda by cross validation. After checking training and testing RMSE, we later choose lambda = 10 and interation = 1 to implement our algprithm. After getting the optimal q matrix for moveis, we used it to do KNN and KRR postprocessing. When implementing KRR, we also did a cross validation. Finally, we use the predictors obtained above to run linear regression.

**The assigned papers:**

1. [Improving regularized singular value decomposition for collaborative filtering](https://github.com/TZstatsADS/fall2019-project4-sec2-grp6/blob/master/doc/paper/P2%20Improving%20regularized%20singular%20value%20decomposition%20for%20collaborative%20filtering%20.pdf)
2. [Large-scale Parallel Collaborative Filtering for the Netflix Prize](https://github.com/TZstatsADS/fall2019-project4-sec2-grp6/blob/master/doc/paper/P4%20Large-scale%20Parallel%20Collaborative%20Filtering%20for%20the%20Netflix%20Prize.pdf)
3. [Collaborative Filtering with Temporal Dynamics](https://github.com/TZstatsADS/fall2019-project4-sec2-grp6/blob/master/doc/paper/P5%20Collaborative%20Filtering%20with%20Temporal%20Dynamics.pdf)
	
**Contribution statement**: 
+ All team members approve our work presented in this GitHub repository including this contributions statement. 
+ Jiyuan Liu and Xiyi Yan are major contributors who worked on the codes for ALS algorithm with temporal dynamics; also the codes for KNN and KRR postprocessing. Jiyuan Liu also worked on linear regression. Jie Jin worked on postprocssing understand. Sen Dai, Jie Jin, and Qiuyu Ruan were responsible for data cleaning, helping understanidng the algorithms,collecting useful references and prepraring for presentation. All of us helped writing notebook.
+ Thomson Batidzirai: Worked on updating the readme section. Was on the team on implementing the Alternative Least Squares with Temporal Dynamics regularization and KNN post processing 


Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
