# Naive Scala Machine Learning Library

This is a self education project in Machine Learning algorithms, that was started from my other [sparx-ml](https://github.com/tupol/sparx-ml) project.

Besides the education part, some ideas and algorithms might be clearer outside the distributed computed environment (in this case Spark), plus, playing with it require no additional infrastructure.


## Algorithms explored

The algorithms were implemented using a similar pattern as the one used in Spark:
- a trainer class
- a predictor class

The trainer class, upon the call of the `train()` method will return the corresponding predictor class. 

### Linear Regression

Not much to say here, except the maybe worth to mention a naive optimized trainer, that will adjust the learning rate if too large.

### KMeans

Besides the standard KMeans algorithm there are a few nice ideas gathered here.

1. The prediction will essentially return a tuple of cluster number and distance to centroid.
2. The `KMeansGaussian` will attach also a probability and a probability by feature on prediction.
3. An algorithm for helping choosing the K number was also implemented.
4. An algorithm for guessing the K was also implemented

The probability is calculated based on the gaussian distribution as a product of probabilities for each dimension.
Since we have access to the probabilities for each dimension we can say which dimensions (features) contributed the most to a point being labeled as anomaly.

### Choosing the K

The algorithm for choosing the K is based on the idea that we can approximate the function that a certain model quality parameter (like sum of squared errors) follows based on the K.
Approximating this function allows us to calculate the derivatives and the second derivatives for an entire range of Ks and choosing one that has an acceptable derivative (e.g. close to 0).

I've made a lot of measurements in the [sparx-mllib](https://github.com/tupol/sparx-mllib/) project and all the graphs show the same basic evolution for different model quality measurements ([see the KDDCup sessions here](https://github.com/tupol/sparx-mllib/blob/master/docs/kddcup.md)).

The simple choose K function takes the following arguments:

**k_measure** A sequence of tuples of k and sse representing evolution of a measurement (e.g. SSE: Sum of Squared Errors) over k.
It is recommended to use evenly spread ks (like 2, 10, 20, 30, 40....). 
At the same time it is recommended that the first K to be the smallest acceptable K, which is 2.

**epsilon** The acceptable decrease ratio (derivative) of the measurements for every k. 
From the integration test the optimal value seems to be 0.0003, which works very well for small values of K (smaller than 10), and reasonably well for large values of K.


**maxK** Having a K equal or greater than the data set itself does not make a lot of sense, so as an extra measure, we specify it.

```
def chooseK(k_measure: Seq[(Int, Double)], epsilon: Double = 0.005, maxK: Int = 500): Int
```
 
### Guessing the K

Starting with the `chooseK()` function the next logical step is to take over and try to train a sequence of models, for a sequence of K and apply the choose K function.

The first attempt was to use the Fibonacci sequence for K, but the grow rate is too large, and to best approximate the measurements function we need more measurements, but still not too many.

The current implementation uses a custom sequence, based of powers of x like the following:
2, 5, 8, 16, 25, 36, 49, 64, 81, 100, 125, 216, 343, 512, 729, 1000, 1331, 1728, 2197, 2744, 3375, 4096, 4913, 5832, 6859, 8000
The Fibonacci sequence would look like the following, so quite similar, but quite spread apart for large numbers:
2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946

For each K in the sequence above, a model is generated, by training multiple times a model for each K and choosing the best model, starting with the premise that smaller is better.

The step before will essentially generate a sequence of measurements for different Ks, so the next natural step is to call the `chooseK()` function and find K,

```
def guessK(data: Seq[Point], runs: Int = 3, kMeansTrainerFactory: (Int) => KMeansTrainer, epsilon: Double = 0.05, maxK: Int = 500)
```
 
The same limitations pointed out at the `chooseK()` apply here as well, so the algorithm will not produce very good results for when the actual number of visible clusters is small.
 
## TODO

- [ ] improve the unit tests and documentation
- [ ] add some naive plotting functionality
- [ ] refactor the code
- [ ] improve the framework
- [ ] think AKKA
- [ ] add entropy and purity for classification problems (inclusing KMeans with labeled data)

## References

- [Machine Learning by Stanford University on Coursera](https://www.coursera.org/learn/machine-learning)
