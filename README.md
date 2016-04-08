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

The probability is calculated based on the gaussian distribution as a product of probabilities for each dimension.
Since we have access to the probabilities for each dimension we can say which dimensions (features) contributed the most to a point being labeled as anomaly.

The algorithm for choosing the K is based on the idea that we can approximate the function that a certain model quality parameter (like sum of squared errors) follows based on the K.
Approximating this function allows us to calculate the derivatives and the second derivatives for an entire range of ks and choosing one that has an acceptable derivative (e.g. close to 0)
 
 
## TODO

- [ ] improve the unit tests and documentation
- [ ] add some naive plotting functionality
- [ ] refactor the code
- [ ] improve the framework
- [ ] think AKKA
- [ ] add entropy and purity for classification problems (inclusing KMeans with labeled data)

## References

- [Machine Learning by Stanford University on Coursera](https://www.coursera.org/learn/machine-learning)
