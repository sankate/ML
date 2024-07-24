# Bagging and Random Forest

### How 63% data gets in bootstrapped sample
Let’s say we have n observations:

```Observation (O) = {x1, x2 , x3, x4…. xk...... xn}```

Suppose we are creating a bootstrap sample of size n. 
Bootstrap sample ```(B) = { } ```
Size = `n`.

In the first trial probability of each random element `xk` getting selected in `B = 1/n`

In the first trial probability of `xk` not getting selected in `B = 1-(1/n)`

As all the draws in a bootstrap sample are independent of each other. 

In all the n trials probability of `xk` not getting selected in $B = (1-(1/n))^n $ 
This is because all the draws in a bootstrap sample are independent events. 


Now when n tends to infinity the expression will change as follows: 
$$\lim\limits_n \rightarrow \infty (1-(1/n))^n = (1/e) ≅ 0.3678 $$

 

Hence the probability of a random element xk getting selected in `n` trials = `1-0.3678 = 0.6322`

So the probability of a random element `xk` getting selected in `n` trials = `0.6322`


For example:

Let’s say we have 100 observations in the data. The probability of a random observation not getting selected in a bootstrap sample of size `100` is:

$$(1 - 1 / 100)^100  = (1 - 0.01)^100  = 0.99100  = 0.366 $$

Hence the probability of a random observations getting selected in a bootstrap sample of size 100, in 100 trials is = `1 - 0.366 = 0.634`
