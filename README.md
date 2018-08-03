# Graph Fits
The goal of this project is to find how close the Barvinok samples and the true
posterior distribution over the permutations are for a graph with adjacency
matrix B, that is a realisation from an extended Erdos-Renyi (EER) random graph
with parameter matrix A.

The true distribution can be found by considering each permutation P and
finding the probabilitiy P.T B P came from distribution EER(A).

The Barvinok samples are found by finding the optimal doubly stochastic matrix
Q such that
    || A - Q.T B Q ||_{Fro}
is minimised. This is a convex optimisation. We can then sample permutation
matrices by sampling a Gaussian random varaible x and finding the P that
minimises
    ||Px - Qx||_2
which can be shown to be the P that orders x in the same way as Qx.

## Testing
To test that the doubly stochastic optimisation works we can do two things:

1) Sample many random doubly stochastic matrices and make sure Q is the best.
2) For different random starting points, make sure they all converge to Q.

## Note
Can maybe do the doubly stochastic matrix using a KL type norm so it fits
more into a probability framework. I think I may have proved some result in
my thesis that this becomes
    || A - Q.T log(B) Q ||_{Fro}

## Note
Even if we cannot match the Barvinok sampling distribution to the true
posterior, we can still use it for making proposals to an MCMC algorithm for
actual sampling from the posterior.

By varying slightly a few coordinates of the x vector, it may even provide a
better proposal scheme than the Lyzinski one.

## Note
Should maybe also have some annealing on the EER graph to avoid 0 probabilities
to start with e.g.
    A' = c * A + (1-c) * ONES
where ONES is a matrix of ones. The c values can be gradually increased towards
1. This may even be equivalent to putting some uniform prior on A in an
expectation maximisation context.

## Note
Why even sample at all the Barvinok way. In an EM algorithm, all the samples
will be summed together anyway. If this makes the optimal matrix Q, then
the sampling was irrelevant. Of course we need to test it does indeed average
to the matrix Q.

## Note
The arithmetic mean of the posterior permutation graphs is the contribution
to the new EER in an EM scheme, for each observation. However, it should also
provide the log likelihood a given observation comes from a certain EER
distribution by multiplying by with log(B).
