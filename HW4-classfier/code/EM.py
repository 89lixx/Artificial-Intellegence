import numpy as np
from scipy import stats
import math
observations = np.array([[1,0,0,0,1,1,0,1,0,1],
                         [1,1,1,1,0,1,1,1,1,1],
                         [1,0,1,1,1,1,1,0,1,1],
                         [0,1,1,1,0,1,1,1,0,1]])

def em_single(priors, observations):
    # 单次迭代
    counts = {'A':{'H': 0, 'T': 0}, 'B':{'H': 0, 'T': 0}}
    thetaA = priors[0]
    thetaB = priors[1]

    for observation in observations:
        obLen = len(observation)
        numH = observation.sum()
        numT = obLen - numH

        contributionA = stats.binom.pmf(numH, obLen, thetaA)
        contributionB = stats.binom.pmf(numT, obLen, thetaB)
        contribution = contributionA + contributionB
        weightA = contributionA / contribution
        weightB = contributionB / contribution

        #更新A、B的正反面次数
        counts['A']['H'] += weightA * numH
        counts['A']['T'] += weightA * numT
        counts['B']['H'] += weightB * numH
        counts['B']['T'] += weightB * numT

    newThetaA = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
    newThetaB = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])

    return [newThetaA, newThetaB]

def em(observations, prior):
    iteration = 0
    iterations = 1000
    tol = 1e-6
    while iteration < iterations:
        newPrior = em_single(prior, observations)
        change = np.abs(prior[0]- newPrior[0])
        if(change < tol):
            break
        else:
            prior = newPrior
            iteration += 1
    return [newPrior, iteration]

print(em(observations, [0.5,0.6]))




    
