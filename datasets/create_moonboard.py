import numpy as np

# make train/test dividing reproducible for debugging
np.random.seed(0)

loadproblems = np.loadtxt('climbs.txt', dtype=np.uint8)
grades = np.loadtxt('grades.txt', dtype=np.uint8)

rows = 18
cols = 11
nproblems = len(grades)

problems = []
for i in range(nproblems):
    problems.append(loadproblems[i * rows:(i + 1) * rows])


# divide problems into training and test set
randidx = np.random.randint(nproblems, size=nproblems)
trainidx   = randidx[:int(4*nproblems/5)]
testidx    = randidx[ int(4*nproblems/5):]
trainproblems = np.take(problems, trainidx, axis=0)
traingrades   = np.take(grades  , trainidx, axis=0)
testproblems  = np.take(problems, testidx, axis=0)
testgrades    = np.take(grades  , testidx, axis=0)

print('trainproblems', len(trainproblems))
print('traingrades',   len(traingrades))
print('testproblems',  len(testproblems))
print('testgrades',    len(testgrades))

np.savez(
    'moonboard.npz', x_train=trainproblems, x_test=testproblems,
    y_train=traingrades, y_test=testgrades)
