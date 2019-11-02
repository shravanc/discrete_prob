import sys
import numpy as np
from lib.multinomial_nb import MyMultinomialNB


x = np.array([ 'a', 'a', 'n'])
y = np.array([ 1, 1, 0])


nb = MyMultinomialNB()
nb.fit(x, y)

letter = sys.argv[1]

print(nb.predict_prob(letter))
print(nb.predict(letter))

