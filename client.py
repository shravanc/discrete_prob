import sys
import numpy as np
from lib.multinomial_nb import MyMultinomialNB


x = np.array([ 'a', 'a', 'n', 'n', 's', 's'])
y = np.array([ 1, 1, 0, 1, 1, 1])


nb = MyMultinomialNB()
nb.fit(x, y)

letter = sys.argv[1]

print(nb.predict_prob(letter))
print(nb.predict(letter))

