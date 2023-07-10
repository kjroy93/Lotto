"""Main of the proyect, in order to obtain the recommended numbers. This data is the start point of for to make the lottery tickets"""

# Standard Libraries of Python
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP, getcontext
getcontext().prec = 5

# Dependencies
import pandas as pd
import numpy as np
np.set_printoptions(precision=5)

# Libraries made for this Proyect
from src.parse import Tickets
from numbers_scan import euromillions

lotto = Tickets(euromillions)
lotto.draw_skips()
lotto.skips_evaluation()
lotto.first_number()
lotto.suggested_numbers()