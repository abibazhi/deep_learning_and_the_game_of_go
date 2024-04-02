from scipy.stats import binom_test

a = binom_test(12,20,0.5)
print(a)

b = (18,17,16,19,15, 18,13,18 ,19,19,16,15 ,18,17,16,14 ,12,15,16,15 ,19,19,14,16)

sum_won = 0
sum_total = 0
for index,value in enumerate(b):
    sum_won = sum_won + value
    sum_total = 32 * (index + 1)
    print(index,value,sum_won,sum_total,binom_test(sum_won, sum_total, 0.5))
