numb = 1
limit = 128 * 1024 // 2
I, K, J = 1, 1, 1
while(True):
    if I*K + K*(J+1) + I*(J+1) <= limit:
        J += 1
    elif I*(K+1) + (K+1)*1 + I*1 <= limit:
        K += 1
        J = 1
    elif (I+1)*1 + 1*1 + (I+1)*1 <=limit:
        I += 1
        K, J = 1, 1
    else:
        break
    numb += 1
print('numb = ', numb)
