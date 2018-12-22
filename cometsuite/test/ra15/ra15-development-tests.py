import numpy as np

ci = 0
for j in range(1, 8):
    for i in range(2, j + 1):
        print('b[:, {}] += c[{}] * temp'.format(i - 2, ci))
        ci += 1
    print('\n')


ri = 0
C = 5
g = np.arange(6)
r = np.arange(21)
for j in range(1, 8):
    print(j, ri, ri + j - 2)
    rr = np.cumprod(r[ri:ri + j - 1][::-1])[::-1]
    print(g[:j-1] * rr)
    for i in range(len(rr)):
        print('{} * {} + '.format(i, rr[i]), end='')
        ri += 1
    print('\n')

# gk = np.arange(3)
# g = np.arange(21).reshape((3, 7))
# r = np.arange(21)
# ri = 0
# if j == 1:
#     g[:, 0] = gk
# else:
#     rr = np.cumprod(r[ri:ri + j - 1][::-1])[::-1]
#     g[:, j - 1] = gk * rr[0] - np.sum(gk * rr, 1)
# 


#            e[0] = s[0] * np.sum(np.arange(1, 8)* b, 1)
#            e[1] = s[1] * np.sum(np.cumsum(np.arange(6) + 1) * b[1:], 1)
#            e[2] = s[2] * np.sum(np.cumsum((arange(6) * (arange(6) + 1)) / 2) * b[2:], 1)
#
#      e[0][k] = s[0]*( 7.0*b[6][k] +  6.0*b[5][k] +  5.0*b[4][k] + 4.0*b[3][k] + 3.0*b[2][k] + 2.0*b[1][k] + b[0][k]);
#      e[1][k] = s[1]*(21.0*b[6][k] + 15.0*b[5][k] + 10.0*b[4][k] + 6.0*b[3][k] + 3.0*b[2][k] + b[1][k]);
#      e[2][k] = s[2]*(35.0*b[6][k] + 20.0*b[5][k] + 10.0*b[4][k] + 4.0*b[3][k] +     b[2][k]);
#      e[3][k] = s[3]*(35.0*b[6][k] + 15.0*b[5][k] +  5.0*b[4][k] +     b[3][k]);
#      e[4][k] = s[4]*(21.0*b[6][k] +  6.0*b[5][k] +      b[4][k]);
#      e[5][k] = s[5]*( 7.0*b[6][k] +      b[5][k]);
#      e[6][k] = s[6]*(     b[6][k]);

emat = np.matrix([[1, 2, 3, 4, 5, 6, 7],
                  [0, 1, 3, 6, 10, 15, 21],
                  [0, 0, 1, 4, 10, 20, 35],
                  [0, 0, 0, 1, 5, 15, 35],
                  [0, 0, 0, 0, 1, 6, 21],
                  [0, 0, 0, 0, 0, 1, 7],
                  [0, 0, 0, 0, 0, 0, 1]])

b = np.arange(21).reshape((3, 7))
s = 2**np.arange(7)
print(emat * b.T)
print(((emat * b.T).A).T)
print(s * ((emat * b.T).A).T)


print('\n' * 2)
s = np.zeros(7)
s[0] = 0.5
s[1] = s[0] * s[0];
s[2] = s[1] * s[0];
s[3] = s[2] * s[0];
s[4] = s[3] * s[0];
s[5] = s[4] * s[0];
s[6] = s[5] * s[0];

ss = s[0]**np.arange(1, 8)

print(s)
print(ss)
