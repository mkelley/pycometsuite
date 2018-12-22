from ctypes import *
#everhart(double *x, double *v, double tFinal, double et, double beta,
#	     const int planets, double tol, double *minStep, double *DnFC,
#	     double *DnSeq, const int planetLookUp, const int closeApproaches)
everhart = CDLL('./everhart.so')
vector = c_double * 3
x = vector(5.49750435e+08,  -8.08647435e+07,   3.40086154e+07)
v = vector(-5.14032484,  7.00787504,  0.85952905)
tf = c_double(2000000)
et = c_double(0)
beta = c_double(0)
planets = c_int(0)
tol = c_double(1e-8)
mstep = c_double(0)
dnfc = c_double(0)
dnseq = c_double(0)
plu = c_int(0)
ca = c_int(0)

print(everhart.everhart(x, v, tf, et, beta, planets, tol, byref(mstep),
                        byref(dnfc), byref(dnseq), plu, ca))
print([x[i] for i in range(3)])
print([v[i] for i in range(3)])
