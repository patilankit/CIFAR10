# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")


#-------------------------------------------------------- Write a seperate GF calculation function
def gf(A,B,N):
    if N == 0: return A;
    elif N == 1: return B;
    else:
        return (gf(A,B,(N-1)) + gf(A,B,(N-2)))              #Recursion

# a = gf(3,4,5);
# print a;


def solution(A, B, N):
    # write your code in Python 2.7
    return gf(A,B,N)%(1000000007)

A = [1, 3, 6, 4, 1, 2];
# A = [1,2,3]
#A = [-2,-3]
A.sort()


b= 0;
for i in range(1,max(A) + 1):
    if i in A:pass;
    else: b = i
if b==0: b = (max(A) + 1);
if b<0: b = 1;
print b