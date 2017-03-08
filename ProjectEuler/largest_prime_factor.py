
'''
The prime factors of 13195 are 5, 7, 13 and 29.

What is the largest prime factor of the number 600851475143 ?
'''

def largest_prime_factor(a):
    for i in range(2,a):
        if a%i != 0:
            i+=1
            print(i)

largest_prime_factor(13195)
