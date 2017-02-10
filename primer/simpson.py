''' An integral of a funtion f(x) can be approximated as :
        ((b-a)/3*n) * (f(a) + f(b) + 4*summation(f(a+(2*i - 1)*h)) + 2*summation(f(a+2*i*h)))

    Here, h = (b-a)/n and n must be an even integer.
    We make a function Simpson(f, a, b, n=500) that returns the formula above.
    We then apply the formula to (3/2*integral(sin(x)**3)) from 0 to pi, which has the exact value 2, for n = 2,6,12,100,500.

'''

    
