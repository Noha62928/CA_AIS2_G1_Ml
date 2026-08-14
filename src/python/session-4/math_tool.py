"""
Author : Noha Nasser 
Version : 
"""
def factorial(num: int)->int:
    """
    calculate n! using recursion function 
    Args:
     num(int): user input the int number 
    Return:
     num(int): return the factorial 
    Example: 
    factorial(5) = 120 
    """
    if num == 0:
        return 1 
    return num * factorial(num - 1) 

def is_prime(num: int)->bool:
    """
    this function check for the number is prime or not 
    Args: 
        num(int): number for check 
    Return:
        bool: True if that prime or False if that not prime
    Example: 
        is_prime(17)-> True 
    """
    if num < 2:
        return False
    for i in range (2, num):
        if num % i == 0: 
            return False 
    return False 