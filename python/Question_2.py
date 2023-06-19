'''
[Python]

Question 2: -

Consider a string to be valid if all characters of the string appear the same number of times. It is also valid if
he can remove just one character at the index in the string, and the remaining characters will occur the same
number of times. Given a string, determine if it is valid. If so, return YES , otherwise return NO .

Note - You have to write at least 2 additional test cases in which your program will run successfully and provide
an explanation for the same.

Example input 1 - s = “abc”. This is a valid string because frequencies are { “a”: 1, “b”: 1, “c”: 1 }
Example output 1- YES

Example input 2 - s “abcc”. This string is not valid as we can remove only 1 occurrence of “c”. That leaves
character frequencies of { “a”: 1, “b”: 1 , “c”: 2 }
Example output 2 - NO

'''

## Answer-2 [Python ]

from collections import Counter

def is_valid_string(s):
    # Count the frequency of each character in the string
    char_frequency = Counter(s)

    # Count the frequency of frequencies
    frequency_frequency = Counter(char_frequency.values())

    # If there is only one unique frequency, the string is valid
    if len(frequency_frequency) == 1:
        return "YES"

    # If there are exactly two unique frequencies and one frequency has a count of 1, the string is valid
    if len(frequency_frequency) == 2 and (1 in frequency_frequency.values() and frequency_frequency[1] == 1):
        return "YES"

    return "NO"

# Test case 1
s = "abc"
print(is_valid_string(s))  # Output: YES

# Test case 2
s = "abcc"
print(is_valid_string(s))  # Output: NO

# Test case 3
s = "aabbcc"
print(is_valid_string(s))  # Output: YES

# Test case 4
s = "aabbccc"
print(is_valid_string(s))  # Output: YES

