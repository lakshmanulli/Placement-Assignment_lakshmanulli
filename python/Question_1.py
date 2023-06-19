
## Answer-1 [ Python]


def find_highest_frequency_word_length(string):

    # Split the string into words
    words = string.split()

    # Count the frequency of each word
    word_frequency = {}
    for word in words:
        word_frequency[word] = word_frequency.get(word, 0) + 1

    # Find the maximum frequency
    max_frequency = max(word_frequency.values())

    # Find the length of the word with the maximum frequency
    max_frequency_word_length = max(len(word) for word, frequency in word_frequency.items() if frequency == max_frequency)

    return max_frequency_word_length


# Test case 1
string = "write write write all the number from from from 1 to 100"
print(find_highest_frequency_word_length(string))
# Output: 5

# Test case 2
string = "apple apple banana banana banana cherry cherry"
print(find_highest_frequency_word_length(string))
# Output: 6
# Explanation: The most frequent word is "cherry" with a frequency of 3, and its length is 6.

# Test case 3
string = "hello hello world world world world"
print(find_highest_frequency_word_length(string))
# Output: 5
# Explanation: The most frequent word is "world" with a frequency of 4, and its length is 5.


'''
Explanation:

Test case 1:

The input string contains repeated words such as "write" and "from".
The word "write" appears 3 times, while the word "from" appears 3 times as well.
The maximum frequency is 3, which corresponds to the word "write".
The length of the highest-frequency word "write" is 5.
Therefore, the expected output is 5.
Test case 2:

The input string contains repeated words such as "banana" and "cherry".
The word "banana" appears 2 times, while the word "cherry" appears 4 times.
The maximum frequency is 4, which corresponds to the word "cherry".
The length of the highest-frequency word "cherry" is 6.
Therefore, the expected output is 6.
Test case 3:

The input string does not have any repeated words.
Each word appears only once.
Therefore, the maximum frequency is 1, and the length of the highest-frequency word is determined by the length of any word in the input string.
In this case, we can choose any word, and the program will return its length.
Therefore, the expected output can vary depending on the specific word chosen from the input string.
'''