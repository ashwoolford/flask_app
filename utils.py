import re

def is_bangla_word(word):
    # Check if the word consists only of Bangla characters
    return all('\u0980' <= char <= '\u09FF' for char in word)

def bangla_words_percentage(input_string):
    # Split the string into words using spaces and punctuation as delimiters
    words = re.findall(r'\S+', input_string)
    
    # Count the number of Bangla words
    bengali_word_count = sum(1 for word in words if is_bangla_word(word))
    
    # Total number of words
    total_word_count = len(words)
    
    # Handle the case where there are no words in the string
    if total_word_count == 0:
        return 0.0
    
    # Calculate the percentage of Bangla words
    return (bengali_word_count / total_word_count) * 100

# Test the function
test_string = "এটি     একটি        বাংলা  "
print(f"Bangla words percentage: {bangla_words_percentage(test_string):.2f}%")
