#!/usr/bin/env python3

def ascii_to_binary(text):
    """Convert ASCII text to binary representation."""
    binary_result = []
    for char in text:
        binary_char = format(ord(char), '08b')
        binary_result.append(binary_char)
    return ' '.join(binary_result)

def main():
    """Convert 'Doo doo' to binary."""
    text = "Doo doo"
    print(f"Text: {text}")
    binary = ascii_to_binary(text)
    print(f"Binary: {binary}")

if __name__ == "__main__":
    main()