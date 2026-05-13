import sys
import time

print("Sync Test")
while True:
    try:
        val = input("Input: ")
        print(f"Got: {val}")
    except EOFError:
        break
    except KeyboardInterrupt:
        break
