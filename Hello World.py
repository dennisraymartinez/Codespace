def say_hello():
    return "Hello, big guy"

# Simple test
if __name__ == "__main__":
    assert say_hello() == "Hello, big guy"
    print("Test passed!")