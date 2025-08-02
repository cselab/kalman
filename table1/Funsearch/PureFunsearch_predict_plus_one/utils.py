def seen(a, b):
    a = a.tobytes()
    b = b.tobytes()
    ans = (a, b) in Hash
    if not ans:
        Hash.add((a, b))
    return ans


def seen(a):
    a = a.tobytes()
    ans = (a) in Hash
    if not ans:
        Hash.add((a))
    return ans
