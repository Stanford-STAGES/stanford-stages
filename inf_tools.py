def myprint(str,*args):
    silent = True
    silent = False
    if not silent:
        print(str,*args) #  print(*args) - also works if we goto myprint(*args)
