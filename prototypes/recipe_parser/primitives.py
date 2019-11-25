

def action_A(arg):
    arg.val *= 2
    print('[action_A] I multiply the input by 2! val={}'.format(arg.val))
    return arg

def action_B(arg):
    arg.val -= 2
    print('[action_B] I subtract the input by 2! val={}'.format(arg.val))
    return arg

def action_C(arg):
    arg.val -= 10
    print('[action_C] I subtract the input by 10! val={}'.format(arg.val))
    return arg

def action_D(arg):
    print('[action_D] I do nothing ~')
    return arg