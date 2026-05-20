#!/usr/bin/env python3

import sys

def state_to_num(state):
    num = 0
    for (val_i, val) in enumerate(state):
        num |= val << val_i

    return num

def num_to_state(num):
    return [(num >> i) & 1 for i in range(11)]

def mirror(state):
    UL, U, UR, LL, L, C, R, RR, DL, D, DR = state
    return [UR, U, UL, RR, R, C, L, LL, DR, D, DL]

def map_state(state, direction):
    if direction == 'right':
        state = mirror(state)

    UL, U, UR, LL, L, C, R, RR, DL, D, DR = state

    if C != 0:
        moves_down = D == 0
        moves_downleft = D != 0 and DL == 0 and L == 0
        moves_downright = D != 0 and DL != 0 and DR == 0 and R == 0 and RR == 0

        if moves_down or moves_downleft or moves_downright:
            return 0
        else:
            return 1
    else:
        fill_from_up = U != 0
        fill_from_upright = U == 0 and UR != 0 and R != 0
        fill_from_upleft = U == 0 and UR == 0 and UL != 0 and L != 0 and LL != 0

        if fill_from_up or fill_from_upright or fill_from_upleft:
            return 1
        else:
            return 0

    print('Error: all cases should have been handled above')
    sys.exit(1)

def map_vertical_only(state, direction):
    if state[1] != 0:
        return 1

    if state[9] != 0:
        return state[5]

    if state[9] == 0 and state[1] == 0:
        return 0

    print('Error: all cases should have been handled above')
    sys.exit(1)

print('mapped values for priority left:')
print([map_state(num_to_state(num), 'left') for num in range(2**11)])

print('mapped values for priority right:')
print([map_state(num_to_state(num), 'right') for num in range(2**11)])

print('mapped values for vertical only:')
print([map_vertical_only(num_to_state(num), 'right') for num in range(2**11)])
