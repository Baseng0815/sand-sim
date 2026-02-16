#!/usr/bin/env python3

import sys

def state_to_num(state):
    num = 0
    for (val_i, val) in enumerate(state):
        num |= val << val_i

    return num

def num_to_state(num):
    return [(num >> i) & 1 for i in range(11)]

def map_state(state, direction):
    all_top_empty = state[0] == 0 and state[1] == 0 and state[2] == 0
    any_bot_empty = state[8] == 0 or state[9] == 0 or state[10] == 0

    if state[5] != 0:
        # filled center

        if state[9] != 0:
            # filled below => fill
            return 1
        else:
            # empty below => empty
            return 0

    elif state[5] == 0:
        # empty center

        if state[1] != 0:
            # filled above => fill
            return 1
        else:
            # empty above => empty
            return 0

    # if all_top_empty:
    #     # nothing fills from above
    #     if any_bot_empty:
    #         # current value "falls down" => empty
    #         return 0
    #     else:
    #         # current value doesn't fall down => keep it
    #         return state[5]
    # elif state[1] != 0 and state[5] != 0 and any_bot_empty:
    #     # 2-tower with empty below => empty since the center falls down and up falls left or right to conserve mass
    #     return 0
    # else:
    #     # something fills from above, result depends on direction
    #     if direction == 'left':
    #         if state[2] != 0 and state[6] != 0:
    #             # up-right falls into current => full
    #             return 1
    #         elif state[0] != 0 and state[3] != 0 and state[4] != 0:
    #             # up-left falls but doesn't fall into left-out or left
    #             return 1
    #         elif state[1] != 0 and state[5] == 0:
    #             # up falls into empty current => full
    #             return 1
    #         else:
    #             return 0

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

# print('mapped values for priority right:')
# print([map_state(num_to_state(num), 'right') for num in range(2**11)])

# print('mapped values for vertical only:')
# print([map_vertical_only(num_to_state(num), 'right') for num in range(2**11)])
