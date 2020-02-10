import numpy as np
import pandas as pd
import time

initial_S = [0, 2]
blocks = [3, 5]

N_STATES = 100
ACTIONS = ['cc', 'cp', 'pc', 'pp']   # available actions



EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor

MAX_EPISODES = 100  # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move

# Utility Function
def is_done(vehicle_coordinate):
	return (vehicle_coordinate > 8) or (vehicle_coordinate % 2 == 1)

def merge(lst1, lst2): 
    return [sub[item] for item in range(len(lst2)) 
                      for sub in [lst1, lst2]]

def split(merged):
	return [merged[::2], merged[1::2]]

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table

def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    first_coordinate = S[0]
    second_coordinate = S[1]
    if is_done(first_coordinate) and is_done(second_coordinate):
    	S_ = 'terminal'
    	R = 0

    else:

	    if is_done(second_coordinate):
	    	if A[0] == 'c':
	    		first_coordinate_ = first_coordinate + 2
	    	else:
	    		first_coordinate_ = first_coordinate + 1
	    	second_coordinate_ = second_coordinate

	    # If the first vehicle has parked:
	    elif is_done(first_coordinate):
	    	if A[1] == 'c':
	    		second_coordinate_ = second_coordinate + 2
	    	else:
	    		second_coordinate_ = second_coordinate + 1
	    	first_coordinate_ = first_coordinate

	    # None of these vehicle has parked, they have to both on the left lane
	    else:
	    	if A == 'cc':
	    		first_coordinate_ = first_coordinate + 2
	    		second_coordinate_ = second_coordinate + 2

	    	elif A == 'cp':
	    		first_coordinate_ = first_coordinate + 2
	    		second_coordinate_ = second_coordinate + 1

	    	elif A == 'pc':

	    		first_coordinate_ = first_coordinate + 1
	    		second_coordinate_ = second_coordinate + 2

	    	else:
	    		first_coordinate_ = first_coordinate + 1
	    		second_coordinate_ = second_coordinate + 1

	    S_ = [first_coordinate_, second_coordinate_]
	    R = -1

    if any(i in blocks for i in S_):
    	R = -100

    return S_, R


def change_char(s, p, r):
    a = list(s)
    a[p] = r
    return ''.join(a)

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = '---X--X--'
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print(interaction)
        time.sleep(2)
        
    else:

    	for elem in S:
    		if elem <= 8:
    			env_list = change_char(env_list, elem, 'o')
    	interaction = [env_list]
    	a = split(interaction)
    	left_lane = a[0]
    	right_lane = a[1]
    	print(left_lane)
    	print(right_lane)	
    	time.sleep(FRESH_TIME)

def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = [0, 2]
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S[0] * 10 + S[1], q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            # Q_predict value

            q_predict = q_table.loc[S[0] * 10 + S[1], A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_[0] * 10 + S_[1], :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S[0] * 10 + S[1], A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)

