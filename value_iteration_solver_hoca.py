import itertools
import copy
import pandas as pd
import time


def value_iteration_solver(D, storage_capacity, h, p, production_capacity, operational_prob, c, gamma=0.9, epsilon=0.01, max_iterations=100):
    """
    Value Iteration ile stok problemi çözer.
    """
    convergence_status = None

    m = len(D)
    n = len(production_capacity)

    # Tüm olası envanter seviyeleri
    ranges = [range(capacity + 1) for capacity in storage_capacity.values()]
    states = list(itertools.product(*ranges))
    print(f"State sayisi : {len(states)}")

    # Feasible production actions
    def generate_feasible_actions():
        actions = []

        def generate_for_line(line, remaining_capacity, current_action):
            if line > n:
                actions.append(copy.deepcopy(current_action))
                return

            def generate_for_product(product, remaining_capacity_line, current_action_line):
                if product > m:
                    generate_for_line(line + 1, production_capacity[line + 1] if line + 1 <= n else None, copy.deepcopy(current_action_line))
                    return
                for qty in range(remaining_capacity_line + 1):
                    current_action_line[product][line] = qty
                    generate_for_product(product + 1, remaining_capacity_line - qty, copy.deepcopy(current_action_line))

            generate_for_product(1, remaining_capacity, copy.deepcopy(current_action))

        initial_action = {i: {j: 0 for j in range(1, n + 1)} for i in range(1, m + 1)}
        generate_for_line(1, production_capacity[1], initial_action)
        return actions

    actions = generate_feasible_actions()
    print(f"Olusturulan aksiyon sayisi : {len(actions)}")

    # Value function ve policy başlat
    V = {state: 0 for state in states}
    policy = {state: None for state in states}

    for iteration in range(max_iterations):
        delta = 0
        new_V = {}

        print(f"iterasyon: {iteration+1}")
        start_iter_time = time.time()

        for state in states:
            V_state = float('inf')
            best_action = None

            for action in actions:
                expected_cost = 0.0
                for line_status in itertools.product([0, 1], repeat=n):
                    prob_status = 1.0
                    for j in range(1, n + 1):
                        prob_status *= operational_prob[j] if line_status[j - 1] == 1 else 1 - operational_prob[j]
                    actual_production = {}
                    for i in range(1, m + 1):
                        prod_i = sum(action[i][j] * line_status[j - 1] for j in range(1, n + 1))
                        actual_production[i] = prod_i
                    inventory = {}
                    stockout = {}
                    holding = {}
                    for i in range(1, m + 1):
                        inv = state[i - 1] + actual_production[i] - D[i]
                        if inv >= 0:
                            stockout[i] = 0
                            holding[i] = inv
                        else:
                            stockout[i] = -inv
                            holding[i] = 0
                        inv = min(inv, storage_capacity[i])
                        inventory[i] = max(0, inv)
                    production_cost = sum(c[(i, j)] * action[i][j] * line_status[j - 1] for i in range(1, m + 1) for j in range(1, n + 1))
                    holding_cost = sum(h[i] * holding[i] for i in range(1, m + 1))
                    penalty_cost = sum(p[i] * stockout[i] for i in range(1, m + 1))
                    immediate_cost = production_cost + holding_cost + penalty_cost
                    next_state = tuple(inventory[i] for i in range(1, m + 1))
                    future_cost = V[next_state]
                    total_cost = immediate_cost + gamma * future_cost
                    expected_cost += prob_status * total_cost
                if expected_cost < V_state:
                    V_state = expected_cost
                    best_action = action
                    
            if iteration == 0: 
                for state in states: 
                    new_V[state] = V_state * (1/(1-gamma))/2
            else:
                new_V[state] = V_state
    
            policy[state] = best_action
            delta = max(delta, abs(V[state] - new_V[state]))
            
        V = new_V

        end_iter_time = time.time()
        print(f"iterasyon süresi: {end_iter_time - start_iter_time} saniye")
        print(f"Delta: {delta}")

        if delta < epsilon:
            convergence_status = "Yakınsama"
            break

    if convergence_status is None:
        convergence_status = "Max İterasyon"

    output_data = []
    for state in states:
        action = policy[state]
        row = {'state': str(state)}
        for i in range(1, m + 1):
            row[f'policy for product {i}'] = ','.join([str(action[i][j]) for j in range(1, n + 1)])
        row['V_state'] = V[state]
        row['Çıkış Sebebi'] = convergence_status
        row['Delta'] = delta
        output_data.append(row)

    output_df = pd.DataFrame(output_data)
    return output_df
