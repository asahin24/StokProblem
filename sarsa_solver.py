import itertools
import random
import copy
import pandas as pd

def sarsa_solver(D, storage_capacity, h, p, production_capacity, operational_prob, c,
                 alpha=0.5, epsilon=0.2, gamma=0.95, max_iterations=100000):
    """
    SARSA ile stok kontrol problemi çözer.

    Parametreler:
    - D: {ürün: talep} dict
    - storage_capacity: {ürün: stok kapasitesi} dict
    - h: {ürün: elde tutma maliyeti} dict
    - p: {ürün: ceza maliyeti} dict
    - production_capacity: {hat: kapasite} dict
    - operational_prob: {hat: çalışabilirlik ihtimali} dict
    - c: {(ürün, hat): üretim maliyeti} dict
    - alpha: öğrenme oranı
    - epsilon: keşif oranı
    - gamma: iskonto faktörü
    - max_iterations: maksimum iterasyon sayısı

    Çıkış:
    - output_df: sonuçları içeren pandas DataFrame
    """

    m = len(D)  # Ürün sayısı
    n = len(production_capacity)  # Hat sayısı

    # Durum uzayı
    ranges = [range(capacity + 1) for capacity in storage_capacity.values()]
    states = list(itertools.product(*ranges))
    reward_dict = {state: 0 for state in states}

    # Feasible action'ları üret
    def generate_feasible_actions():
        actions = []

        def generate_for_line(line, remaining_cap, current_action):
            if line > n:
                actions.append(copy.deepcopy(current_action))
                return

            def generate_for_product(product, remaining_line_cap, current_action_line):
                if product > m:
                    generate_for_line(line + 1, production_capacity[line + 1] if line + 1 <= n else None,
                                       copy.deepcopy(current_action_line))
                    return

                for qty in range(remaining_line_cap + 1):
                    current_action_line[product][line] = qty
                    generate_for_product(product + 1, remaining_line_cap - qty, copy.deepcopy(current_action_line))

            generate_for_product(1, remaining_cap, copy.deepcopy(current_action))

        initial_action = {i: {j: 0 for j in range(1, n + 1)} for i in range(1, m + 1)}
        generate_for_line(1, production_capacity[1], initial_action)
        return actions

    actions = generate_feasible_actions()
    actions_tuples = [tuple((i, j, action[i][j]) for i in range(1, m + 1) for j in range(1, n + 1))
                      for action in actions]

    # Q fonksiyonunu küçük rastgele değerlerle başlat
    Q = {state: {action: random.uniform(-0.1, 0.1) for action in actions_tuples} for state in states}

    # SARSA algoritması
    for iteration in range(max_iterations):
        state = random.choice(states)

        # Epsilon-greedy başlangıç eylemi
        if random.random() < epsilon:
            action = random.choice(actions_tuples)
        else:
            action = max(actions_tuples, key=lambda a: Q[state][a])

        # Ortam simülasyonu
        line_status = [1 if random.random() < operational_prob[j] else 0 for j in range(1, n + 1)]

        # Gerçek üretim miktarları
        actual_production = {}
        for i in range(1, m + 1):
            prod_i = sum(action[i-1][j-1] * line_status[j-1] for j in range(1, n + 1))
            actual_production[i] = prod_i

        # Envanter güncelleme
        inventory = {}
        stockout = {}
        holding = {}
        for i in range(1, m + 1):
            inv = state[i-1] + actual_production[i] - D[i]
            if inv >= 0:
                stockout[i] = 0
                holding[i] = inv
            else:
                stockout[i] = -inv
                holding[i] = 0
            inventory[i] = max(0, min(inv, storage_capacity[i]))

        next_state = tuple(inventory[i] for i in range(1, m + 1))

        # Maliyet hesaplama
        production_cost = sum(c[(i, j)] * action[i-1][j-1] * line_status[j-1]
                              for i in range(1, m + 1) for j in range(1, n + 1))
        holding_cost = sum(h[i] * holding[i] for i in range(1, m + 1))
        penalty_cost = sum(p[i] * stockout[i] for i in range(1, m + 1))

        reward = - (production_cost + holding_cost + penalty_cost)
        reward_dict[state] = reward

        # Sonraki eylem (epsilon-greedy)
        if random.random() < epsilon:
            next_action = random.choice(actions_tuples)
        else:
            next_action = max(actions_tuples, key=lambda a: Q[next_state][a])

        # Q değeri güncelle
        Q[state][action] = (1 - alpha) * Q[state][action] + alpha * (reward + gamma * Q[next_state][next_action])

        # Bir sonraki adım için güncelle
        state, action = next_state, next_action

    # Optimal politika çıkar
    policy = {state: max(actions_tuples, key=lambda a: Q[state][a]) for state in states}

    # Çıktı verilerini hazırla
    output_data = []
    for state in states:
        action = policy[state]
        row = {
            'state': str(state),
            **{f'policy for product {i}': ','.join([str(action[i-1][j-1]) for j in range(1, n + 1)])
               for i in range(1, m + 1)},
            'Q_State': round(max(Q[state].values()), 4),
            'Reward': round(reward_dict[state], 4)
        }
        output_data.append(row)

    output_df = pd.DataFrame(output_data)
    return output_df
