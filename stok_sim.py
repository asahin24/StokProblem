import os
import csv
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
from value_iteration_solver_hoca import value_iteration_solver
from sarsa_solver import sarsa_solver

urun_sayisi = 3
hat_sayisi = 3
sarsa_tekrar_sayisi = 5

input_excel = "Problems.xlsx"
df = pd.read_excel(input_excel, decimal=',')
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

def append_to_summary_csv(problem_no, vi_time, sarsa_time, output_dir="outputs"):
    summary_csv_path = os.path.join(output_dir, "summary_results.csv")
    file_exists = os.path.isfile(summary_csv_path)

    with open(summary_csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Problem No', 'Value Iteration Süresi (dk)', 'SARSA Süresi (dk)'])
        writer.writerow([problem_no, round(vi_time, 2), round(sarsa_time, 2)])


def solve_problem(idx_row_tuple):
    idx, row = idx_row_tuple
    print(f"\nProblem {idx+1} çözümüne başlanıyor...")

    D = {i: int(row[f'D{i}']) for i in range(1, urun_sayisi+1)}
    storage_capacity = {i: int(row[f'S{i}']) for i in range(1, urun_sayisi+1)}
    h = {i: float(row['h']) for i in range(1, urun_sayisi+1)}
    p = {i: float(row['p']) for i in range(1, urun_sayisi+1)}
    production_capacity = {j: int(row[f'ÜH{j}']) for j in range(1, hat_sayisi+1)}
    operational_prob = {j: float(row[f'Oper_Prob {j}']) for j in range(1, hat_sayisi+1)}

    c = {}
    for i in range(1, urun_sayisi+1):
        for j in range(1, hat_sayisi+1):
            c[(i, j)] = float(str(row[f'Ü({i},{j})']).replace(',', '.'))

    start_time_vi = time.time()
    output_df_vi = value_iteration_solver(D, storage_capacity, h, p, production_capacity, operational_prob, c, max_iterations=10)
    end_time_vi = time.time()
    elapsed_minutes_vi = (end_time_vi - start_time_vi) / 60

    reward_list = []
    total_sarsa_time = 0
    for tekrar in range(sarsa_tekrar_sayisi):
        start_time_sarsa = time.time()
        output_df_sarsa_temp = sarsa_solver(D, storage_capacity, h, p, production_capacity, operational_prob, c, max_iterations=20)
        end_time_sarsa = time.time()
        total_sarsa_time += (end_time_sarsa - start_time_sarsa)
        reward_list.append(output_df_sarsa_temp['Reward'])

    average_rewards = sum(reward_list) / sarsa_tekrar_sayisi
    final_output_df_sarsa = output_df_sarsa_temp.copy()
    final_output_df_sarsa['Reward'] = average_rewards
    elapsed_minutes_sarsa = total_sarsa_time / 60

    output_df_vi['VI_Süre (dk)'] = round(elapsed_minutes_vi, 2)
    final_output_df_sarsa['SARSA_Süre (dk)'] = round(elapsed_minutes_sarsa, 2)

    output_filename = os.path.join(output_dir, f"Problem_{idx+1}.xlsx")
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        output_df_vi.to_excel(writer, sheet_name="ValueIteration", index=False)
        final_output_df_sarsa.to_excel(writer, sheet_name="SARSA", index=False)

    print(f"Problem {idx+1} VI süresi: {elapsed_minutes_vi:.2f} dk | SARSA süresi: {elapsed_minutes_sarsa:.2f} dk")

    append_to_summary_csv(idx + 1, elapsed_minutes_vi, elapsed_minutes_sarsa, output_dir)


if __name__ == "__main__":
    # Problemleri hazırla
    problems = list(df.iterrows())

    # Çekirdek sayısı kadar paralel çalıştır
    with Pool(processes=cpu_count()) as pool:
        pool.map(solve_problem, problems)

    print(f"\nTüm çıktılar '{output_dir}/' klasörüne kaydedildi.")
