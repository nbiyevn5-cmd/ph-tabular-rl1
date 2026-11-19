"""
Kapsamlı Eğitim, Bileşen Analizi (Ablation), 50-Bölümlük Monte Carlo Testi,
Ölü Zaman Hassasiyet Analizi ve 6 Akademik Grafik Üretim Betiği.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from ph_env import PHNeutralizationEnv
from agents import (
    NonUniformDiscretizer,
    TabularQLearningAgent,
    TabularSARSAAgent,
    IndustrialPIDController,
)


class UniformDiscretizer:
    """Ablation 1: Titrasyon eğrisini dikkate almayan eşit aralıklı ayrıklaştırıcı."""

    def __init__(self):
        self.ph_bins = np.linspace(1.0, 13.0, 14)  # 15 eşit bölge (0.86 pH genişlik)
        self.dph_bins = np.linspace(-0.5, 0.5, 6)
        self.flow_bins = np.array([1.35, 1.85])
        self.pipe_bins = np.array([-0.18, -0.03, 0.03, 0.18])
        self.dims = (15, 7, 3, 2, 5)
        self.n_states = int(np.prod(self.dims))

    def encode(self, obs: dict) -> int:
        ph_hist = obs["ph_history"]
        dph = ph_hist[-1] - ph_hist[-2]
        level_idx = 1 if obs["level"] >= 1.05 else 0
        net_pipe = sum(ub - ua for ua, ub in obs["past_actions"])
        return int(
            np.ravel_multi_index(
                (
                    int(np.digitize(ph_hist[-1], self.ph_bins)),
                    int(np.digitize(dph, self.dph_bins)),
                    int(np.digitize(obs["q_in"], self.flow_bins)),
                    level_idx,
                    int(np.digitize(net_pipe, self.pipe_bins)),
                ),
                self.dims,
            )
        )


class NoPipeMemoryDiscretizer(NonUniformDiscretizer):
    """Ablation 2: Ölü zaman boru hattı hafızası (past_actions) olmayan ayrıklaştırıcı."""

    def __init__(self):
        super().__init__()
        self.dims = (
            len(self.ph_bins) + 1,
            len(self.dph_bins) + 1,
            len(self.flow_bins) + 1,
            2,
        )
        self.n_states = int(np.prod(self.dims))

    def encode(self, obs: dict) -> int:
        ph_hist = obs["ph_history"]
        dph = ph_hist[-1] - ph_hist[-2]
        level_idx = 1 if obs["level"] >= 1.05 else 0
        return int(
            np.ravel_multi_index(
                (
                    int(np.digitize(ph_hist[-1], self.ph_bins)),
                    int(np.digitize(dph, self.dph_bins)),
                    int(np.digitize(obs["q_in"], self.flow_bins)),
                    level_idx,
                ),
                self.dims,
            )
        )


def train_q(env: PHNeutralizationEnv, disc, episodes: int = 25000, seed: int = 42):
    agent = TabularQLearningAgent(
        n_states=disc.n_states,
        n_actions=env.n_actions,
        alpha=0.12,
        gamma=0.96,
        eps_decay=0.99975,
        seed=seed,
    )
    rewards_hist = []
    for _ in range(episodes):
        obs = env.reset(randomize=True)
        state = disc.encode(obs)
        ep_rew = 0.0
        done = False
        while not done:
            action = agent.select_action(state, greedy=False)
            next_obs, reward, done, _ = env.step(action)
            next_state = disc.encode(next_obs)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_rew += reward
        agent.decay_epsilon()
        rewards_hist.append(ep_rew)
    return agent, np.array(rewards_hist)


def train_sarsa(env: PHNeutralizationEnv, disc, episodes: int = 25000, seed: int = 99):
    agent = TabularSARSAAgent(
        n_states=disc.n_states,
        n_actions=env.n_actions,
        alpha=0.12,
        gamma=0.96,
        eps_decay=0.99975,
        seed=seed,
    )
    rewards_hist = []
    for _ in range(episodes):
        obs = env.reset(randomize=True)
        state = disc.encode(obs)
        action = agent.select_action(state, greedy=False)
        ep_rew = 0.0
        done = False
        while not done:
            next_obs, reward, done, _ = env.step(action)
            next_state = disc.encode(next_obs)
            next_action = agent.select_action(next_state, greedy=False)
            agent.update_sarsa(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
            ep_rew += reward
        agent.decay_epsilon()
        rewards_hist.append(ep_rew)
    return agent, np.array(rewards_hist)


def run_episode(env: PHNeutralizationEnv, controller, disc=None, randomize: bool = False, seed: int | None = None):
    obs = env.reset(randomize=randomize, seed=seed)
    if isinstance(controller, IndustrialPIDController):
        controller.reset()

    ph_trace, u_acid_trace, u_base_trace, q_in_trace = [], [], [], []
    total_reward, total_chem, in_band_steps = 0.0, 0.0, 0
    done = False

    while not done:
        if isinstance(controller, IndustrialPIDController):
            action = controller.select_action(obs)
        else:
            state = disc.encode(obs)
            action = controller.select_action(state, greedy=True)

        obs, reward, done, info = env.step(action)
        ph_trace.append(info["true_ph"])
        u_acid_trace.append(info["u_acid"])
        u_base_trace.append(info["u_base"])
        q_in_trace.append(obs["q_in"])
        total_reward += reward
        total_chem += info["chemical_used"]
        if info["in_band"]:
            in_band_steps += 1

    ph_arr = np.array(ph_trace)
    t_arr = np.arange(1, len(ph_arr) + 1) * env.dt
    errors = np.abs(ph_arr - env.target_ph)
    iae = float(np.sum(errors) * env.dt)
    ise = float(np.sum(errors ** 2) * env.dt)
    itae = float(np.sum(t_arr * errors) * env.dt)
    compliance = 100.0 * in_band_steps / env.max_steps
    max_overshoot = float(np.max(np.abs(ph_arr[15:] - env.target_ph)))
    net_u = np.array(u_base_trace) - np.array(u_acid_trace)
    tv = float(np.sum(np.abs(np.diff(net_u))))  # Total Variation (vana aşınma metriği)

    return {
        "ph": ph_arr,
        "u_acid": np.array(u_acid_trace),
        "u_base": np.array(u_base_trace),
        "q_in": np.array(q_in_trace),
        "total_reward": total_reward,
        "iae": iae,
        "ise": ise,
        "itae": itae,
        "compliance": compliance,
        "max_overshoot": max_overshoot,
        "total_chem": total_chem,
        "tv": tv,
    }


def evaluate_monte_carlo(env: PHNeutralizationEnv, controller, disc=None, n_runs: int = 50):
    metrics = [run_episode(env, controller, disc, randomize=True, seed=1000 + i) for i in range(n_runs)]
    summary = {}
    for key in ["total_reward", "iae", "ise", "compliance", "max_overshoot", "total_chem", "tv"]:
        vals = np.array([m[key] for m in metrics])
        summary[key] = (float(np.mean(vals)), float(np.std(vals)))
    return summary


def moving_average(x: np.ndarray, window: int = 400) -> np.ndarray:
    return np.convolve(x, np.ones(window) / window, mode="valid")


def generate_all_figures(
    env: PHNeutralizationEnv,
    disc: NonUniformDiscretizer,
    q_agent: TabularQLearningAgent,
    curves: dict,
    det_results: dict,
    mc_results: dict,
    delay_results: dict,
):
    os.makedirs("results", exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 200})

    # FIG 1: Teorik Titrasyon Eğrisi ve Eşitsiz Ayrıklaştırma Gerekçesi
    x_vals = np.linspace(-1.0e-3, 1.0e-3, 2000)
    ph_vals = np.array([PHNeutralizationEnv.x_to_ph(x) for x in x_vals])
    h_plus = 10.0 ** (-ph_vals)
    gain_vals = 1.0 / (np.log(10.0) * (h_plus + PHNeutralizationEnv.KW / h_plus))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax1.plot(x_vals * 1e3, ph_vals, color="#1f77b4", lw=2.2, label="Analitik Titrasyon Eğrisi")
    ax1.axhspan(6.8, 7.2, color="#2ca02c", alpha=0.18, label="Hedef Deşarj Bandı (6.8–7.2)")
    for b in disc.ph_bins:
        ax1.axhline(b, color="gray", linestyle=":", lw=0.65, alpha=0.7)
    ax1.set_title("Asit-Baz Titrasyon Eğrisi ve Eşitsiz Durum Bölgeleri")
    ax1.set_xlabel("Net İyon Derişimi Farkı x = [B+] - [A-] (mmol/L)")
    ax1.set_ylabel("Çözelti pH Değeri")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right", fontsize=8.5)

    ax2.semilogy(ph_vals, gain_vals, color="#d62728", lw=2.2)
    ax2.axvline(7.0, color="#2ca02c", linestyle="--", lw=1.2, label="Eşdeğerlik Noktası (pH 7.0)")
    ax2.set_title("Anlık Statik Proses Kazancı |d(pH)/dx| (Log Ölçek)")
    ax2.set_xlabel("Çözelti pH Değeri")
    ax2.set_ylabel("Statik Kazanç Kp [pH / (mol/L)]")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig("results/fig1_titration_nonlinearity.png")
    plt.close(fig)

    # FIG 2: Öğrenme ve Bileşen Analizi (Ablation) Yakınsama Eğrileri
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(moving_average(curves["q_full"]), label="Önerilen Tabular Q-Learning (Eşitsiz + Boru Hafızası)", color="#1f77b4", lw=2.2)
    ax.plot(moving_average(curves["sarsa_full"]), label="Tabular SARSA (Eşitsiz + Boru Hafızası)", color="#ff7f0e", lw=1.9)
    ax.plot(moving_average(curves["q_nopipe"]), label="Ablation: Ölü Zaman Boru Hafızası Yok", color="#9467bd", linestyle="--", lw=1.6)
    ax.plot(moving_average(curves["q_uniform"]), label="Ablation: Eşit Aralıklı (Uniform) pH Bölgeleri", color="#8c564b", linestyle=":", lw=1.6)
    ax.set_title("Klasik Pekiştirmeli Öğrenme Yakınsama ve Bileşen Analizi (400-Bölüm Hareketli Ortalama)")
    ax.set_xlabel("Eğitim Bölümü (Episode)")
    ax.set_ylabel("Toplam Bölüm Ödülü")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig("results/fig2_learning_curves.png")
    plt.close(fig)

    # FIG 3: Deterministik Şok Senaryosu Zaman Domeni Karşılaştırması
    dt = env.dt
    res_q = det_results["Tabular Q-Learning"]
    res_sarsa = det_results["Tabular SARSA"]
    res_pid = det_results["Klasik PID"]
    t = np.arange(len(res_q["ph"])) * dt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9.5, 7.6), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.1, 0.9]})
    ax1.axhspan(6.8, 7.2, color="#2ca02c", alpha=0.16, label="Yasal Deşarj Limiti (6.8 - 7.2)")
    ax1.axhline(7.0, color="#2ca02c", linestyle="--", lw=1.0)
    ax1.plot(t, res_pid["ph"], label="Klasik Endüstriyel PID", color="#d62728", linestyle="-.", lw=1.6)
    ax1.plot(t, res_sarsa["ph"], label="Tabular SARSA", color="#ff7f0e", lw=1.7)
    ax1.plot(t, res_q["ph"], label="Önerilen Tabular Q-Learning", color="#1f77b4", lw=2.2)
    ax1.axvline(40 * dt, color="gray", linestyle=":", lw=1.2)
    ax1.axvline(80 * dt, color="gray", linestyle=":", lw=1.2)
    ax1.text(40 * dt + 5, 10.6, "Bazik Tekstil Şoku", fontsize=8.5, color="#333333")
    ax1.text(80 * dt + 5, 10.6, "Yüksek Asit Şoku", fontsize=8.5, color="#333333")
    ax1.set_ylabel("Çıkış pH")
    ax1.set_ylim(3.0, 11.5)
    ax1.set_title("Çoklu Bozucu Etki Şokları Altında Kapalı Çevrim pH Kontrol Performansı")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)

    ax2.step(t, res_q["u_base"] - res_q["u_acid"], where="post", color="#1f77b4", lw=1.7, label="Q-Learning Net Pompa (Baz - Asit)")
    ax2.step(t, res_pid["u_base"] - res_pid["u_acid"], where="post", color="#d62728", alpha=0.65, lw=1.2, label="PID Net Pompa")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Dozaj (L/s)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8.5)

    ax3.step(t, res_q["q_in"], where="post", color="#2b5c8f", lw=1.6, label="Giriş Atık Su Debisi q_in (L/s)")
    ax3.set_xlabel("Simülasyon Zamanı (saniye)")
    ax3.set_ylabel("Giriş Debisi")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig("results/fig3_shock_response.png")
    plt.close(fig)

    # FIG 4: 50-Bölümlük Monte Carlo Hata Çubuklu Metrik Karşılaştırması
    names = ["Klasik PID", "Uniform Q", "Hafızasız Q", "SARSA", "Q-Learning"]
    colors = ["#d62728", "#8c564b", "#9467bd", "#ff7f0e", "#1f77b4"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0))

    for idx, (metric, title, ylabel) in enumerate([
        ("iae", "İntegral Mutlak Hata (IAE ↓)", "pH · s"),
        ("compliance", "Deşarj Limiti Uyumu (↑)", "Süre Oranı (%)"),
        ("total_chem", "Toplam Reaktif Tüketimi (↓)", "Litre (L)"),
    ]):
        means = [mc_results[n][metric][0] for n in names]
        stds = [mc_results[n][metric][1] for n in names]
        axes[idx].bar(names, means, yerr=stds, capsize=4, color=colors, alpha=0.9)
        axes[idx].set_title(title)
        axes[idx].set_ylabel(ylabel)
        axes[idx].tick_params(axis="x", rotation=22)
        axes[idx].grid(True, axis="y", alpha=0.3)
    axes[1].set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig("results/fig4_ablation_and_metrics.png")
    plt.close(fig)

    # FIG 5: Ölü Zaman Gecikmesi (k = 1..4) Dayanıklılık Analizi
    ks = sorted(delay_results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for label, col, marker in [
        ("Klasik PID", "#d62728", "s"),
        ("Tabular SARSA", "#ff7f0e", "^"),
        ("Tabular Q-Learning", "#1f77b4", "o"),
    ]:
        ax1.plot(ks, [delay_results[k][label]["iae"] for k in ks], marker=marker, color=col, lw=2, label=label)
        ax2.plot(ks, [delay_results[k][label]["compliance"] for k in ks], marker=marker, color=col, lw=2, label=label)

    ax1.set_title("Ölü Zaman Artışına Göre Hata (IAE) Değişimi")
    ax1.set_xlabel("Ölü Zaman Adımı k (τ_d = 5k saniye)")
    ax1.set_ylabel("IAE (pH · s)")
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    ax2.set_title("Ölü Zaman Artışına Göre Deşarj Uyumu (%)")
    ax2.set_xlabel("Ölü Zaman Adımı k (τ_d = 5k saniye)")
    ax2.set_ylabel("Limit İçi Süre (%)")
    ax2.set_xticks(ks)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/fig5_deadtime_robustness.png")
    plt.close(fig)

    # FIG 6: Öğrenilmiş Optimal Politika ve Değer Fonksiyonu Isı Haritaları
    n_ph = len(disc.ph_bins) + 1
    n_dph = len(disc.dph_bins) + 1
    policy_map = np.zeros((n_ph, n_dph))
    value_map = np.zeros((n_ph, n_dph))
    net_action_vals = np.array([ub - ua for ua, ub in env.actions])

    for i_ph in range(n_ph):
        for i_dph in range(n_dph):
            s_idx = np.ravel_multi_index((i_ph, i_dph, 1, 0, 2), disc.dims)
            best_a = int(np.argmax(q_agent.q_table[s_idx]))
            policy_map[i_ph, i_dph] = net_action_vals[best_a]
            value_map[i_ph, i_dph] = float(np.max(q_agent.q_table[s_idx]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    im1 = ax1.imshow(policy_map, cmap="coolwarm_r", aspect="auto", vmin=-0.35, vmax=0.35)
    ax1.set_title("Öğrenilmiş Optimal Politika π*(s) [L/s]")
    ax1.set_xlabel("ΔpH Bölgesi (0: Hızlı Düşüş → 6: Hızlı Artış)")
    ax1.set_ylabel("pH Bölgesi (0: Çok Asidik → 14: Çok Bazik)")
    fig.colorbar(im1, ax=ax1, label="Net Dozaj (- Asit / + Baz)")

    im2 = ax2.imshow(value_map, cmap="viridis", aspect="auto")
    ax2.set_title("Optimal Durum-Değer Fonksiyonu V*(s) = max_a Q(s,a)")
    ax2.set_xlabel("ΔpH Bölgesi (0: Hızlı Düşüş → 6: Hızlı Artış)")
    ax2.set_ylabel("pH Bölgesi (0: Çok Asidik → 14: Çok Bazik)")
    fig.colorbar(im2, ax=ax2, label="Beklenen İndirgenmiş Ödül")

    fig.tight_layout()
    fig.savefig("results/fig6_q_policy_and_value.png")
    plt.close(fig)


def main():
    env = PHNeutralizationEnv(dead_time_steps=3)
    disc_full = NonUniformDiscretizer()
    disc_uni = UniformDiscretizer()
    disc_nopipe = NoPipeMemoryDiscretizer()

    print("[1/5] Önerilen Tabular Q-Learning eğitiliyor (25.000 bölüm)...")
    q_full, rew_q_full = train_q(env, disc_full, episodes=25000, seed=42)

    print("[2/5] Tabular SARSA eğitiliyor (25.000 bölüm)...")
    sarsa_full, rew_sarsa_full = train_sarsa(env, disc_full, episodes=25000, seed=99)

    print("[3/5] Bileşen analizi (Ablation) modelleri eğitiliyor...")
    q_uni, rew_q_uni = train_q(env, disc_uni, episodes=25000, seed=7)
    q_nopipe, rew_q_nopipe = train_q(env, disc_nopipe, episodes=25000, seed=13)

    pid = IndustrialPIDController(env.actions, dt=env.dt)

    print("[4/5] Deterministik ve 50-Bölümlük Monte Carlo testleri çalıştırılıyor...")
    det_results = {
        "Klasik PID": run_episode(env, pid, randomize=False),
        "Uniform Q": run_episode(env, q_uni, disc_uni, randomize=False),
        "Hafızasız Q": run_episode(env, q_nopipe, disc_nopipe, randomize=False),
        "Tabular SARSA": run_episode(env, sarsa_full, disc_full, randomize=False),
        "Tabular Q-Learning": run_episode(env, q_full, disc_full, randomize=False),
    }

    mc_results = {
        "Klasik PID": evaluate_monte_carlo(env, pid, n_runs=50),
        "Uniform Q": evaluate_monte_carlo(env, q_uni, disc_uni, n_runs=50),
        "Hafızasız Q": evaluate_monte_carlo(env, q_nopipe, disc_nopipe, n_runs=50),
        "Tabular SARSA": evaluate_monte_carlo(env, sarsa_full, disc_full, n_runs=50),
        "Q-Learning": evaluate_monte_carlo(env, q_full, disc_full, n_runs=50),
    }

    print("[5/5] Ölü zaman hassasiyet analizi (k = 1, 2, 3, 4) yapılıyor...")
    delay_results = {}
    for k_val in [1, 2, 3, 4]:
        env_k = PHNeutralizationEnv(dead_time_steps=k_val)
        delay_results[k_val] = {
            "Klasik PID": run_episode(env_k, IndustrialPIDController(env_k.actions, dt=env_k.dt), randomize=False),
            "Tabular SARSA": run_episode(env_k, sarsa_full, disc_full, randomize=False),
            "Tabular Q-Learning": run_episode(env_k, q_full, disc_full, randomize=False),
        }

    print("\n=== TABLO 1: DETERMINISTIK COKLU SOK SENARYOSU ===")
    for name, r in det_results.items():
        print(
            f"{name:<20} | Ödül: {r['total_reward']:7.2f} | IAE: {r['iae']:7.2f} | "
            f"Uyum: {r['compliance']:5.1f}% | Aşım: {r['max_overshoot']:5.3f} | "
            f"Kimyasal: {r['total_chem']:6.2f} L | TV: {r['tv']:5.3f}"
        )

    print("\n=== TABLO 2: 50-BOLUMLUK MONTE CARLO STOKASTIK TEST (ORTALAMA ± STD) ===")
    for name, s in mc_results.items():
        print(
            f"{name:<20} | Ödül: {s['total_reward'][0]:6.1f}±{s['total_reward'][1]:4.1f} | "
            f"IAE: {s['iae'][0]:6.1f}±{s['iae'][1]:4.1f} | "
            f"Uyum: {s['compliance'][0]:5.1f}±{s['compliance'][1]:4.1f}% | "
            f"Kimyasal: {s['total_chem'][0]:5.2f}±{s['total_chem'][1]:4.2f} L"
        )

    generate_all_figures(
        env,
        disc_full,
        q_full,
        {"q_full": rew_q_full, "sarsa_full": rew_sarsa_full, "q_uniform": rew_q_uni, "q_nopipe": rew_q_nopipe},
        det_results,
        mc_results,
        delay_results,
    )
    print("\n[TAMAMLANDI] 6 adet makale grafiği 'results/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
