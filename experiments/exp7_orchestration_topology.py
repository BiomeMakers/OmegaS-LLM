# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

"""
TEST 2 (Parte 2 del hilo): ¿Aporta algo aplicar la métrica de Omega-S
(Tr(A^3) sobre una matriz de "adyacencia" de tráfico) a la orquestación
de agentes/microservicios de un sistema de chatbots?
=============================================================================
Objetivo real de este script: comprobación rápida de "no" -- generar un
grafo sintético de tráfico entre N agentes, simular colas con un
balanceador estándar (least-connections) vs. un balanceador guiado por la
métrica topológica de Omega-S, y comparar latencia/saturación máxima.

Requiere: numpy, simpy (pip install simpy si no está instalado en Colab)
"""

import numpy as np
import simpy
import random

# ---------------------------------------------------------------------------
# 0. CONFIGURACIÓN
# ---------------------------------------------------------------------------
N_AGENTS = 8            # número de agentes/microservicios
N_REQUESTS = 2000        # peticiones a simular
SERVICE_TIME_MEAN = 0.05 # tiempo medio de proceso por petición (s)
ARRIVAL_RATE = 120        # peticiones por segundo (Poisson)
SEED = 42


# ---------------------------------------------------------------------------
# 1. GRAFO SINTÉTICO DE TRÁFICO (para el cálculo de Tr(A^3))
# ---------------------------------------------------------------------------
def generate_traffic_graph(n_agents, hub_bias=True, seed=SEED):
    """
    Genera una matriz de adyacencia de tráfico histórico entre agentes.
    hub_bias=True crea un par de "hubs" que concentran tráfico desproporcionado
    (el escenario que Omega-S intentaría corregir).
    """
    rng = np.random.default_rng(seed)
    A = rng.random((n_agents, n_agents)) * 0.2
    if hub_bias:
        # dos nodos reciben mucho más tráfico que el resto
        A[0, :] += 0.8
        A[:, 0] += 0.8
        A[1, :] += 0.5
        A[:, 1] += 0.5
    np.fill_diagonal(A, 0)
    A = (A + A.T) / 2  # simétrica, como una matriz de co-tráfico
    return A


def clustering_metric(A):
    """Tr(A^3) normalizado -- proxy de Omega-S para 'concentración estructural'."""
    A3 = A @ A @ A
    return np.trace(A3)


def omega_weighted_capacity(A):
    """
    A partir del grafo de tráfico, deriva una 'capacidad ajustada' por agente:
    penaliza a los nodos que concentran más conexión (los hubs), igual que
    Omega-S intenta redistribuir peso lejos de los monopolios.
    """
    degree = A.sum(axis=1)
    # cuanto más grado tiene un nodo, menos peticiones nuevas debería recibir
    inv_degree = 1.0 / (degree + 1e-6)
    weights = inv_degree / inv_degree.sum()
    return weights


# ---------------------------------------------------------------------------
# 2. SIMULACIÓN DE COLAS: least-connections vs. Omega-guided
# ---------------------------------------------------------------------------
class AgentPool:
    def __init__(self, env, n_agents, strategy, omega_weights=None):
        self.env = env
        self.servers = [simpy.Resource(env, capacity=1) for _ in range(n_agents)]
        self.strategy = strategy
        self.omega_weights = omega_weights
        self.load_counts = np.zeros(n_agents)
        self.wait_times = []

    def pick_agent(self):
        if self.strategy == "least_connections":
            queue_lengths = [len(s.queue) + s.count for s in self.servers]
            return int(np.argmin(queue_lengths))
        elif self.strategy == "omega_guided":
            # combina longitud de cola actual con el peso topológico precomputado
            queue_lengths = np.array([len(s.queue) + s.count for s in self.servers])
            score = queue_lengths - 3.0 * self.omega_weights  # favorece nodos de bajo "hub score"
            return int(np.argmin(score))
        else:
            raise ValueError("estrategia desconocida")

    def handle_request(self, req_id):
        agent_id = self.pick_agent()
        start_wait = self.env.now
        with self.servers[agent_id].request() as req:
            yield req
            wait = self.env.now - start_wait
            self.wait_times.append(wait)
            self.load_counts[agent_id] += 1
            service_time = random.expovariate(1.0 / SERVICE_TIME_MEAN)
            yield self.env.timeout(service_time)


def request_generator(env, pool, n_requests, arrival_rate):
    for i in range(n_requests):
        yield env.timeout(random.expovariate(arrival_rate))
        env.process(pool.handle_request(i))


def run_simulation(strategy, omega_weights=None):
    random.seed(SEED)
    env = simpy.Environment()
    pool = AgentPool(env, N_AGENTS, strategy, omega_weights)
    env.process(request_generator(env, pool, N_REQUESTS, ARRIVAL_RATE))
    env.run()
    return pool


# ---------------------------------------------------------------------------
# 3. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("GRAFO DE TRÁFICO SINTÉTICO")
    print("=" * 70)
    A = generate_traffic_graph(N_AGENTS, hub_bias=True)
    tr_a3 = clustering_metric(A)
    print(f"Tr(A^3) del grafo con sesgo de hubs: {tr_a3:.2f}")

    omega_weights = omega_weighted_capacity(A)
    print("Pesos derivados de Omega (menor peso = nodo más 'hub', se evita):")
    for i, w in enumerate(omega_weights):
        print(f"  Agente {i}: {w:.4f}")

    print("\n" + "=" * 70)
    print("SIMULACIÓN: least-connections (baseline estándar)")
    print("=" * 70)
    pool_baseline = run_simulation("least_connections")
    wait_baseline = np.array(pool_baseline.wait_times)
    print(f"Latencia media de espera: {wait_baseline.mean():.4f}s")
    print(f"Latencia p95: {np.percentile(wait_baseline, 95):.4f}s")
    print(f"Carga máxima en un solo agente: {int(pool_baseline.load_counts.max())} "
          f"de {N_REQUESTS} peticiones "
          f"({pool_baseline.load_counts.max() / N_REQUESTS:.2%})")

    print("\n" + "=" * 70)
    print("SIMULACIÓN: omega_guided (usando la métrica topológica)")
    print("=" * 70)
    pool_omega = run_simulation("omega_guided", omega_weights)
    wait_omega = np.array(pool_omega.wait_times)
    print(f"Latencia media de espera: {wait_omega.mean():.4f}s")
    print(f"Latencia p95: {np.percentile(wait_omega, 95):.4f}s")
    print(f"Carga máxima en un solo agente: {int(pool_omega.load_counts.max())} "
          f"de {N_REQUESTS} peticiones "
          f"({pool_omega.load_counts.max() / N_REQUESTS:.2%})")

    print("\n" + "=" * 70)
    print("VEREDICTO")
    print("=" * 70)
    delta_wait = (wait_omega.mean() - wait_baseline.mean()) / wait_baseline.mean()
    if delta_wait < -0.05:
        print(f"Omega-guided reduce la latencia media un {-delta_wait:.1%} frente "
              f"a least-connections. Señal (poco esperada) de que vale la pena mirar más.")
    elif delta_wait > 0.05:
        print(f"Omega-guided empeora la latencia media un {delta_wait:.1%} frente "
              f"a least-connections. Confirma la sospecha: un balanceador estándar "
              f"ya resuelve esto mejor, esta línea no aporta valor de producto.")
    else:
        print("Omega-guided y least-connections quedan prácticamente empatados. "
              "No hay evidencia de que la métrica topológica aporte algo que el "
              "balanceador estándar no dé ya. Recomendación: cerrar esta línea.")
