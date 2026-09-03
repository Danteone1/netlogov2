"""
SITER-CDMX v5.4 — Mesa + Streamlit (estilo NetLogo)
===================================================
Gemelo Digital Sociofísico Territorial de la Ciudad de México.

- Motor: Mesa (Agent-Based Modeling)
- UI: Streamlit con controles estilo NetLogo (Setup / Go / Monitors)
- 5 modos de datos: real | dummy | coherent | pure | calib
- Jerarquía: Alcaldía → Sección (Manzana preparada)
- Broker insertable
- Behaviors intercambiables (camino a +35 modelos)

Ejecutar:
    pip install -r requirements.txt
    streamlit run app_siter_cdmx_v54.py
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st

# Mesa
from mesa import Agent, Model
from mesa.time import SimultaneousActivation, RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pydeck as pdk
    HAS_PYDECK = True
except ImportError:
    HAS_PYDECK = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# =============================================================================
# CONFIG Y CONSTANTES CDMX
# =============================================================================
CONFIG = {
    "model": {"name": "voter", "beta": 1.2, "steps": 20, "seed": 42},
    "simulation": {"n_agentes": 300},
    "network": {"p_intra": 0.08, "p_inter": 0.02},
}

ALCALDIAS_CDMX = [
    "ALVARO OBREGON", "AZCAPOTZALCO", "BENITO JUAREZ", "COYOACAN",
    "CUAJIMALPA DE MORELOS", "CUAUHTEMOC", "GUSTAVO A MADERO",
    "IZTACALCO", "IZTAPALAPA", "LA MAGDALENA CONTRERAS",
    "MIGUEL HIDALGO", "MILPA ALTA", "TLAHUAC", "TLALPAN",
    "VENUSTIANO CARRANZA", "XOCHIMILCO"
]

ALCALDIA_COORDS = {
    "CUAUHTEMOC": (19.4326, -99.1332), "BENITO JUAREZ": (19.3984, -99.1576),
    "MIGUEL HIDALGO": (19.4285, -99.2000), "COYOACAN": (19.3467, -99.1617),
    "IZTAPALAPA": (19.3550, -99.0620), "GUSTAVO A MADERO": (19.4900, -99.1100),
    "ALVARO OBREGON": (19.3580, -99.2270), "TLALPAN": (19.2880, -99.1670),
    "XOCHIMILCO": (19.2630, -99.1040), "VENUSTIANO CARRANZA": (19.4200, -99.1000),
    "AZCAPOTZALCO": (19.4870, -99.1860), "IZTACALCO": (19.3950, -99.0980),
    "CUAJIMALPA DE MORELOS": (19.3570, -99.2900), "LA MAGDALENA CONTRERAS": (19.3200, -99.2400),
    "TLAHUAC": (19.2700, -99.0050), "MILPA ALTA": (19.1920, -99.0230),
}

COLOR_INTENCION = {"SIMPATIZANTE": "#2ecc71", "OPOSITOR": "#e74c3c", "INDECISO": "#95a5a6"}
COLOR_RGBA = {
    "SIMPATIZANTE": [46, 204, 113, 200],
    "OPOSITOR": [231, 76, 60, 200],
    "INDECISO": [149, 165, 166, 180],
    "BROKER": [255, 215, 0, 255],
}

def sha256(v) -> str:
    p = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    return hashlib.sha256(p.encode()).hexdigest()

# =============================================================================
# BEHAVIORS (aquí se irán sumando los +35 modelos)
# =============================================================================
class Behavior:
    """Contrato común para todos los modelos sociofísicos."""
    name = "base"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def step_agent(self, agent: "SeccionAgent"):
        raise NotImplementedError


class VoterBehavior(Behavior):
    """Voter clásico ponderado por influencia."""
    name = "voter"

    def step_agent(self, agent):
        neighbors = agent.model.grid.get_neighbors(agent.pos, include_center=False)
        if not neighbors:
            return
        other = agent.random.choice(neighbors)
        beta = self.params.get("beta", 1.2)
        prob = min(0.9, other.influencia * beta)
        if agent.es_broker:
            prob *= 0.3  # resistencia
        if agent.spin != other.spin and agent.random.random() < prob:
            agent.spin = other.spin
            agent.opinion = 0.7 if agent.spin == 1 else (-0.7 if agent.spin == -1 else 0.0)
            agent.intencion = {1: "SIMPATIZANTE", -1: "OPOSITOR", 0: "INDECISO"}[agent.spin]


class DeffuantBehavior(Behavior):
    """Bounded confidence + repulsión."""
    name = "deffuant"

    def step_agent(self, agent):
        neighbors = agent.model.grid.get_neighbors(agent.pos, include_center=False)
        if not neighbors:
            return
        other = agent.random.choice(neighbors)
        mu = self.params.get("mu", 0.3)
        eps = self.params.get("epsilon", 0.40)
        eps_rep = self.params.get("epsilon_rep", 0.80)
        d = abs(agent.opinion - other.opinion)
        if d < eps:
            delta = mu * (other.opinion - agent.opinion)
            agent.opinion = float(np.clip(agent.opinion + delta, -1, 1))
            other.opinion = float(np.clip(other.opinion - delta, -1, 1))
        elif d > eps_rep:
            delta = 0.2 * mu * (other.opinion - agent.opinion)
            agent.opinion = float(np.clip(agent.opinion - delta, -1, 1))
            other.opinion = float(np.clip(other.opinion + delta, -1, 1))
        # actualizar spin
        agent.spin = 1 if agent.opinion > 0.25 else (-1 if agent.opinion < -0.25 else 0)
        agent.intencion = {1: "SIMPATIZANTE", -1: "OPOSITOR", 0: "INDECISO"}[agent.spin]


class ABMSAFBehavior(Behavior):
    """Comportamiento compuesto inspirado en el ABM-SAF original."""
    name = "abm_saf"

    def step_agent(self, agent):
        neighbors = agent.model.grid.get_neighbors(agent.pos, include_center=False)
        if not neighbors:
            return
        if agent.fatiga > 0.8 and agent.random.random() < 0.7:
            return
        for other in neighbors:
            prob = agent.influencia * 0.55 * self.params.get("beta", 1.2)
            if agent.es_broker:
                prob *= 1.45
            if agent.random.random() < min(prob, 0.75):
                # Deffuant parcial
                d = abs(agent.opinion - other.opinion)
                if d < 0.40:
                    other.opinion = float(np.clip(other.opinion + 0.3 * (agent.opinion - other.opinion), -1, 1))
                # Voter
                if abs(agent.spin - other.spin) >= 1 and agent.spin != 0:
                    other.spin = agent.spin
                    other.opinion = 0.7 if other.spin == 1 else -0.7
                    other.intencion = {1: "SIMPATIZANTE", -1: "OPOSITOR", 0: "INDECISO"}[other.spin]
                agent.fatiga = min(1.0, agent.fatiga + 0.05)
        agent.fatiga *= 0.95


BEHAVIOR_REGISTRY = {
    "voter": VoterBehavior,
    "deffuant": DeffuantBehavior,
    "abm_saf": ABMSAFBehavior,
}

# =============================================================================
# AGENTES MESA
# =============================================================================
class SeccionAgent(Agent):
    """Agente territorial = Sección Electoral (extensible a Manzana)."""

    def __init__(self, unique_id, model, alcaldia="CUAUHTEMOC", seccion="0001",
                 lat=19.43, lon=-99.13, opinion=0.0, capital_social=0.5,
                 influencia=0.5, es_broker=False, es_adversario=False, **kwargs):
        super().__init__(unique_id, model)
        self.alcaldia = alcaldia
        self.seccion = str(seccion)
        self.lat = lat
        self.lon = lon
        self.opinion = float(opinion)
        self.spin = 1 if self.opinion > 0.25 else (-1 if self.opinion < -0.25 else 0)
        self.intencion = {1: "SIMPATIZANTE", -1: "OPOSITOR", 0: "INDECISO"}[self.spin]
        self.capital_social = capital_social
        self.influencia = influencia
        self.fatiga = 0.0
        self.es_broker = es_broker
        self.es_adversario = es_adversario
        self.level = "seccion"  # preparado para manzana / utm

    def step(self):
        self.model.behavior.step_agent(self)


# =============================================================================
# MODELO MESA PRINCIPAL
# =============================================================================
class SITERModel(Model):
    """Mundo CDMX estilo NetLogo implementado en Mesa."""

    def __init__(self, df_state: pd.DataFrame, adj: dict,
                 behavior: str = "voter", seed: int = 42, **params):
        super().__init__()
        self.seed = seed
        self.random.seed(seed)
        np.random.seed(seed)

        # Grafo de red
        G = nx.Graph()
        for i in range(len(df_state)):
            G.add_node(i)
        for k, vs in adj.items():
            for v in vs:
                if int(k) < len(df_state) and int(v) < len(df_state):
                    G.add_edge(int(k), int(v))
        self.G = G
        self.grid = NetworkGrid(G)
        self.schedule = SimultaneousActivation(self)

        # Behavior intercambiable
        behavior_cls = BEHAVIOR_REGISTRY.get(behavior, VoterBehavior)
        self.behavior = behavior_cls(params)
        self.behavior_name = behavior

        # Crear agentes
        for i, row in df_state.reset_index(drop=True).iterrows():
            agent = SeccionAgent(
                unique_id=i,
                model=self,
                alcaldia=row.get("alcaldia", "CUAUHTEMOC"),
                seccion=row.get("seccion", row.get("territorial_unit_id", f"{i:04d}")),
                lat=float(row.get("lat", 19.43)),
                lon=float(row.get("lon", -99.13)),
                opinion=float(row.get("opinion_continua", 0.0)),
                capital_social=float(row.get("capital_social", 0.5)),
                influencia=float(row.get("influencia_SAF", row.get("habilidades_sociales", 0.5))),
                es_broker=bool(row.get("es_broker_insertado", False)),
                es_adversario=bool(row.get("es_adversario", False)),
            )
            self.schedule.add(agent)
            self.grid.place_agent(agent, i)

        self.datacollector = DataCollector(
            model_reporters={
                "SIMPATIZANTE": lambda m: m.count_spin(1),
                "OPOSITOR": lambda m: m.count_spin(-1),
                "INDECISO": lambda m: m.count_spin(0),
                "n_brokers": lambda m: sum(1 for a in m.schedule.agents if a.es_broker),
            },
            agent_reporters={"opinion": "opinion", "spin": "spin", "alcaldia": "alcaldia"}
        )
        self.datacollector.collect(self)  # estado inicial
        self.running = True

    def count_spin(self, value):
        agents = self.schedule.agents
        if not agents:
            return 0.0
        return sum(1 for a in agents if a.spin == value) / len(agents)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

    def get_agents_df(self) -> pd.DataFrame:
        rows = []
        for a in self.schedule.agents:
            rows.append({
                "agent_id": f"{'BROKER' if a.es_broker else 'SEC'}-{a.unique_id}",
                "alcaldia": a.alcaldia,
                "seccion": a.seccion,
                "lat": a.lat,
                "lon": a.lon,
                "opinion": a.opinion,
                "spin": a.spin,
                "intencion": a.intencion,
                "influencia": a.influencia,
                "es_broker": a.es_broker,
                "fatiga": a.fatiga,
            })
        return pd.DataFrame(rows)

    def insert_broker(self, lat: float, lon: float, alcaldia: str,
                      capital: float = 0.9, intencion: str = "SIMPATIZANTE",
                      grado: int = 15):
        """Inserta un broker en la ubicación dada y lo conecta a vecinos influyentes."""
        new_id = max(a.unique_id for a in self.schedule.agents) + 1
        opinion = 0.75 if intencion == "SIMPATIZANTE" else (-0.75 if intencion == "OPOSITOR" else 0.0)
        broker = SeccionAgent(
            unique_id=new_id, model=self, alcaldia=alcaldia, seccion=f"BRK{new_id}",
            lat=lat, lon=lon, opinion=opinion, capital_social=capital,
            influencia=0.95, es_broker=True
        )
        self.schedule.add(broker)
        # Conectar a los más influyentes de la misma alcaldía o cercanos
        candidatos = sorted(
            [a for a in self.schedule.agents if a.unique_id != new_id],
            key=lambda x: x.influencia, reverse=True
        )[:grado * 2]
        self.G.add_node(new_id)
        for c in candidatos[:grado]:
            self.G.add_edge(new_id, c.unique_id)
        self.grid.place_agent(broker, new_id)
        return broker


# =============================================================================
# DATA PROVIDER (5 modos) — simplificado y compatible
# =============================================================================
class DataProvider:
    def __init__(self, mode="synth_pure", seed=42):
        self.mode = mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def load(self, n=300, **kwargs):
        if self.mode == "dummy":
            return self._dummy(n)
        if self.mode == "synth_coherent":
            return self._coherent(n)
        if self.mode == "synth_calib":
            return self._calib(kwargs.get("scenario", "polarizacion_alta"), n)
        # default: synth_pure
        return self._pure(n)

    def _pure(self, n):
        rows = []
        for i in range(n):
            alc = str(self.rng.choice(ALCALDIAS_CDMX))
            lat, lon = ALCALDIA_COORDS[alc]
            opinion = float(self.rng.uniform(-0.8, 0.8))
            rows.append({
                "seccion": f"{i:04d}", "alcaldia": alc,
                "lat": lat + self.rng.normal(0, 0.012),
                "lon": lon + self.rng.normal(0, 0.012),
                "opinion_continua": opinion,
                "capital_social": float(self.rng.uniform(0.3, 0.8)),
                "influencia_SAF": float(self.rng.uniform(0.2, 0.9)),
                "es_broker_insertado": False, "es_adversario": False,
            })
        df = pd.DataFrame(rows)
        adj = self._build_adj(df)
        return df, adj

    def _dummy(self, n):
        return self._pure(min(n, 60))

    def _coherent(self, n):
        rows = []
        for i in range(n):
            alc = str(self.rng.choice(ALCALDIAS_CDMX))
            lat, lon = ALCALDIA_COORDS[alc]
            share = float(np.clip(self.rng.normal(0.37, 0.14), 0.05, 0.95))
            opinion = 2 * share - 1
            rows.append({
                "seccion": f"C{i:04d}", "alcaldia": alc,
                "lat": lat + self.rng.normal(0, 0.01),
                "lon": lon + self.rng.normal(0, 0.01),
                "opinion_continua": opinion,
                "capital_social": float(np.clip(1 - self.rng.normal(0.4, 0.15), 0.2, 0.9)),
                "influencia_SAF": float(self.rng.uniform(0.3, 0.85)),
                "es_broker_insertado": False, "es_adversario": False,
            })
        df = pd.DataFrame(rows)
        adj = self._build_adj(df, p_intra=0.07, p_inter=0.02)
        return df, adj

    def _calib(self, scenario, n):
        # polarizacion_alta simplificada
        half = n // 2
        rows = []
        for i in range(n):
            bloque = 0 if i < half else 1
            alc = ALCALDIAS_CDMX[bloque * 4 + (i % 4)]
            lat, lon = ALCALDIA_COORDS[alc]
            opinion = 0.75 if bloque == 0 else -0.75
            opinion = float(np.clip(opinion + self.rng.normal(0, 0.08), -1, 1))
            rows.append({
                "seccion": f"P{i:04d}", "alcaldia": alc,
                "lat": lat + self.rng.normal(0, 0.008),
                "lon": lon + self.rng.normal(0, 0.008),
                "opinion_continua": opinion,
                "capital_social": 0.6, "influencia_SAF": 0.5,
                "es_broker_insertado": False, "es_adversario": False,
            })
        df = pd.DataFrame(rows)
        adj = self._build_adj(df, p_intra=0.12, p_inter=0.005)
        return df, adj

    def _build_adj(self, df, p_intra=0.08, p_inter=0.02):
        adj = defaultdict(list)
        for alc in df["alcaldia"].unique():
            idxs = df.index[df["alcaldia"] == alc].tolist()
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    if self.rng.random() < p_intra:
                        x, y = idxs[a], idxs[b]
                        adj[x].append(y)
                        adj[y].append(x)
        # inter simple
        alcs = list(df["alcaldia"].unique())
        for i, a1 in enumerate(alcs):
            for a2 in alcs[i+1:]:
                idxs1 = df.index[df["alcaldia"] == a1].tolist()
                idxs2 = df.index[df["alcaldia"] == a2].tolist()
                for x in idxs1[:5]:
                    for y in idxs2[:5]:
                        if self.rng.random() < p_inter:
                            adj[x].append(y)
                            adj[y].append(x)
        return dict(adj)


# =============================================================================
# STREAMLIT UI — ESTILO NETLOGO
# =============================================================================
st.set_page_config(page_title="SITER-CDMX v5.4 Mesa", page_icon="🧠", layout="wide")
st.title("🧠 SITER-CDMX v5.4 — Mesa + NetLogo-style")
st.caption("Motor Mesa · Behaviors intercambiables · 5 modos de datos · Controles Setup/Go · CDMX")

if "s54" not in st.session_state:
    st.session_state.s54 = {
        "model": None, "df": pd.DataFrame(), "adj": {},
        "tray": [], "running": False, "tick": 0,
        "behavior": "voter", "meta": {}
    }
S = st.session_state.s54

# ----- SIDEBAR -----
st.sidebar.header("⚙️ Setup (NetLogo-style)")

data_mode = st.sidebar.selectbox("Modo de datos", ["synth_pure", "dummy", "synth_coherent", "synth_calib"], index=0)
seed = st.sidebar.number_input("Seed", 1, 99999, 42)
n = st.sidebar.slider("N secciones", 50, 800, 300, 50)
behavior = st.sidebar.selectbox("Behavior (modelo)", list(BEHAVIOR_REGISTRY.keys()), index=0)
beta = st.sidebar.slider("Beta", 0.3, 2.5, 1.2, 0.1)
steps_per_go = st.sidebar.slider("Pasos por Go", 1, 20, 5)

if data_mode == "synth_calib":
    scenario = st.sidebar.selectbox("Escenario", ["polarizacion_alta"])
else:
    scenario = "polarizacion_alta"

col_setup, col_go, col_stop = st.sidebar.columns(3)

with col_setup:
    if st.button("🔄 Setup", use_container_width=True):
        provider = DataProvider(mode=data_mode, seed=seed)
        df, adj = provider.load(n=n, scenario=scenario)
        model = SITERModel(df, adj, behavior=behavior, seed=seed, beta=beta)
        S.update({
            "model": model, "df": df, "adj": adj,
            "tray": [{"step": 0, "SIMPATIZANTE": model.count_spin(1),
                      "OPOSITOR": model.count_spin(-1), "INDECISO": model.count_spin(0)}],
            "tick": 0, "running": False, "behavior": behavior
        })
        st.success(f"Setup OK · {len(df)} secciones · {behavior}")

with col_go:
    if st.button("▶️ Go", use_container_width=True):
        if S["model"] is None:
            st.warning("Haz Setup primero")
        else:
            for _ in range(steps_per_go):
                S["model"].step()
                S["tick"] += 1
                S["tray"].append({
                    "step": S["tick"],
                    "SIMPATIZANTE": S["model"].count_spin(1),
                    "OPOSITOR": S["model"].count_spin(-1),
                    "INDECISO": S["model"].count_spin(0),
                })
            st.rerun()

with col_stop:
    if st.button("⏹ Reset", use_container_width=True):
        S["model"] = None
        S["tray"] = []
        S["tick"] = 0
        st.rerun()

# Broker rápido
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Insertar Broker")
broker_alc = st.sidebar.selectbox("Alcaldía del broker", ALCALDIAS_CDMX)
broker_int = st.sidebar.selectbox("Intención", ["SIMPATIZANTE", "OPOSITOR", "INDECISO"])
if st.sidebar.button("➕ Insertar Broker"):
    if S["model"] is None:
        st.sidebar.warning("Haz Setup primero")
    else:
        lat, lon = ALCALDIA_COORDS[broker_alc]
        S["model"].insert_broker(lat, lon, broker_alc, capital=0.9, intencion=broker_int)
        st.sidebar.success("Broker insertado")
        st.rerun()

# ----- MAIN PANEL -----
if S["model"] is None:
    st.info("➡️ Pulsa **Setup** en la barra lateral para generar el universo (estilo NetLogo).")
else:
    model: SITERModel = S["model"]
    agents_df = model.get_agents_df()

    # Monitors (estilo NetLogo)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tick", S["tick"])
    c2.metric("Simpatizantes", f"{model.count_spin(1):.1%}")
    c3.metric("Opositores", f"{model.count_spin(-1):.1%}")
    c4.metric("Indecisos", f"{model.count_spin(0):.1%}")
    c5.metric("Brokers", sum(1 for a in model.schedule.agents if a.es_broker))

    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ World View", "📈 Plots", "📋 Agentes", "ℹ️ Modelo"])

    with tab1:
        st.subheader("World View (NetLogo-style)")
        if HAS_PLOTLY:
            fig = px.scatter(
                agents_df, x="lon", y="lat",
                color="intencion", size="influencia",
                hover_name="agent_id",
                color_discrete_map=COLOR_INTENCION,
                hover_data=["alcaldia", "seccion", "opinion", "spin"],
                title=f"Secciones CDMX · Behavior: {S['behavior']} · Tick {S['tick']}",
                size_max=28
            )
            # Brokers como estrellas
            brokers = agents_df[agents_df["es_broker"]]
            if not brokers.empty:
                fig.add_trace(go.Scatter(
                    x=brokers["lon"], y=brokers["lat"],
                    mode="markers", marker=dict(symbol="star", size=22, color="gold",
                                                line=dict(width=1, color="black")),
                    name="Broker", text=brokers["agent_id"]
                ))
            fig.update_layout(height=600, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(agents_df)

        # Resumen por alcaldía
        by_alc = agents_df.groupby("alcaldia")["intencion"].value_counts(normalize=True).unstack(fill_value=0)
        st.markdown("**Distribución por Alcaldía**")
        st.dataframe(by_alc.style.format("{:.1%}"), use_container_width=True)

    with tab2:
        if S["tray"]:
            tray_df = pd.DataFrame(S["tray"])
            st.line_chart(tray_df.set_index("step")[["SIMPATIZANTE", "OPOSITOR", "INDECISO"]])
            if HAS_PLOTLY:
                fig = go.Figure()
                for col, color in [("SIMPATIZANTE", "#2ecc71"), ("OPOSITOR", "#e74c3c"), ("INDECISO", "#95a5a6")]:
                    fig.add_trace(go.Scatter(x=tray_df["step"], y=tray_df[col], name=col, line=dict(color=color)))
                fig.update_layout(title="Trayectoria de opiniones", height=400, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.dataframe(agents_df, use_container_width=True)

    with tab4:
        st.json({
            "behavior": S["behavior"],
            "n_agents": len(model.schedule.agents),
            "n_edges": model.G.number_of_edges(),
            "tick": S["tick"],
            "seed": model.seed,
        })
        st.markdown("""
        **Behaviors disponibles (camino a +35 modelos):**
        - `voter` — Voter clásico ponderado
        - `deffuant` — Bounded confidence + repulsión
        - `abm_saf` — Compuesto (Voter + Deffuant + fatiga)

        Próximos: Threshold, q-Voter, Schelling, Complex Contagion, HK, DeGroot, FJ, Axelrod...
        """)

st.sidebar.markdown("---")
st.sidebar.caption("SITER-CDMX v5.4 · Mesa · Estilo NetLogo · Sin PII")
