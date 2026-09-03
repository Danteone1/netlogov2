
import hashlib
import io
import json
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# SITER-CDMX + MESA
# Skeleton v5.4
# Streamlit UI + Mesa ABM + interchangeable Behaviors
#
# Scope:
# - CDMX only
# - aggregated/synthetic territorial units
# - no PII / no individual profiling
# - opinion/state variables are simulation constructs, not voter
#   predictions
# ============================================================

try:
    from mesa import Agent, Model
    from mesa.datacollection import DataCollector
    MESA_OK = True
except Exception as exc:
    MESA_OK = False
    MESA_IMPORT_ERROR = str(exc)

try:
    # Mesa 3.x compatibility path
    from mesa.space import NetworkGrid as LegacyNetworkGrid
except Exception:
    LegacyNetworkGrid = None

try:
    # Mesa 3/4 newer API
    from mesa.discrete_space import Network as MesaNetwork
except Exception:
    MesaNetwork = None


CDMX_ALCALDIAS = [
    "ÁLVARO OBREGÓN", "AZCAPOTZALCO", "BENITO JUÁREZ",
    "COYOACÁN", "CUAJIMALPA DE MORELOS", "CUAUHTÉMOC",
    "GUSTAVO A. MADERO", "IZTACALCO", "IZTAPALAPA",
    "LA MAGDALENA CONTRERAS", "MIGUEL HIDALGO", "MILPA ALTA",
    "TLÁHUAC", "TLALPAN", "VENUSTIANO CARRANZA", "XOCHIMILCO"
]

DEFAULTS = {
    "n_agents": 120,
    "p_intra": 0.06,
    "p_inter": 0.015,
    "initial_opinion": 0.0,
    "seed": 42,
    "steps": 50,
    "confidence": 0.25,
    "noise": 0.02,
    "threshold": 0.50,
    "shock": 0.0,
}


def sha256_obj(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def gini(values):
    x = np.asarray(values, dtype=float)
    if len(x) == 0 or np.allclose(x, 0):
        return 0.0
    x = np.sort(np.abs(x))
    n = len(x)
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * np.sum(x))) - (n + 1) / n)


# ============================================================
# DataProvider — 5 modos
# ============================================================

class DataProvider:
    MODES = [
        "Sintético reproducible",
        "CSV agregado",
        "DataFrame externo",
        "Demo territorial CDMX",
        "Carga manual"
    ]

    def __init__(self, seed=42):
        self.seed = int(seed)

    def synthetic(self, n=120):
        rng = np.random.default_rng(self.seed)
        rows = []
        for i in range(n):
            alcaldia = CDMX_ALCALDIAS[i % len(CDMX_ALCALDIAS)]
            rows.append({
                "territorial_unit_id": f"CDMX-SECCION-{i+1:05d}",
                "alcaldia": alcaldia,
                "seccion": f"{1000+i}",
                "manzana": "",
                "opinion_continua": float(rng.uniform(-1, 1)),
                "capital_social": float(rng.beta(4, 4)),
                "acceso_informacion": float(rng.beta(4, 4)),
                "influencia_liderazgo": float(rng.beta(4, 4)),
                "arraigo": float(rng.beta(4, 4)),
                "nivel_movilizacion": float(rng.beta(4, 4)),
                "desconfianza": float(rng.beta(3, 5)),
                "exposicion_problema": float(rng.beta(4, 4)),
            })
        return pd.DataFrame(rows)

    def from_csv(self, uploaded_file):
        df = pd.read_csv(uploaded_file)
        return self.normalize(df)

    def from_dataframe(self, df):
        return self.normalize(df.copy())

    def normalize(self, df):
        df = df.copy()
        if "territorial_unit_id" not in df:
            df["territorial_unit_id"] = [f"TU-{i+1:05d}" for i in range(len(df))]
        if "alcaldia" not in df:
            df["alcaldia"] = "NO_ESPECIFICADA"
        if "seccion" not in df:
            df["seccion"] = [str(i+1) for i in range(len(df))]
        defaults = {
            "manzana": "",
            "opinion_continua": 0.0,
            "capital_social": 0.5,
            "acceso_informacion": 0.5,
            "influencia_liderazgo": 0.5,
            "arraigo": 0.5,
            "nivel_movilizacion": 0.5,
            "desconfianza": 0.5,
            "exposicion_problema": 0.5,
        }
        for c, v in defaults.items():
            if c not in df:
                df[c] = v
        return df


# ============================================================
# Behaviors
# ============================================================

class Behavior:
    name = "Base"

    def __init__(self, params=None):
        self.params = params or {}

    def step_agent(self, agent):
        raise NotImplementedError

    def step_model(self, model):
        pass


class VoterBehavior(Behavior):
    name = "Voter / difusión local"

    def step_agent(self, agent):
        neigh = agent.neighbors()
        if not neigh:
            return
        other = agent.model.random.choice(neigh)
        influence = float(self.params.get("influence", 0.25))
        agent.next_opinion = clamp(
            agent.opinion + influence * (other.opinion - agent.opinion)
        )


class DeffuantBehavior(Behavior):
    name = "Deffuant-Weisbuch"

    def step_agent(self, agent):
        neigh = agent.neighbors()
        if not neigh:
            agent.next_opinion = agent.opinion
            return
        other = agent.model.random.choice(neigh)
        eps = float(self.params.get("confidence", 0.25))
        mu = float(self.params.get("mu", 0.50))
        if abs(agent.opinion - other.opinion) <= eps:
            agent.next_opinion = clamp(
                agent.opinion + mu * (other.opinion - agent.opinion)
            )
        else:
            agent.next_opinion = agent.opinion


class SAFBehavior(Behavior):
    name = "ABM-SAF"

    def step_agent(self, agent):
        neigh = agent.neighbors()
        if neigh:
            mean_neighbor = float(np.mean([a.opinion for a in neigh]))
        else:
            mean_neighbor = agent.opinion

        # Habilidad SAF agregada: no representa una persona real.
        skill = agent.saf_skill
        coupling = float(self.params.get("coupling", 0.20))
        field_pressure = float(self.params.get("field_pressure", 0.10))
        agent.next_opinion = clamp(
            agent.opinion
            + coupling * skill * (mean_neighbor - agent.opinion)
            + field_pressure * agent.exposure * (-agent.opinion)
        )


BEHAVIORS = {
    "Voter / difusión local": VoterBehavior,
    "Deffuant-Weisbuch": DeffuantBehavior,
    "ABM-SAF": SAFBehavior,
}


def get_behavior(name, params):
    cls = BEHAVIORS.get(name, SAFBehavior)
    return cls(params)


# ============================================================
# Agents
# ============================================================

class SeccionAgent(Agent):
    """
    Agente territorial agregado.
    En Mesa 3.x/4.x el ID lo administra Mesa; el primer parámetro
    es model. Se mantiene territorial_unit_id como identificador
    externo reproducible.
    """

    def __init__(
        self,
        model,
        territorial_unit_id,
        alcaldia,
        seccion,
        opinion=0.0,
        capital_social=0.5,
        acceso_informacion=0.5,
        influencia_liderazgo=0.5,
        arraigo=0.5,
        nivel_movilizacion=0.5,
        desconfianza=0.5,
        exposicion_problema=0.5,
        **kwargs
    ):
        super().__init__(model)
        self.territorial_unit_id = str(territorial_unit_id)
        self.alcaldia = str(alcaldia)
        self.seccion = str(seccion)
        self.manzana = str(kwargs.get("manzana", ""))
        self.opinion = clamp(opinion)
        self.next_opinion = self.opinion
        self.capital_social = float(capital_social)
        self.acceso_informacion = float(acceso_informacion)
        self.influencia_liderazgo = float(influencia_liderazgo)
        self.arraigo = float(arraigo)
        self.nivel_movilizacion = float(nivel_movilizacion)
        self.desconfianza = float(desconfianza)
        self.exposure = float(exposicion_problema)
        self.fatiga = 0.0
        self.influencia = 0.0
        self.es_broker = False
        self.es_adversario = False

    @property
    def spin(self):
        if self.opinion > 0.25:
            return 1
        if self.opinion < -0.25:
            return -1
        return 0

    @property
    def saf_skill(self):
        return clamp(
            0.30 * self.capital_social
            + 0.25 * self.acceso_informacion
            + 0.25 * self.influencia_liderazgo
            + 0.10 * self.arraigo
            + 0.10 * self.nivel_movilizacion
            - 0.15 * self.desconfianza,
            0, 1
        )

    def neighbors(self):
        return self.model.neighbors_of(self)

    def step(self):
        self.model.behavior.step_agent(self)

    def advance(self):
        self.opinion = clamp(self.next_opinion)


class BrokerAgent(SeccionAgent):
    """
    Actor territorial sintético/agregado.
    No representa una persona identificable.
    """

    def __init__(self, model, territorial_unit_id, alcaldia, seccion, **kwargs):
        super().__init__(
            model,
            territorial_unit_id,
            alcaldia,
            seccion,
            **kwargs
        )
        self.es_broker = True
        self.capital_social = float(kwargs.get("capital", 0.90))
        self.influencia_liderazgo = float(kwargs.get("liderazgo", 0.90))


# ============================================================
# SITERModel — Mesa
# ============================================================

class SITERModel(Model):
    """
    Mundo territorial CDMX implementado sobre Mesa.

    Scheduler moderno:
      RandomActivation    -> self.agents.shuffle_do("step")
      Simultaneous        -> self.agents.do("step") + advance()

    Se mantiene un NetworkX graph como representación canónica de
    relaciones; se intenta montar además el espacio de red de Mesa
    cuando la versión instalada lo permite.
    """

    def __init__(
        self,
        df_state,
        behavior="ABM-SAF",
        seed=42,
        p_intra=0.06,
        p_inter=0.015,
        activation="Simultaneous",
        **params
    ):
        super().__init__(seed=seed)
        self.seed_value = int(seed)
        self.behavior_name = behavior
        self.behavior = get_behavior(behavior, params)
        self.p_intra = float(p_intra)
        self.p_inter = float(p_inter)
        self.activation = activation
        self.params = params
        self.df_state = df_state.reset_index(drop=True).copy()

        self.G = nx.Graph()
        self.agent_by_tu = {}
        self._build_agents()
        self._build_network()
        self._build_mesa_space()

        self.datacollector = DataCollector(
            model_reporters={
                "SIMPATIZANTE": lambda m: m.count_spin(1),
                "OPOSITOR": lambda m: m.count_spin(-1),
                "INDECISO": lambda m: m.count_spin(0),
                "Gini": lambda m: m.compute_gini(),
                "Polarizacion": lambda m: m.compute_polarization(),
                "MeanOpinion": lambda m: m.mean_opinion(),
            },
            agent_reporters={
                "opinion": "opinion",
                "spin": "spin",
                "saf_skill": "saf_skill",
                "influencia": "influencia",
                "es_broker": "es_broker",
            },
        )
        self.datacollector.collect(self)

    def _build_agents(self):
        for _, row in self.df_state.iterrows():
            a = SeccionAgent(
                self,
                row["territorial_unit_id"],
                row["alcaldia"],
                row["seccion"],
                opinion=row.get("opinion_continua", 0.0),
                capital_social=row.get("capital_social", 0.5),
                acceso_informacion=row.get("acceso_informacion", 0.5),
                influencia_liderazgo=row.get("influencia_liderazgo", 0.5),
                arraigo=row.get("arraigo", 0.5),
                nivel_movilizacion=row.get("nivel_movilizacion", 0.5),
                desconfianza=row.get("desconfianza", 0.5),
                exposicion_problema=row.get("exposicion_problema", 0.5),
                manzana=row.get("manzana", ""),
            )
            self.agent_by_tu[a.territorial_unit_id] = a
            self.G.add_node(a.unique_id, territorial_unit_id=a.territorial_unit_id)

    def _build_network(self):
        rng = np.random.default_rng(self.seed_value)
        agents = list(self.agent_by_tu.values())
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                p = self.p_intra if a.alcaldia == b.alcaldia else self.p_inter
                if rng.random() < p:
                    self.G.add_edge(a.unique_id, b.unique_id)

        # Asegurar conectividad local mínima sin imponer una red completa.
        if len(agents) > 1:
            for i in range(len(agents) - 1):
                if not self.G.has_edge(agents[i].unique_id, agents[i+1].unique_id):
                    if rng.random() < 0.20:
                        self.G.add_edge(agents[i].unique_id, agents[i+1].unique_id)

        degree = dict(self.G.degree())
        max_degree = max(degree.values(), default=1)
        for a in agents:
            a.influencia = degree.get(a.unique_id, 0) / max_degree if max_degree else 0.0

    def _build_mesa_space(self):
        self.mesa_space = None
        if LegacyNetworkGrid is not None:
            try:
                self.mesa_space = LegacyNetworkGrid(self.G)
                for a in self.agent_by_tu.values():
                    self.mesa_space.place_agent(a, a.unique_id)
                return
            except Exception:
                self.mesa_space = None

        if MesaNetwork is not None:
            try:
                self.mesa_space = MesaNetwork(self.G, random=self.random)
                for a in self.agent_by_tu.values():
                    self.mesa_space[a.unique_id].agents.add(a)
            except Exception:
                self.mesa_space = None

    def neighbors_of(self, agent):
        ids = list(self.G.neighbors(agent.unique_id))
        reverse = {a.unique_id: a for a in self.agent_by_tu.values()}
        return [reverse[i] for i in ids if i in reverse]

    def step(self):
        if self.activation.lower().startswith("random"):
            self.agents.shuffle_do("step")
            # Random activation updates immediately through advance only
            # after all decisions have been calculated.
            self.agents.do("advance")
        else:
            self.agents.do("step")
            self.agents.do("advance")
        self.datacollector.collect(self)

    def count_spin(self, value):
        n = len(self.agents)
        return 0.0 if n == 0 else sum(a.spin == value for a in self.agents) / n

    def mean_opinion(self):
        return float(np.mean([a.opinion for a in self.agents])) if len(self.agents) else 0.0

    def compute_polarization(self):
        vals = np.asarray([a.opinion for a in self.agents], dtype=float)
        return float(np.std(vals)) if len(vals) else 0.0

    def compute_gini(self):
        return gini([abs(a.opinion) for a in self.agents])

    def inject_broker(self, territorial_unit_id=None, source_agent=None):
        if source_agent is None:
            if territorial_unit_id and territorial_unit_id in self.agent_by_tu:
                source_agent = self.agent_by_tu[territorial_unit_id]
            else:
                source_agent = self.random.choice(list(self.agent_by_tu.values()))

        broker_id = f"{source_agent.territorial_unit_id}-BROKER"
        if broker_id in self.agent_by_tu:
            return self.agent_by_tu[broker_id]

        broker = BrokerAgent(
            self,
            broker_id,
            source_agent.alcaldia,
            source_agent.seccion,
            opinion=source_agent.opinion,
            capital=0.90,
            liderazgo=0.90,
            acceso_informacion=0.90,
            arraigo=0.85,
            nivel_movilizacion=0.85,
            desconfianza=0.10,
            exposicion_problema=source_agent.exposure,
        )
        self.agent_by_tu[broker_id] = broker
        self.G.add_node(broker.unique_id, territorial_unit_id=broker_id)

        # Conecta el broker con los nodos territorialmente próximos
        # del mismo municipio/alcaldía.
        candidates = [
            a for a in self.agent_by_tu.values()
            if a is not broker and a.alcaldia == source_agent.alcaldia
        ]
        candidates = sorted(candidates, key=lambda a: a.influencia, reverse=True)[:5]
        for a in candidates:
            self.G.add_edge(broker.unique_id, a.unique_id)

        broker.influencia = 1.0
        return broker

    def model_dataframe(self):
        return self.datacollector.get_model_vars_dataframe()

    def agent_dataframe(self):
        return self.datacollector.get_agent_vars_dataframe()

    def snapshot(self):
        return {
            "seed": self.seed_value,
            "behavior": self.behavior_name,
            "activation": self.activation,
            "n_agents": len(self.agents),
            "n_edges": self.G.number_of_edges(),
            "step": int(self.steps),
            "mean_opinion": self.mean_opinion(),
            "gini": self.compute_gini(),
            "polarization": self.compute_polarization(),
        }


# ============================================================
# UI
# ============================================================

st.set_page_config(
    page_title="SITER-CDMX + Mesa",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 SITER-CDMX + Mesa")
st.caption(
    "Laboratorio computacional territorial · Streamlit + Mesa · "
    "CDMX únicamente · agentes agregados/sintéticos"
)

if not MESA_OK:
    st.error("Mesa no pudo importarse.")
    st.code(MESA_IMPORT_ERROR)
    st.stop()

with st.sidebar:
    st.header("⚙️ Setup")

    seed = st.number_input("Seed", min_value=0, max_value=999999, value=42)
    n_agents = st.slider("Unidades territoriales", 20, 500, 120, 10)

    behavior_name = st.selectbox("Behavior", list(BEHAVIORS.keys()))
    activation = st.radio(
        "Activación",
        ["Simultaneous", "Random"],
        index=0
    )

    p_intra = st.slider("p intra-alcaldía", 0.0, 0.30, 0.06, 0.005)
    p_inter = st.slider("p inter-alcaldía", 0.0, 0.10, 0.015, 0.005)

    confidence = st.slider("Confianza Deffuant", 0.01, 1.0, 0.25, 0.01)
    coupling = st.slider("Acoplamiento SAF", 0.0, 1.0, 0.20, 0.01)
    field_pressure = st.slider("Presión de campo", 0.0, 1.0, 0.10, 0.01)

    steps_to_run = st.slider("Pasos por Go", 1, 25, 1)

    if st.button("🔄 SETUP", use_container_width=True):
        st.session_state.pop("siter_model", None)
        st.session_state.pop("data_provider", None)
        st.rerun()

provider = DataProvider(seed=seed)

if "siter_model" not in st.session_state:
    df = provider.synthetic(n_agents)
    model = SITERModel(
        df,
        behavior=behavior_name,
        seed=seed,
        p_intra=p_intra,
        p_inter=p_inter,
        activation=activation,
        confidence=confidence,
        coupling=coupling,
        field_pressure=field_pressure,
    )
    st.session_state.siter_model = model
    st.session_state.source_df = df

model = st.session_state.siter_model

# Rebuild if setup controls changed
signature = (
    seed, n_agents, behavior_name, activation, round(p_intra, 4),
    round(p_inter, 4), round(confidence, 4), round(coupling, 4),
    round(field_pressure, 4)
)
if st.session_state.get("signature") != signature:
    df = provider.synthetic(n_agents)
    model = SITERModel(
        df,
        behavior=behavior_name,
        seed=seed,
        p_intra=p_intra,
        p_inter=p_inter,
        activation=activation,
        confidence=confidence,
        coupling=coupling,
        field_pressure=field_pressure,
    )
    st.session_state.siter_model = model
    st.session_state.source_df = df
    st.session_state.signature = signature

model = st.session_state.siter_model

# ------------------------------------------------------------
# NetLogo-like controls
# ------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    if st.button("▶ GO", use_container_width=True):
        for _ in range(steps_to_run):
            model.step()
        st.rerun()

with c2:
    if st.button("⏩ GO ×10", use_container_width=True):
        for _ in range(10):
            model.step()
        st.rerun()

with c3:
    if st.button("🧬 INSERT BROKER", use_container_width=True):
        model.inject_broker()
        st.rerun()

with c4:
    if st.button("↩ RESET", use_container_width=True):
        st.session_state.pop("siter_model", None)
        st.rerun()

# ------------------------------------------------------------
# Monitors
# ------------------------------------------------------------
snap = model.snapshot()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Paso", snap["step"])
m2.metric("Simpatizante", f"{model.count_spin(1):.2%}")
m3.metric("Opositor", f"{model.count_spin(-1):.2%}")
m4.metric("Gini", f"{model.compute_gini():.3f}")
m5.metric("Polarización", f"{model.compute_polarization():.3f}")

# ------------------------------------------------------------
# Main view
# ------------------------------------------------------------
left, right = st.columns([1.25, 1])

with left:
    st.subheader("🌐 World View — Red territorial")
    pos = nx.spring_layout(model.G, seed=seed, k=0.45, iterations=30)
    edges_x, edges_y = [], []
    for u, v in model.G.edges():
        edges_x += [pos[u][0], pos[v][0], None]
        edges_y += [pos[u][1], pos[v][1], None]

    st.plotly_chart(
        go.Figure(
            data=[
                go.Scatter(
                    x=edges_x,
                    y=edges_y,
                    mode="lines",
                    hoverinfo="skip",
                ),
                go.Scatter(
                    x=[pos[a.unique_id][0] for a in model.agents],
                    y=[pos[a.unique_id][1] for a in model.agents],
                    mode="markers",
                    text=[
                        f"{a.alcaldia}<br>{a.territorial_unit_id}"
                        for a in model.agents
                    ],
                    hovertemplate="%{text}<extra></extra>",
                    marker=dict(
                        size=[
                            14 if a.es_broker else 7
                            for a in model.agents
                        ],
                        color=[
                            a.opinion for a in model.agents
                        ],
                        colorscale="RdBu",
                        cmin=-1,
                        cmax=1,
                        showscale=True,
                        colorbar=dict(title="Estado"),
                    ),
                )
            ],
            layout=dict(
                height=650,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
            ),
        ),
        use_container_width=True,
    )

with right:
    st.subheader("📈 Monitores")
    hist = model.model_dataframe().reset_index()
    if not hist.empty:
        fig = go.Figure()
        for col in ["SIMPATIZANTE", "OPOSITOR", "INDECISO"]:
            if col in hist:
                fig.add_trace(go.Scatter(
                    x=hist["Step"],
                    y=hist[col],
                    mode="lines+markers",
                    name=col,
                ))
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        for col in ["Gini", "Polarizacion", "MeanOpinion"]:
            if col in hist:
                fig2.add_trace(go.Scatter(
                    x=hist["Step"],
                    y=hist[col],
                    mode="lines",
                    name=col,
                ))
        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------
# Territorial table
# ------------------------------------------------------------
st.subheader("🗺️ Estado territorial agregado")

rows = []
for a in model.agents:
    rows.append({
        "territorial_unit_id": a.territorial_unit_id,
        "alcaldia": a.alcaldia,
        "seccion": a.seccion,
        "opinion": round(a.opinion, 4),
        "spin": a.spin,
        "saf_skill": round(a.saf_skill, 4),
        "influencia": round(a.influencia, 4),
        "broker": bool(a.es_broker),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Experiment / reproducibility
# ------------------------------------------------------------
st.subheader("🧪 Experiment / reproducibilidad")

experiment_payload = {
    "seed": seed,
    "behavior": behavior_name,
    "activation": activation,
    "n_agents": n_agents,
    "p_intra": p_intra,
    "p_inter": p_inter,
    "confidence": confidence,
    "coupling": coupling,
    "field_pressure": field_pressure,
    "step": int(model.steps),
}
experiment_id = "EXP-" + sha256_obj(experiment_payload)[:12]
output_hash = sha256_obj({
    "experiment": experiment_payload,
    "snapshot": model.snapshot(),
})

e1, e2 = st.columns(2)
e1.code(f"experiment_id = {experiment_id}")
e2.code(f"output_hash = {output_hash[:20]}…")

export = {
    "metadata": {
        "experiment_id": experiment_id,
        "seed": seed,
        "output_hash": output_hash,
        "data_origin": "CALCULATED_FROM_SYNTHETIC_OR_AGGREGATED",
        "geography": "CDMX",
    },
    "model": model.snapshot(),
    "territorial_fields": rows,
    "model_timeseries": model.model_dataframe().reset_index().to_dict("records"),
}

st.download_button(
    "⬇️ Exportar experimento JSON",
    data=json.dumps(export, ensure_ascii=False, indent=2),
    file_name=f"{experiment_id}.json",
    mime="application/json",
)

st.info(
    "Principio metodológico: la granularidad de la conclusión no debe "
    "superar la granularidad de la evidencia. Este esqueleto trabaja "
    "con unidades territoriales agregadas/sintéticas."
)
