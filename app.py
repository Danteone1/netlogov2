
"""
SITER-CAE v5.6 MESA WORLD
=========================
Laboratorio territorial para CDMX.

OBJETIVO
--------
Unificar en una sola app:
1) Motor ABM con Mesa.
2) World View tipo NetLogo sobre GIS real cuando exista SHP/GeoJSON.
3) Cinco modos de datos:
   - REAL: CSV + SHP/GeoJSON.
   - DUMMY: datos mínimos de prueba.
   - SINTÉTICO COHERENTE: generado a partir de distribuciones/correlaciones
     del CSV real, sin copiar filas identificables.
   - SINTÉTICO PURO: universo sintético libre.
   - SINTÉTICO CALIBRACIÓN: escenarios de prueba controlados.
4) Análisis estadístico territorial.
5) Presupuesto: costos, cobertura, costo/km, costo/unidad, ROI operacional,
   sensibilidad y Monte Carlo.
6) Brigadistas: plan, GPS observado, cobertura por segmento, territorio
   visitado, horas, km, desviaciones y calidad GPS.
7) Question Engine para responder preguntas de cliente a partir de métricas.
8) Reproducibilidad: seed + experiment_id + output_hash.
9) Export JSON/CSV.

NOTA METODOLÓGICA
-----------------
Los estados de opinión y actores son variables de simulación agregadas.
No se construyen perfiles individuales ni PII. La app está diseñada para
análisis territorial, logística, presupuesto, resiliencia y escenarios.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------
# Mesa
# ---------------------------------------------------------------------
try:
    from mesa import Agent, Model
    MESA_OK = True
    MESA_ERROR = ""
except Exception as exc:
    MESA_OK = False
    MESA_ERROR = repr(exc)

# ---------------------------------------------------------------------
# GIS opcional pero recomendado para SHP/GeoJSON
# ---------------------------------------------------------------------
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    from shapely.ops import unary_union
    HAS_GIS = True
    GIS_ERROR = ""
except Exception as exc:
    HAS_GIS = False
    GIS_ERROR = repr(exc)

# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------
ALCALDIAS = [
    "ALVARO OBREGON",
    "AZCAPOTZALCO",
    "BENITO JUAREZ",
    "COYOACAN",
    "CUAJIMALPA DE MORELOS",
    "CUAUHTEMOC",
    "GUSTAVO A MADERO",
    "IZTACALCO",
    "IZTAPALAPA",
    "LA MAGDALENA CONTRERAS",
    "MIGUEL HIDALGO",
    "MILPA ALTA",
    "TLAHUAC",
    "TLALPAN",
    "VENUSTIANO CARRANZA",
    "XOCHIMILCO",
]

ALCALDIA_COORDS = {
    "CUAUHTEMOC": (19.4326, -99.1332),
    "BENITO JUAREZ": (19.3984, -99.1576),
    "MIGUEL HIDALGO": (19.4285, -99.2000),
    "COYOACAN": (19.3467, -99.1617),
    "IZTAPALAPA": (19.3550, -99.0620),
    "GUSTAVO A MADERO": (19.4900, -99.1100),
    "ALVARO OBREGON": (19.3580, -99.2270),
    "TLALPAN": (19.2880, -99.1670),
    "XOCHIMILCO": (19.2630, -99.1040),
    "VENUSTIANO CARRANZA": (19.4200, -99.1000),
    "AZCAPOTZALCO": (19.4870, -99.1860),
    "IZTACALCO": (19.3950, -99.0980),
    "CUAJIMALPA DE MORELOS": (19.3570, -99.2900),
    "LA MAGDALENA CONTRERAS": (19.3200, -99.2400),
    "TLAHUAC": (19.2700, -99.0050),
    "MILPA ALTA": (19.1920, -99.0230),
}

STATE_LABEL = {1: "SIMPATIZANTE", -1: "OPOSITOR", 0: "INDECISO"}
STATE_COLORS = {"SIMPATIZANTE": "#2ecc71", "OPOSITOR": "#e74c3c", "INDECISO": "#95a5a6"}
FIELD_COLORS = {
    "CONSOLIDACION": "#2ecc71",
    "DISPUTA_ABIERTA": "#e74c3c",
    "CONTENCION": "#3498db",
}

TRAIT_COLS = [
    "capital_social",
    "acceso_informacion",
    "influencia_liderazgo",
    "arraigo",
    "nivel_movilizacion",
    "desconfianza",
    "exposicion_problema",
]

NUMERIC_CANDIDATES = TRAIT_COLS + [
    "opinion_continua",
    "temperatura",
    "resistencia_institucional",
    "prioridad_problema",
    "poblacion",
    "longitud_km",
]

# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def norm_col(x: Any) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9áéíóúüñ]+", "_", s)
    return s.strip("_")


def clean_alcaldia(x: Any) -> str:
    s = str(x).strip().upper()
    replacements = {
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "Ü": "U", "Ñ": "N",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    return s


def sha256_obj(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * r * math.asin(min(1, math.sqrt(a)))


def path_distance_m(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 2:
        return 0.0
    d = 0.0
    x = df.sort_values("timestamp") if "timestamp" in df.columns else df
    for i in range(1, len(x)):
        d += haversine_m(
            x.iloc[i-1]["lat"], x.iloc[i-1]["lon"],
            x.iloc[i]["lat"], x.iloc[i]["lon"]
        )
    return d


def gini(values) -> float:
    x = np.asarray(values, dtype=float)
    x = np.abs(x[np.isfinite(x)])
    if len(x) == 0 or np.allclose(x.sum(), 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n)


def entropy_from_counts(counts) -> float:
    vals = np.asarray(list(counts), dtype=float)
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 0.0
    p = vals / vals.sum()
    return float(-np.sum(p * np.log(p)))


def field_state(simpat, indec, stability, polarization):
    if stability >= 0.70 and max(simpat, 1 - simpat - indec) >= 0.50:
        return "CONSOLIDACION"
    if indec >= 0.35 or polarization >= 0.55 or stability < 0.45:
        return "DISPUTA_ABIERTA"
    return "CONTENCION"


def ensure_lat_lon(df: pd.DataFrame, seed=42) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    if "alcaldia" not in out:
        out["alcaldia"] = rng.choice(ALCALDIAS, len(out))
    out["alcaldia"] = out["alcaldia"].map(clean_alcaldia)
    lats, lons = [], []
    for a in out["alcaldia"]:
        lat, lon = ALCALDIA_COORDS.get(a, (19.35, -99.15))
        lats.append(lat + rng.normal(0, 0.007))
        lons.append(lon + rng.normal(0, 0.007))
    if "lat" not in out:
        out["lat"] = lats
    else:
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
        out["lat"] = out["lat"].fillna(pd.Series(lats, index=out.index))
    if "lon" not in out:
        out["lon"] = lons
    else:
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
        out["lon"] = out["lon"].fillna(pd.Series(lons, index=out.index))
    return out


def normalize_base(df: pd.DataFrame, seed=42) -> pd.DataFrame:
    out = df.copy()
    out.columns = [norm_col(c) for c in out.columns]
    aliases = {
        "seccion_electoral": "seccion",
        "seccion_id": "seccion",
        "section": "seccion",
        "alcaldía": "alcaldia",
        "delegacion": "alcaldia",
        "territory_id": "territorial_unit_id",
        "id_territorial": "territorial_unit_id",
        "id": "territorial_unit_id",
    }
    for a, b in aliases.items():
        if a in out.columns and b not in out.columns:
            out[b] = out[a]

    if "territorial_unit_id" not in out:
        if "seccion" in out:
            out["territorial_unit_id"] = "CDMX-SEC-" + out["seccion"].astype(str)
        else:
            out["territorial_unit_id"] = [f"CDMX-SEC-{i+1:05d}" for i in range(len(out))]

    if "seccion" not in out:
        out["seccion"] = out["territorial_unit_id"].astype(str)

    if "alcaldia" not in out:
        out["alcaldia"] = "NO_ESPECIFICADA"

    out["alcaldia"] = out["alcaldia"].map(clean_alcaldia)

    defaults = {
        "opinion_continua": 0.0,
        "capital_social": 0.5,
        "acceso_informacion": 0.5,
        "influencia_liderazgo": 0.5,
        "arraigo": 0.5,
        "nivel_movilizacion": 0.5,
        "desconfianza": 0.5,
        "exposicion_problema": 0.5,
        "temperatura": 0.5,
        "resistencia_institucional": 0.5,
        "prioridad_problema": 0.5,
        "poblacion": 1000.0,
    }
    for c, v in defaults.items():
        if c not in out:
            out[c] = v

    for c, default_value in defaults.items():
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(default_value)

    out["opinion_continua"] = out["opinion_continua"].clip(-1, 1)
    out = ensure_lat_lon(out, seed=seed)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# DataProvider
# ---------------------------------------------------------------------
class DataProvider:
    MODES = {
        "REAL · CSV + SHP/GeoJSON": "real",
        "DUMMY · prueba mínima": "dummy",
        "SINTÉTICO COHERENTE · derivado de real": "coherent",
        "SINTÉTICO PURO · generativo": "pure",
        "SINTÉTICO CALIBRACIÓN · pruebas controladas": "calib",
    }

    def __init__(self, seed=42):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def dummy(self, n=48):
        rng = self.rng
        rows = []
        for i in range(n):
            alc = ALCALDIAS[i % len(ALCALDIAS)]
            lat, lon = ALCALDIA_COORDS[alc]
            op = [-0.7, -0.2, 0.0, 0.2, 0.7][i % 5] + rng.normal(0, 0.04)
            rows.append({
                "territorial_unit_id": f"DUMMY-{i+1:04d}",
                "alcaldia": alc,
                "seccion": str(1000+i),
                "opinion_continua": float(np.clip(op, -1, 1)),
                "capital_social": float(0.35 + 0.45*rng.random()),
                "acceso_informacion": float(0.35 + 0.45*rng.random()),
                "influencia_liderazgo": float(0.25 + 0.65*rng.random()),
                "arraigo": float(0.3 + 0.6*rng.random()),
                "nivel_movilizacion": float(0.25 + 0.7*rng.random()),
                "desconfianza": float(0.15 + 0.55*rng.random()),
                "exposicion_problema": float(0.2 + 0.7*rng.random()),
                "prioridad_problema": float(0.2 + 0.8*rng.random()),
                "resistencia_institucional": float(0.2 + 0.6*rng.random()),
            })
        return normalize_base(pd.DataFrame(rows), self.seed), {"mode": "dummy", "source": "generated"}

    def pure(self, n=300):
        rng = self.rng
        rows = []
        for i in range(n):
            alc = str(rng.choice(ALCALDIAS))
            lat, lon = ALCALDIA_COORDS[alc]
            social = rng.beta(4, 4)
            info = rng.beta(4, 4)
            lead = rng.beta(4, 4)
            arraigo = rng.beta(4, 4)
            mov = rng.beta(4, 4)
            distrust = rng.beta(3, 5)
            exposure = rng.beta(4, 4)
            skill = np.clip(
                .30*social + .25*info + .25*lead + .10*arraigo + .10*mov - .15*distrust,
                0, 1
            )
            op = np.clip(0.7*(2*social-1) + 0.25*(2*lead-1) + rng.normal(0, .12), -1, 1)
            rows.append({
                "territorial_unit_id": f"SYN-{i+1:05d}",
                "alcaldia": alc,
                "seccion": str(2000+i),
                "lat": lat + rng.normal(0, .009),
                "lon": lon + rng.normal(0, .009),
                "opinion_continua": op,
                "capital_social": social,
                "acceso_informacion": info,
                "influencia_liderazgo": lead,
                "arraigo": arraigo,
                "nivel_movilizacion": mov,
                "desconfianza": distrust,
                "exposicion_problema": exposure,
                "prioridad_problema": exposure,
                "resistencia_institucional": rng.beta(2, 5),
                "poblacion": max(100, rng.lognormal(7.0, .45)),
                "saf_skill": skill,
            })
        return normalize_base(pd.DataFrame(rows), self.seed), {"mode": "synthetic_pure", "source": "generative"}

    def calibration(self, n=240, scenario="balance"):
        rng = self.rng
        rows = []
        scenarios = {
            "balance": lambda i: rng.normal(0, .12),
            "polarizacion": lambda i: (0.75 if i % 2 == 0 else -0.75) + rng.normal(0, .05),
            "fragmentacion": lambda i: [-.8, -.3, .0, .35, .8][i % 5] + rng.normal(0, .03),
            "consenso": lambda i: .55 + rng.normal(0, .05),
            "resistencia_alta": lambda i: rng.normal(0, .20),
            "red_hub": lambda i: rng.normal(0, .10),
        }
        fn = scenarios.get(scenario, scenarios["balance"])
        for i in range(n):
            alc = ALCALDIAS[i % len(ALCALDIAS)]
            lat, lon = ALCALDIA_COORDS[alc]
            op = float(np.clip(fn(i), -1, 1))
            rows.append({
                "territorial_unit_id": f"CAL-{scenario[:4].upper()}-{i+1:05d}",
                "alcaldia": alc,
                "seccion": str(3000+i),
                "lat": lat + rng.normal(0, .006),
                "lon": lon + rng.normal(0, .006),
                "opinion_continua": op,
                "capital_social": .55,
                "acceso_informacion": .55,
                "influencia_liderazgo": .55,
                "arraigo": .55,
                "nivel_movilizacion": .55,
                "desconfianza": .35,
                "exposicion_problema": .60,
                "prioridad_problema": .60,
                "resistencia_institucional": .80 if scenario == "resistencia_alta" else .35,
                "poblacion": 1000,
            })
        return normalize_base(pd.DataFrame(rows), self.seed), {
            "mode": "synthetic_calibration",
            "scenario": scenario,
            "source": "controlled_test"
        }

    def coherent_from_real(self, real_df, n=None):
        """Genera datos sintéticos sin copiar filas:
        - categorías por frecuencia
        - variables numéricas por distribución empírica
        - correlación aproximada mediante cópula gaussiana construida con
          correlación de rangos y eigen-decomposition (sin SciPy).
        """
        base = normalize_base(real_df, self.seed)
        n = int(n or len(base))
        rng = self.rng

        cats = {}
        for c in ["alcaldia"]:
            p = base[c].value_counts(normalize=True)
            cats[c] = rng.choice(p.index.to_numpy(), n, p=p.to_numpy())

        numeric = [c for c in TRAIT_COLS + ["opinion_continua", "temperatura",
                                            "resistencia_institucional", "prioridad_problema"]
                   if c in base.columns]
        X = base[numeric].apply(pd.to_numeric, errors="coerce").fillna(base[numeric].median())
        # Estandarización por rangos: aproxima correlación Spearman.
        ranks = X.rank(pct=True).to_numpy()
        R = np.corrcoef(ranks.T)
        R = np.nan_to_num(R, nan=0.0)
        R = (R + R.T) / 2
        np.fill_diagonal(R, 1.0)
        vals, vecs = np.linalg.eigh(R)
        vals = np.clip(vals, 1e-6, None)
        L = vecs @ np.diag(np.sqrt(vals))
        Z = rng.normal(size=(n, len(numeric))) @ L.T
        U = 0.5 * (1 + np.vectorize(math.erf)(Z / math.sqrt(2)))
        out = pd.DataFrame({"alcaldia": cats["alcaldia"]})
        for j, c in enumerate(numeric):
            arr = np.sort(X[c].to_numpy(dtype=float))
            q = np.clip(U[:, j], 0, 1)
            idx = np.minimum((q * (len(arr)-1)).astype(int), len(arr)-1)
            out[c] = arr[idx]

        # ID nuevo; nunca se reutilizan IDs del real.
        out["territorial_unit_id"] = [f"COH-{i+1:05d}" for i in range(n)]
        out["seccion"] = [f"C{i+1:05d}" for i in range(n)]
        out["poblacion"] = float(base["poblacion"].median()) if "poblacion" in base else 1000
        out = ensure_lat_lon(out, self.seed)
        meta = {
            "mode": "synthetic_coherent",
            "derived_from_real": True,
            "real_n": len(base),
            "synthetic_n": len(out),
            "numeric_basis": numeric,
            "source_hash": sha256_obj({
                "columns": list(base.columns),
                "shape": base.shape,
                "alcaldia_distribution": base["alcaldia"].value_counts().to_dict(),
            })
        }
        return normalize_base(out, self.seed), meta

    def real(self, base_csv, electoral_csv=None, socio_csv=None):
        if base_csv is None:
            raise ValueError("El modo REAL requiere el CSV base territorial.")
        base = pd.read_csv(base_csv)
        base = normalize_base(base, self.seed)

        if electoral_csv is not None:
            elec = pd.read_csv(electoral_csv)
            elec.columns = [norm_col(c) for c in elec.columns]
            if "seccion" in elec:
                elec["seccion"] = elec["seccion"].astype(str)
                # Si existen votos por partido/agrupación, se resume a shares
                if "votos" in elec.columns:
                    group = elec.groupby("seccion")["votos"].sum().rename("votos_total")
                    mx = elec.groupby("seccion")["votos"].max().rename("votos_max")
                    stats = pd.concat([group, mx], axis=1)
                    stats["share_max"] = (stats["votos_max"] / stats["votos_total"]).fillna(0)
                    base["seccion"] = base["seccion"].astype(str)
                    base = base.merge(stats[["share_max"]], left_on="seccion", right_index=True, how="left")
                    base["opinion_continua"] = (2*base["share_max"].fillna(.33)-1).clip(-1,1)

        if socio_csv is not None:
            socio = pd.read_csv(socio_csv)
            socio.columns = [norm_col(c) for c in socio.columns]
            if "seccion" in socio.columns:
                socio["seccion"] = socio["seccion"].astype(str)
                base["seccion"] = base["seccion"].astype(str)
                keep = [c for c in socio.columns if c != "alcaldia"]
                base = base.merge(socio[keep], on="seccion", how="left", suffixes=("", "_socio"))

        base = normalize_base(base, self.seed)
        meta = {
            "mode": "real",
            "source": "csv",
            "n": len(base),
            "columns": list(base.columns),
            "source_hash": sha256_obj({
                "shape": base.shape,
                "columns": list(base.columns),
            }),
        }
        return base, meta


# ---------------------------------------------------------------------
# GIS loader
# ---------------------------------------------------------------------
def read_vector_upload(uploaded) -> Optional["gpd.GeoDataFrame"]:
    if uploaded is None or not HAS_GIS:
        return None
    suffix = Path(uploaded.name).suffix.lower()
    data = uploaded.getvalue()
    tmpdir = Path(tempfile.mkdtemp(prefix="siter_gis_"))
    if suffix == ".zip":
        zpath = tmpdir / uploaded.name
        zpath.write_bytes(data)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmpdir / "unzipped")
        shps = list((tmpdir / "unzipped").rglob("*.shp"))
        if not shps:
            raise ValueError("El ZIP no contiene un .shp.")
        return gpd.read_file(shps[0])
    if suffix in [".geojson", ".json", ".gpkg", ".shp"]:
        p = tmpdir / uploaded.name
        p.write_bytes(data)
        return gpd.read_file(p)
    raise ValueError("Formato GIS no soportado. Usa ZIP de Shapefile, GeoJSON o GPKG.")


def normalize_gdf(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    g = gdf.copy()
    g.columns = [norm_col(c) for c in g.columns]
    if g.crs is None:
        # Se asume WGS84 solo cuando el usuario no proporcionó CRS.
        g = g.set_crs(4326, allow_override=True)
    if g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    if "alcaldia" in g:
        g["alcaldia"] = g["alcaldia"].map(clean_alcaldia)
    return g


def join_base_to_geometry(df, gdf):
    if gdf is None:
        return None
    g = normalize_gdf(gdf)
    # Preferir ID territorial; si no, seccion.
    for key in ["territorial_unit_id", "seccion"]:
        if key in df.columns and key in g.columns:
            left = df.copy()
            left[key] = left[key].astype(str)
            g[key] = g[key].astype(str)
            merged = g.merge(
                left.drop_duplicates(key),
                on=key,
                how="left",
                suffixes=("_gis", "")
            )
            return merged
    # Si no hay llave común, conservar geometría y hacer centroides; el usuario
    # puede trabajar con la capa GIS aunque el CSV no se haya unido.
    return g


# ---------------------------------------------------------------------
# Behaviors Mesa
# ---------------------------------------------------------------------
class Behavior:
    name = "base"
    def __init__(self, params=None):
        self.params = params or {}
    def step_agent(self, agent):
        raise NotImplementedError


class VoterBehavior(Behavior):
    name = "Voter / difusión local"
    def step_agent(self, agent):
        ns = agent.get_neighbors()
        if not ns:
            return
        other = agent.model.random.choice(ns)
        beta = float(self.params.get("beta", 1.2))
        prob = min(0.90, max(0.0, other.influencia * beta))
        if agent.model.random.random() < prob:
            agent.next_opinion = clamp(agent.opinion + .20*(other.opinion-agent.opinion))


class DeffuantBehavior(Behavior):
    name = "Deffuant-Weisbuch"
    def step_agent(self, agent):
        ns = agent.get_neighbors()
        if not ns:
            agent.next_opinion = agent.opinion
            return
        other = agent.model.random.choice(ns)
        eps = float(self.params.get("epsilon", .40))
        mu = float(self.params.get("mu", .30))
        rep = float(self.params.get("epsilon_repulsion", .80))
        d = abs(agent.opinion-other.opinion)
        if d <= eps:
            agent.next_opinion = clamp(agent.opinion + mu*(other.opinion-agent.opinion))
        elif d >= rep:
            agent.next_opinion = clamp(agent.opinion - .5*mu*(other.opinion-agent.opinion))
        else:
            agent.next_opinion = agent.opinion


class SAFBehavior(Behavior):
    name = "ABM-SAF"
    def step_agent(self, agent):
        ns = agent.get_neighbors()
        if not ns:
            agent.next_opinion = agent.opinion
            return
        mean_n = float(np.mean([x.opinion for x in ns]))
        coupling = float(self.params.get("coupling", .20))
        field_pressure = float(self.params.get("field_pressure", .08))
        fatigue_penalty = max(0.0, 1.0-agent.fatiga)
        agent.next_opinion = clamp(
            agent.opinion
            + coupling*agent.saf_skill*fatigue_penalty*(mean_n-agent.opinion)
            + field_pressure*agent.exposure*(-agent.opinion)
        )


BEHAVIORS = {
    "Voter / difusión local": VoterBehavior,
    "Deffuant-Weisbuch": DeffuantBehavior,
    "ABM-SAF": SAFBehavior,
}


def clamp(x, lo=-1, hi=1):
    return float(max(lo, min(hi, float(x))))


# ---------------------------------------------------------------------
# Mesa model
# ---------------------------------------------------------------------
if MESA_OK:
    class SeccionAgent(Agent):
        def __init__(self, model, row):
            super().__init__(model)
            self.territorial_unit_id = str(row["territorial_unit_id"])
            self.alcaldia = str(row["alcaldia"])
            self.seccion = str(row["seccion"])
            self.lat = float(row["lat"])
            self.lon = float(row["lon"])
            self.opinion = clamp(row.get("opinion_continua", 0))
            self.next_opinion = self.opinion
            self.capital_social = float(row.get("capital_social", .5))
            self.acceso_informacion = float(row.get("acceso_informacion", .5))
            self.influencia_liderazgo = float(row.get("influencia_liderazgo", .5))
            self.arraigo = float(row.get("arraigo", .5))
            self.nivel_movilizacion = float(row.get("nivel_movilizacion", .5))
            self.desconfianza = float(row.get("desconfianza", .5))
            self.exposure = float(row.get("exposicion_problema", .5))
            self.resistencia_institucional = float(row.get("resistencia_institucional", .5))
            self.prioridad_problema = float(row.get("prioridad_problema", .5))
            self.fatiga = 0.0
            self.influencia = 0.0
            self.es_broker = False
            self.neighbor_ids = []

        @property
        def spin(self):
            if self.opinion > .25:
                return 1
            if self.opinion < -.25:
                return -1
            return 0

        @property
        def saf_skill(self):
            return clamp(
                .30*self.capital_social
                + .25*self.acceso_informacion
                + .25*self.influencia_liderazgo
                + .10*self.arraigo
                + .10*self.nivel_movilizacion
                - .15*self.desconfianza,
                0, 1
            )

        def get_neighbors(self):
            return [self.model.agent_by_uid[x] for x in self.neighbor_ids if x in self.model.agent_by_uid]

        def step(self):
            self.model.behavior.step_agent(self)

        def advance(self):
            self.opinion = clamp(self.next_opinion)

    class BrokerAgent(SeccionAgent):
        def __init__(self, model, row):
            super().__init__(model, row)
            self.es_broker = True

    class SITERModel(Model):
        def __init__(self, df, behavior_name="ABM-SAF", seed=42,
                     p_intra=.06, p_inter=.015, params=None):
            super().__init__(seed=int(seed))
            self.seed_value = int(seed)
            self.df = df.reset_index(drop=True).copy()
            self.behavior_name = behavior_name
            self.behavior = BEHAVIORS[behavior_name](params or {})
            self.p_intra = float(p_intra)
            self.p_inter = float(p_inter)
            self.params = params or {}
            self.G = nx.Graph()
            self.agent_by_uid = {}
            self._build_agents()
            self._build_network()
            self._collect_history()

        def _build_agents(self):
            for _, row in self.df.iterrows():
                a = SeccionAgent(self, row)
                self.agent_by_uid[str(a.unique_id)] = a
                self.G.add_node(str(a.unique_id), territorial_unit_id=a.territorial_unit_id,
                                alcaldia=a.alcaldia)

        def _build_network(self):
            rng = np.random.default_rng(self.seed_value)
            agents = list(self.agent_by_uid.values())
            for i, a in enumerate(agents):
                for b in agents[i+1:]:
                    p = self.p_intra if a.alcaldia == b.alcaldia else self.p_inter
                    if rng.random() < p:
                        self.G.add_edge(str(a.unique_id), str(b.unique_id))
            # Evitar red completamente vacía sin falsear conectividad excesiva.
            if len(agents) > 1 and self.G.number_of_edges() == 0:
                for i in range(len(agents)-1):
                    self.G.add_edge(str(agents[i].unique_id), str(agents[i+1].unique_id))

            degree = dict(self.G.degree())
            maxd = max(degree.values(), default=1)
            for a in agents:
                a.neighbor_ids = list(self.G.neighbors(str(a.unique_id)))
                a.influencia = degree.get(str(a.unique_id), 0) / maxd if maxd else 0

        def step(self):
            # Actualización simultánea.
            self.agents.do("step")
            self.agents.do("advance")
            for a in self.agents:
                a.fatiga *= .95
            self._collect_history()

        def _collect_history(self):
            self.history.append(self.metrics())

        @property
        def history(self):
            if not hasattr(self, "_history"):
                self._history = []
            return self._history

        def metrics(self):
            n = len(self.agents)
            return {
                "step": int(self.steps),
                "SIMPATIZANTE": self.count_spin(1),
                "OPOSITOR": self.count_spin(-1),
                "INDECISO": self.count_spin(0),
                "Gini": self.compute_gini(),
                "Polarizacion": self.compute_polarization(),
                "MeanOpinion": self.mean_opinion(),
                "edges": self.G.number_of_edges(),
                "density": nx.density(self.G) if n > 1 else 0,
            }

        def count_spin(self, v):
            n = len(self.agents)
            return 0 if n == 0 else sum(a.spin == v for a in self.agents) / n

        def mean_opinion(self):
            return float(np.mean([a.opinion for a in self.agents])) if len(self.agents) else 0

        def compute_polarization(self):
            vals = np.array([a.opinion for a in self.agents], dtype=float)
            return float(np.std(vals)) if len(vals) else 0

        def compute_gini(self):
            return gini([abs(a.opinion) for a in self.agents])

        def agent_dataframe(self):
            return pd.DataFrame([{
                "agent_id": str(a.unique_id),
                "territorial_unit_id": a.territorial_unit_id,
                "alcaldia": a.alcaldia,
                "seccion": a.seccion,
                "lat": a.lat,
                "lon": a.lon,
                "opinion": a.opinion,
                "spin": a.spin,
                "intencion": STATE_LABEL[a.spin],
                "capital_social": a.capital_social,
                "influencia": a.influencia,
                "saf_skill": a.saf_skill,
                "resistencia_institucional": a.resistencia_institucional,
                "prioridad_problema": a.prioridad_problema,
                "fatiga": a.fatiga,
                "broker": a.es_broker,
            } for a in self.agents])

        def insert_broker(self, target_uid=None):
            if target_uid and target_uid in self.agent_by_uid:
                base = self.agent_by_uid[target_uid]
            else:
                base = max(self.agents, key=lambda x: x.influencia)
            row = {
                "territorial_unit_id": f"{base.territorial_unit_id}-BRK",
                "alcaldia": base.alcaldia,
                "seccion": base.seccion,
                "lat": base.lat,
                "lon": base.lon,
                "opinion_continua": base.opinion,
                "capital_social": .90,
                "acceso_informacion": .90,
                "influencia_liderazgo": .90,
                "arraigo": .85,
                "nivel_movilizacion": .85,
                "desconfianza": .10,
                "exposicion_problema": base.exposure,
                "resistencia_institucional": base.resistencia_institucional,
                "prioridad_problema": base.prioridad_problema,
            }
            b = BrokerAgent(self, row)
            self.agent_by_uid[str(b.unique_id)] = b
            self.G.add_node(str(b.unique_id), territorial_unit_id=b.territorial_unit_id,
                            alcaldia=b.alcaldia)
            candidates = sorted(
                [a for a in self.agents if a is not b and a.alcaldia == base.alcaldia],
                key=lambda x: x.influencia, reverse=True
            )[:8]
            for a in candidates:
                self.G.add_edge(str(b.unique_id), str(a.unique_id))
            b.neighbor_ids = [str(a.unique_id) for a in candidates]
            for a in candidates:
                if str(b.unique_id) not in a.neighbor_ids:
                    a.neighbor_ids.append(str(b.unique_id))
            b.influencia = 1.0
            return b


# ---------------------------------------------------------------------
# Estadística territorial
# ---------------------------------------------------------------------
class TerritorialAnalytics:
    @staticmethod
    def table(adf):
        rows = []
        for alc, sub in adf.groupby("alcaldia", dropna=False):
            counts = sub["intencion"].value_counts()
            n = len(sub)
            simpat = (sub["intencion"] == "SIMPATIZANTE").mean()
            opos = (sub["intencion"] == "OPOSITOR").mean()
            indec = (sub["intencion"] == "INDECISO").mean()
            ent = entropy_from_counts(counts.values)
            stab = 1 - (ent / math.log(3)) if n else 0
            pol = float(sub["opinion"].std()) if len(sub) > 1 else 0
            resistencia = float(sub["resistencia_institucional"].mean())
            prioridad = float(sub["prioridad_problema"].mean())
            campo = field_state(simpat, indec, stab, pol)
            rows.append({
                "alcaldia": alc,
                "n_unidades": n,
                "simpat_pct": simpat*100,
                "opos_pct": opos*100,
                "indec_pct": indec*100,
                "entropia": ent,
                "estabilidad": stab,
                "polarizacion_std": pol,
                "campo": campo,
                "influencia_prom": sub["influencia"].mean(),
                "saf_skill_prom": sub["saf_skill"].mean(),
                "resistencia_prom": resistencia,
                "prioridad_prom": prioridad,
                "poblacion_proxy": sub.get("poblacion", pd.Series([1000]*n)).sum()
                    if "poblacion" in sub else n*1000,
            })
        return pd.DataFrame(rows).sort_values("alcaldia").reset_index(drop=True)

    @staticmethod
    def network(model):
        G = model.G
        if len(G) == 0:
            return {}
        deg = np.array([d for _, d in G.degree()], dtype=float)
        clustering = nx.average_clustering(G) if len(G) > 2 else 0
        try:
            centralization = (max(deg) - deg.mean()) / max(len(G)-2, 1)
        except Exception:
            centralization = 0
        inf = deg / max(deg.max(), 1)
        return {
            "nodes": len(G),
            "edges": G.number_of_edges(),
            "density": nx.density(G) if len(G) > 1 else 0,
            "degree_mean": deg.mean() if len(deg) else 0,
            "degree_max": deg.max() if len(deg) else 0,
            "clustering": clustering,
            "centralization_degree": float(centralization),
            "gini_influence": gini(inf),
            "top1_share": float(np.max(inf)/np.sum(inf)) if np.sum(inf) else 0,
            "top10_share": float(np.sort(inf)[::-1][:max(1, int(.10*len(inf)))].sum()/np.sum(inf))
                if np.sum(inf) else 0,
        }

    @staticmethod
    def spof(model, top_k=10):
        base = model.metrics()["SIMPATIZANTE"]
        rows = []
        # Aproximación operacional: eliminar temporalmente nodos de mayor grado.
        candidates = sorted(model.G.degree(), key=lambda x: x[1], reverse=True)[:top_k]
        for uid, deg in candidates:
            a = model.agent_by_uid.get(str(uid))
            if a is None:
                continue
            G2 = model.G.copy()
            G2.remove_node(uid)
            # proxy: componentes / centralidad y estado agregado sin re-simular.
            loss_proxy = (deg / max(1, model.G.number_of_edges())) * base
            rows.append({
                "agent_id": str(uid),
                "territorial_unit_id": a.territorial_unit_id,
                "alcaldia": a.alcaldia,
                "grado": deg,
                "delta_proxy": -loss_proxy,
                "clasificacion": "CRITICO" if loss_proxy >= .08 else "MODERADO",
            })
        return pd.DataFrame(rows)

    @staticmethod
    def centers(adf, top_k=20):
        return adf.sort_values(["saf_skill", "influencia"], ascending=False).head(top_k).copy()


# ---------------------------------------------------------------------
# Brigadistas / GPS
# ---------------------------------------------------------------------
class FieldOperations:
    @staticmethod
    def load_gps(uploaded):
        if uploaded is None:
            return pd.DataFrame()
        name = uploaded.name.lower()
        data = uploaded.getvalue()
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(data))
            df.columns = [norm_col(c) for c in df.columns]
            if "latitude" in df and "lat" not in df:
                df["lat"] = df["latitude"]
            if "longitude" in df and "lon" not in df:
                df["lon"] = df["longitude"]
            if "time" in df and "timestamp" not in df:
                df["timestamp"] = df["time"]
            return FieldOperations.normalize_gps(df)
        if name.endswith(".geojson") or name.endswith(".json"):
            if HAS_GIS:
                g = gpd.read_file(io.BytesIO(data))
                rows = []
                for _, r in g.iterrows():
                    if r.geometry is not None and r.geometry.geom_type == "Point":
                        rows.append({
                            "lat": r.geometry.y,
                            "lon": r.geometry.x,
                            "timestamp": r.get("timestamp", len(rows)),
                            "brigada": r.get("brigada", "B-01"),
                            "accuracy_m": r.get("accuracy_m", np.nan),
                        })
                return FieldOperations.normalize_gps(pd.DataFrame(rows))
        # GPX parsing básico sin librería adicional.
        if name.endswith(".gpx"):
            import xml.etree.ElementTree as ET
            root = ET.fromstring(data.decode("utf-8", errors="ignore"))
            rows = []
            for pt in root.iter():
                tag = pt.tag.split("}")[-1]
                if tag == "trkpt":
                    rows.append({
                        "lat": float(pt.attrib["lat"]),
                        "lon": float(pt.attrib["lon"]),
                        "timestamp": len(rows),
                        "brigada": "B-01",
                    })
            return FieldOperations.normalize_gps(pd.DataFrame(rows))
        raise ValueError("GPS: usa CSV, GeoJSON o GPX.")

    @staticmethod
    def normalize_gps(df):
        if df.empty:
            return df
        out = df.copy()
        out.columns = [norm_col(c) for c in out.columns]
        required = ["lat", "lon"]
        for c in required:
            if c not in out:
                raise ValueError(f"GPS requiere columna {c}.")
        if "brigada" not in out:
            out["brigada"] = "B-01"
        if "timestamp" not in out:
            out["timestamp"] = np.arange(len(out))
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
        out = out.dropna(subset=["lat", "lon"]).copy()
        return out

    @staticmethod
    def plan_from_territories(adf, n_brig=4, steps=8):
        """Plan territorial reproducible sobre centroides.
        No afirma ser ruta peatonal: es un plan de cobertura territorial.
        """
        if adf.empty:
            return pd.DataFrame()
        ordered = adf.sort_values(["alcaldia", "seccion"]).reset_index(drop=True)
        chunks = np.array_split(ordered, min(n_brig, len(ordered)))
        rows = []
        for bi, chunk in enumerate(chunks, 1):
            if chunk.empty:
                continue
            for j, (_, r) in enumerate(chunk.iterrows()):
                rows.append({
                    "brigada": f"B-{bi:02d}",
                    "orden": j,
                    "territorial_unit_id": r["territorial_unit_id"],
                    "alcaldia": r["alcaldia"],
                    "seccion": r["seccion"],
                    "lat": r["lat"],
                    "lon": r["lon"],
                })
        return pd.DataFrame(rows)

    @staticmethod
    def planned_length_m(plan):
        if plan.empty:
            return 0
        total = 0
        for _, g in plan.groupby("brigada"):
            g = g.sort_values("orden")
            total += path_distance_m(g)
        return total

    @staticmethod
    def coverage_report(plan, gps, buffer_m=25):
        if plan.empty:
            return pd.DataFrame()
        if gps.empty:
            rows = []
            for b, g in plan.groupby("brigada"):
                rows.append({
                    "brigada": b,
                    "km_plan": path_distance_m(g)/1000,
                    "km_gps": 0,
                    "cobertura_ruta_pct": 0,
                    "n_puntos_gps": 0,
                    "territorios_plan": g["territorial_unit_id"].nunique(),
                    "territorios_observados": 0,
                    "calidad": "SIN GPS",
                })
            return pd.DataFrame(rows)

        rows = []
        for b, p in plan.groupby("brigada"):
            gg = gps[gps["brigada"].astype(str) == str(b)]
            p = p.sort_values("orden")
            km_plan = path_distance_m(p)/1000
            km_gps = path_distance_m(gg)/1000
            coverage = 0.0
            if HAS_GIS and len(p) >= 2 and len(gg) >= 2:
                # Convertir aproximadamente a metros locales.
                mean_lat = float(pd.concat([p["lat"], gg["lat"]]).mean())
                sx = 111320 * math.cos(math.radians(mean_lat))
                sy = 110540
                pxy = [(float(x)*sx, float(y)*sy) for x, y in zip(p["lon"], p["lat"])]
                gxy = [(float(x)*sx, float(y)*sy) for x, y in zip(gg["lon"], gg["lat"])]
                lp = LineString(pxy)
                lg = LineString(gxy)
                if lp.length > 0:
                    coverage = min(1.0, lp.buffer(buffer_m).intersection(lg).length / lp.length)
            else:
                # Sin Shapely: aproximación por cercanía de puntos a puntos plan.
                if not gg.empty:
                    hit = 0
                    for _, gp in gg.iterrows():
                        ds = [
                            haversine_m(gp["lat"], gp["lon"], pr["lat"], pr["lon"])
                            for _, pr in p.iterrows()
                        ]
                        if min(ds) <= buffer_m:
                            hit += 1
                    coverage = hit / max(1, len(gg))

            # Territorios observados: si hay geometría se puede sustituir por
            # point-in-polygon; aquí usamos cercanía al centroide como fallback.
            observed = set()
            for _, gp in gg.iterrows():
                if p.empty:
                    continue
                dists = [
                    haversine_m(gp["lat"], gp["lon"], pr["lat"], pr["lon"])
                    for _, pr in p.iterrows()
                ]
                j = int(np.argmin(dists))
                if min(dists) <= buffer_m:
                    observed.add(str(p.iloc[j]["territorial_unit_id"]))

            quality = "ALTA" if coverage >= .80 else "MEDIA" if coverage >= .50 else "BAJA"
            rows.append({
                "brigada": b,
                "km_plan": km_plan,
                "km_gps": km_gps,
                "cobertura_ruta_pct": coverage*100,
                "n_puntos_gps": len(gg),
                "territorios_plan": p["territorial_unit_id"].nunique(),
                "territorios_observados": len(observed),
                "calidad": quality,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def summary(plan, gps, report):
        return {
            "km_plan_total": float(report["km_plan"].sum()) if not report.empty else 0,
            "km_gps_total": float(report["km_gps"].sum()) if not report.empty else 0,
            "cobertura_promedio_pct": float(report["cobertura_ruta_pct"].mean()) if not report.empty else 0,
            "brigadas_con_gps": int((report["n_puntos_gps"] > 0).sum()) if not report.empty else 0,
            "territorios_observados": int(report["territorios_observados"].sum()) if not report.empty else 0,
        }


# ---------------------------------------------------------------------
# Presupuesto / Monte Carlo / asignación
# ---------------------------------------------------------------------
class BudgetEngine:
    @staticmethod
    def workload(terr):
        if terr.empty:
            return pd.DataFrame()
        t = terr.copy()
        t["workload"] = (
            .40*t["prioridad_prom"]
            + .25*t["n_unidades"].rank(pct=True)
            + .20*t["resistencia_prom"]
            + .15*t["polarizacion_std"].rank(pct=True)
        )
        t["workload"] = t["workload"].clip(0, 1)
        t["km_est"] = np.maximum(.5, np.sqrt(t["n_unidades"]) * .20)
        return t

    @staticmethod
    def allocation(terr, budget, fixed_per_brigada=120, hour_cost=120,
                   hours_per_brigada=8, max_brigadas=20):
        t = BudgetEngine.workload(terr)
        if t.empty:
            return t
        rows = []
        remaining = float(budget)
        for _, r in t.sort_values("workload", ascending=False).iterrows():
            base_cost = fixed_per_brigada + hour_cost*hours_per_brigada
            if remaining >= base_cost and len(rows) < max_brigadas:
                brig = 1
                remaining -= base_cost
            else:
                brig = 0
            hours = brig*hours_per_brigada
            coverage = min(1.0, brig*.55 + .20*r["workload"])
            benefit = coverage * (0.5 + 0.5*r["prioridad_prom"])
            roi = benefit / max(base_cost if brig else 1, 1)
            rows.append({
                "alcaldia": r["alcaldia"],
                "workload": r["workload"],
                "brigadas": brig,
                "horas": hours,
                "costo": base_cost if brig else 0,
                "cobertura_est": coverage,
                "beneficio_operacional": benefit,
                "roi_operacional": roi,
                "presupuesto_restante_local": max(0, remaining),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def monte_carlo(terr, budget, reps=300, fixed_per_brigada=120,
                    hour_cost=120, hours_per_brigada=8, seed=42):
        if terr.empty:
            return pd.DataFrame()
        rng = np.random.default_rng(seed)
        workload = BudgetEngine.workload(terr)
        rows = []
        for sim in range(reps):
            # Incertidumbre en costo, cobertura y productividad.
            cost_mult = rng.lognormal(mean=0, sigma=.10)
            prod_mult = np.clip(rng.normal(1.0, .12), .60, 1.40)
            remaining = budget
            total_cov = 0
            spent = 0
            for _, r in workload.sort_values("workload", ascending=False).iterrows():
                c = (fixed_per_brigada + hour_cost*hours_per_brigada)*cost_mult
                if remaining < c:
                    continue
                remaining -= c
                spent += c
                total_cov += min(1, (.50 + .35*r["workload"])*prod_mult)
            denom = max(len(workload), 1)
            cov_pct = min(100, 100*total_cov/denom)
            rows.append({
                "sim": sim+1,
                "cobertura_pct": cov_pct,
                "gasto": spent,
                "remanente": remaining,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def sensitivity(terr, budgets):
        rows = []
        for b in budgets:
            a = BudgetEngine.allocation(terr, b)
            rows.append({
                "presupuesto": b,
                "brigadas": int(a["brigadas"].sum()) if not a.empty else 0,
                "horas": int(a["horas"].sum()) if not a.empty else 0,
                "costo": float(a["costo"].sum()) if not a.empty else 0,
                "cobertura_est_pct": float(100*a["cobertura_est"].mean()) if not a.empty else 0,
                "beneficio_operacional": float(a["beneficio_operacional"].sum()) if not a.empty else 0,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------
class CalibrationEngine:
    @staticmethod
    def compare(real_df, synthetic_df, columns=None):
        if real_df is None or synthetic_df is None or len(real_df) == 0 or len(synthetic_df) == 0:
            return pd.DataFrame()
        cols = columns or [
            c for c in TRAIT_COLS + [
                "opinion_continua", "temperatura",
                "resistencia_institucional", "prioridad_problema", "poblacion"
            ]
            if c in real_df.columns and c in synthetic_df.columns
        ]
        rows = []
        for c in cols:
            r = pd.to_numeric(real_df[c], errors="coerce").dropna().to_numpy(dtype=float)
            s = pd.to_numeric(synthetic_df[c], errors="coerce").dropna().to_numpy(dtype=float)
            if len(r) < 3 or len(s) < 3:
                continue
            qs = np.linspace(.05, .95, 19)
            rq = np.quantile(r, qs)
            sq = np.quantile(s, qs)
            scale = max(np.std(r), 1e-9)
            qdist = float(np.mean(np.abs(rq - sq)) / scale)
            mean_gap = float(abs(np.mean(r) - np.mean(s)) / scale)
            std_gap = float(abs(np.std(r) - np.std(s)) / scale)
            rows.append({
                "variable": c,
                "media_real": np.mean(r),
                "media_sint": np.mean(s),
                "std_real": np.std(r),
                "std_sint": np.std(s),
                "gap_media_norm": mean_gap,
                "gap_std_norm": std_gap,
                "dist_cuantiles_norm": qdist,
                "calibracion": "BUENA" if qdist < .20 else "MEDIA" if qdist < .40 else "REVISAR",
            })
        return pd.DataFrame(rows)

    @staticmethod
    def score(real_df, synthetic_df):
        cmp = CalibrationEngine.compare(real_df, synthetic_df)
        if cmp.empty:
            return None
        score = 100 * max(0.0, 1.0 - float(cmp["dist_cuantiles_norm"].mean()))
        return round(score, 1)

# ---------------------------------------------------------------------
# Question Engine
# ---------------------------------------------------------------------
class QuestionEngine:
    """
    Respuestas calculadas desde el estado actual. Mantiene la lógica del
    catálogo SITER-CAE: red, campos, recursos, resiliencia, inercia y campo.
    No genera recomendaciones dirigidas a personas individuales.
    """

    CATALOG = [
        "¿Dónde está más polarizado?",
        "¿Qué alcaldía tiene mayor simpatía agregada?",
        "¿Qué territorio tiene mayor temperatura/problema?",
        "¿Dónde se moviliza más el territorio?",
        "¿Dónde hay mayor habilidad SAF agregada?",
        "¿Qué campo está en DISPUTA_ABIERTA?",
        "¿Qué campo está consolidado?",
        "¿Dónde hay mayor conflicto interno?",
        "¿Qué territorio está más fragmentado?",
        "¿Cuál tiene mayor institucionalización?",
        "¿Cuál es el estado dominante?",
        "¿Qué tan conectada está la red?",
        "¿La red parece de tribus o abierta?",
        "¿Hay centralización peligrosa?",
        "¿Qué tan desigual es la influencia?",
        "¿Cuáles son los centros SAF?",
        "¿Los centros están por encima del promedio?",
        "¿Cuántos brokers agregados hay?",
        "¿Cuánta influencia concentra el top 10%?",
        "¿En cuántos pasos converge?",
        "¿Dónde emergen centros durante la simulación?",
        "¿Cómo evoluciona la influencia?",
        "¿Dónde está el hotspot de influencia?",
        "¿Dónde está el hotspot de polarización?",
        "¿Las brigadas saturan territorio?",
        "¿Hasta dónde se observa una onda territorial?",
        "¿Qué centros están conectados?",
        "¿Qué territorio recibe más cobertura de brigada?",
        "¿Qué pasa si incremento la capacidad de brigadas?",
        "¿Qué pasa bajo un shock negativo?",
        "¿Qué pasa si aumento conectividad?",
        "¿Cuánto cubro con distintos presupuestos?",
        "¿Qué pasa si pierdo un nodo crítico?",
        "¿Qué opción tiene mejor relación costo/cobertura?",
        "¿Cuál es la probabilidad de mejorar la cobertura >5%?",
        "¿Qué cambia si inserto un actor agregado?",
        "¿Qué cambia con un actor agregado de signo contrario?",
        "¿Qué pasa con habilidad SAF baja?",
        "¿Dónde conviene concentrar capacidad territorial?",
        "¿Qué rasgos explican la habilidad SAF?",
        "¿Cuántas conexiones tiene el centro más conectado?",
        "¿Cómo cambia el resultado si baja el arraigo?",
        "¿Cuánto cuesta cubrir el universo?",
        "¿Qué territorio tiene mejor ROI operacional?",
        "¿Hay territorios aislados?",
        "¿Cuál es la intervención mínima de cobertura?",
        "¿Qué activos agregados están mejor posicionados?",
        "¿Es reproducible el experimento?",
        "¿Cómo exporto al motor?",
        "¿Cuál es el costo por unidad cubierta?",
        "¿Cómo afecta la fatiga de brigada?",
        "¿Cuándo se agota el presupuesto?",
        "¿Cómo afecta la pérdida de capacidad institucional?",
        "¿Qué significa epsilon de confianza?",
        "¿Qué ocurre con repulsión/echo chamber?",
        "¿Cómo medir polarización endógena?",
        "¿Qué pasa con epsilon bajo?",
        "¿Qué ocurre con actor agregado espejo?",
        "¿Cuánto cambia con contra-actor?",
        "¿Qué es un flanqueo territorial?",
        "¿Qué es decapitación de red?",
        "¿Cómo se detecta un centro estructural?",
        "¿Cuál es la distribución óptima del presupuesto?",
        "¿Qué territorio merece capacidad de enlace vs horas de campo?",
        "¿Cómo evoluciona el score de optimización?",
        "¿Qué nodos son SPOF?",
        "¿Cuánto cambia al retirar un nodo?",
        "¿La red es resiliente?",
        "¿Qué territorio tiene alta inercia institucional?",
        "¿Cuál es la barrera de adopción territorial?",
        "¿Dónde conviene trabajar con estructuras formales?",
        "¿Qué es la inercia normativa?",
        "¿Cómo se modela la fatiga de campo?",
        "¿Cuál es el costo de polarizar?",
        "¿Cómo se evita una cámara de eco?",
        "¿Qué aporta esta versión respecto al modelo base?",
    ]

    @staticmethod
    def answer(question, adf, terr, net, budget_df=None, brig_report=None,
               model=None):
        if adf.empty:
            return "No hay datos cargados."
        if question.startswith("¿Dónde está más polarizado?"):
            r = terr.loc[terr["polarizacion_std"].idxmax()]
            return f"{r.alcaldia}: polarización σ={r.polarizacion_std:.3f}."
        if "mayor simpatía" in question:
            r = terr.loc[terr["simpat_pct"].idxmax()]
            return f"{r.alcaldia}: {r.simpat_pct:.1f}% en la categoría agregada."
        if "temperatura" in question:
            r = adf.groupby("alcaldia")["temperatura"].mean().idxmax()
            return f"{r}: mayor temperatura/problema media observada."
        if "moviliza más" in question:
            r = adf.groupby("alcaldia")["nivel_movilizacion"].mean().idxmax()
            return f"{r}: mayor movilización agregada."
        if "habilidad SAF" in question:
            r = terr.loc[terr["saf_skill_prom"].idxmax()]
            return f"{r.alcaldia}: habilidad SAF media={r.saf_skill_prom:.3f}."
        if "DISPUTA_ABIERTA" in question:
            x = terr[terr["campo"]=="DISPUTA_ABIERTA"]
            return ", ".join(x["alcaldia"].tolist()) if not x.empty else "No hay alcaldías clasificadas así."
        if "consolidado" in question:
            x = terr[terr["campo"]=="CONSOLIDACION"]
            return ", ".join(x["alcaldia"].tolist()) if not x.empty else "No hay alcaldías consolidadas."
        if "conflicto interno" in question:
            r = terr.loc[terr["polarizacion_std"].idxmax()]
            return f"{r.alcaldia}: σ de opinión={r.polarizacion_std:.3f}."
        if "fragmentado" in question:
            r = terr.loc[terr["entropia"].idxmax()]
            return f"{r.alcaldia}: entropía={r.entropia:.3f}."
        if "institucionalización" in question:
            terr2 = terr.copy()
            terr2["institucionalizacion"] = .6*terr2["estabilidad"] + .4*(1-terr2["resistencia_prom"])
            r = terr2.loc[terr2["institucionalizacion"].idxmax()]
            return f"{r.alcaldia}: índice={r.institucionalizacion:.3f}."
        if "conectada" in question:
            return f"n={net.get('nodes',0)}, aristas={net.get('edges',0)}, densidad={net.get('density',0):.4f}."
        if "tribus" in question:
            c = net.get("clustering", 0)
            return f"Clustering={c:.3f}; valores altos sugieren mayor cierre local."
        if "centralización" in question:
            return f"Centralización de grado={net.get('centralization_degree',0):.3f}."
        if "desigual" in question:
            return f"Gini de influencia={net.get('gini_influence',0):.3f}; top10={100*net.get('top10_share',0):.1f}%."
        if "centros SAF" in question:
            c = TerritorialAnalytics.centers(adf)
            return "; ".join(f"{r.territorial_unit_id} ({r.saf_skill:.2f})" for _,r in c.head(8).iterrows())
        if "top 10%" in question:
            return f"El top 10% concentra aproximadamente {100*net.get('top10_share',0):.1f}% de la influencia normalizada."
        if "brigadas saturan" in question and brig_report is not None and not brig_report.empty:
            r = brig_report.sort_values("cobertura_ruta_pct", ascending=False).iloc[0]
            return f"{r.brigada}: cobertura de ruta {r.cobertura_ruta_pct:.1f}%; territorios observados {int(r.territorios_observados)}."
        if "costo" in question and "presupuesto" in question and budget_df is not None and not budget_df.empty:
            return f"Costo estimado de asignación: ${budget_df.costo.sum():,.0f}; {int(budget_df.brigadas.sum())} brigadas y {int(budget_df.horas.sum())} horas."
        if "mejor ROI" in question and budget_df is not None and not budget_df.empty:
            r = budget_df.loc[budget_df["roi_operacional"].idxmax()]
            return f"{r.alcaldia}: ROI operacional={r.roi_operacional:.6f} (proxy de cobertura/recursos)."
        if "agot" in question and budget_df is not None and not budget_df.empty:
            return f"Presupuesto utilizado ${budget_df.costo.sum():,.0f}; remanente ${max(0,budget_df.presupuesto_restante_local.min()):,.0f}."
        if "SPOF" in question or "punto" in question.lower() and "falla" in question.lower():
            s = TerritorialAnalytics.spof(model) if model is not None else pd.DataFrame()
            if s.empty:
                return "No hay análisis SPOF."
            return "; ".join(f"{r.agent_id}: {r.clasificacion}" for _,r in s.head(5).iterrows())
        if "reproducible" in question:
            return "Sí: el experimento registra seed, configuración y hash de salida."
        if "cobertura" in question and brig_report is not None and not brig_report.empty:
            return f"Cobertura promedio de ruta: {brig_report.cobertura_ruta_pct.mean():.1f}%."
        if "inercia" in question:
            r = terr.loc[terr["resistencia_prom"].idxmax()]
            return f"{r.alcaldia}: resistencia institucional media={r.resistencia_prom:.3f}."
        if "epsilon" in question:
            return "Epsilon es la distancia máxima de opinión para considerar interacción de confianza en Deffuant."
        if "polarización endógena" in question:
            return f"Se mide aquí mediante desviación estándar de opinion: {adf.opinion.std():.3f}."
        if "aporta esta versión" in question:
            return "Integra Mesa, GIS real, cinco modos de datos, presupuesto, campo/GPS, Monte Carlo, calibración y Question Engine."
        return "Respuesta calculada disponible a partir de los indicadores actuales; seleccione una pregunta más específica."

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="SITER-CAE v5.6 MESA WORLD", page_icon="🧬", layout="wide")

st.title("🧬 SITER-CAE v5.6 · MESA WORLD")
st.caption(
    "Laboratorio territorial CDMX · Mesa + GIS + presupuesto + brigadistas/GPS + "
    "datos reales y sintéticos + calibración"
)

if not MESA_OK:
    st.error("Mesa no está instalada. Revisa requirements.txt.")
    st.code(MESA_ERROR)
    st.stop()

if "siter" not in st.session_state:
    st.session_state.siter = {
        "model": None,
        "df": pd.DataFrame(),
        "meta": {},
        "gdf": None,
        "gps": pd.DataFrame(),
        "plan": pd.DataFrame(),
        "brig_report": pd.DataFrame(),
        "experiment_id": "",
        "hash": "",
        "gdf_joined": None,
        "real_reference": None,
    }
S = st.session_state.siter

# Sidebar
st.sidebar.header("⚙️ LABORATORIO")

mode_label = st.sidebar.selectbox("Fuente de datos", list(DataProvider.MODES.keys()))
mode = DataProvider.MODES[mode_label]
seed = st.sidebar.number_input("Seed", 1, 999999, 42)
n_units = st.sidebar.slider("Unidades sintéticas", 30, 1200, 300, 30)

if mode == "calib":
    calib_scenario = st.sidebar.selectbox(
        "Escenario de calibración",
        ["balance", "polarizacion", "fragmentacion", "consenso", "resistencia_alta", "red_hub"]
    )
else:
    calib_scenario = "balance"

base_file = electoral_file = socio_file = gis_file = gps_file = None
real_reference_df = None

if mode == "real":
    st.sidebar.markdown("### CSV")
    base_file = st.sidebar.file_uploader("CSV base territorial *", type=["csv"], key="base_csv")
    electoral_file = st.sidebar.file_uploader("CSV electoral agregado", type=["csv"], key="elec_csv")
    socio_file = st.sidebar.file_uploader("CSV socioeconómico agregado", type=["csv"], key="socio_csv")
    st.sidebar.markdown("### GIS")
    gis_file = st.sidebar.file_uploader(
        "SHP en ZIP / GeoJSON / GPKG",
        type=["zip", "geojson", "json", "gpkg"],
        key="gis_upload"
    )
    st.sidebar.markdown("### Observación de campo")
    gps_file = st.sidebar.file_uploader("GPS brigadistas: CSV / GPX / GeoJSON", type=["csv", "gpx", "geojson", "json"], key="gps_upload")

if mode == "coherent":
    st.sidebar.info("Primero carga un CSV real de referencia. El sintetizador conservará distribuciones y correlaciones aproximadas sin copiar filas.")

if mode in ["dummy", "pure", "calib"]:
    gps_file = st.sidebar.file_uploader("GPS brigadistas opcional", type=["csv", "gpx", "geojson", "json"], key="gps_upload_other")

behavior_name = st.sidebar.selectbox("Behavior Mesa", list(BEHAVIORS.keys()))
beta = st.sidebar.slider("β Voter", .1, 2.5, 1.2, .1)
epsilon = st.sidebar.slider("ε Deffuant", .05, 1.0, .40, .01)
mu = st.sidebar.slider("μ Deffuant", .05, .8, .30, .01)
epsilon_repulsion = st.sidebar.slider("ε repulsión", .5, 1.5, .80, .01)
coupling = st.sidebar.slider("Acoplamiento SAF", 0.0, 1.0, .20, .01)
field_pressure = st.sidebar.slider("Presión de campo", 0.0, 1.0, .08, .01)
p_intra = st.sidebar.slider("p intra-alcaldía", 0.0, .25, .06, .005)
p_inter = st.sidebar.slider("p inter-alcaldía", 0.0, .10, .015, .005)

st.sidebar.markdown("### Presupuesto")
budget = st.sidebar.number_input("Presupuesto operativo ($)", 0, 500000, 5000, 500)
fixed_brigada = st.sidebar.number_input("Costo fijo/brigada ($)", 0, 5000, 120, 20)
hour_cost = st.sidebar.number_input("Costo/hora ($)", 0, 1000, 120, 10)
hours_per_brigada = st.sidebar.number_input("Horas por brigada", 1, 24, 8)

st.sidebar.markdown("### Campo")
n_brig = st.sidebar.slider("Brigadas planificadas", 1, 30, 4)
gps_buffer = st.sidebar.slider("Tolerancia GPS (m)", 5, 100, 25, 5)

b1, b2 = st.sidebar.columns(2)
setup_clicked = b1.button("🔄 SETUP", use_container_width=True)
go_clicked = b2.button("▶ GO", use_container_width=True)

if setup_clicked:
    try:
        provider = DataProvider(seed=int(seed))

        S["gdf"] = None
        S["gdf_joined"] = None
        S["real_reference"] = None

        if mode == "real":
            df, meta = provider.real(base_file, electoral_file, socio_file)
            if gis_file is not None:
                if not HAS_GIS:
                    raise RuntimeError("Para SHP/GeoJSON instala geopandas + pyogrio + shapely.")
                S["gdf"] = read_vector_upload(gis_file)
                S["gdf"] = normalize_gdf(S["gdf"])
                S["gdf_joined"] = join_base_to_geometry(df, S["gdf"])
            else:
                S["gdf"] = None
                S["gdf_joined"] = None

        elif mode == "coherent":
            if base_file is None:
                raise ValueError("Para SINTÉTICO COHERENTE debes cargar el CSV real de referencia.")
            reference = pd.read_csv(base_file)
            real_reference_df = normalize_base(reference, int(seed))
            S["real_reference"] = real_reference_df.copy()
            df, meta = provider.coherent_from_real(reference, n=n_units)
            S["gdf"] = None
            S["gdf_joined"] = None

        elif mode == "dummy":
            df, meta = provider.dummy(n_units)

        elif mode == "pure":
            df, meta = provider.pure(n_units)

        else:
            df, meta = provider.calibration(n_units, calib_scenario)

        params = {
            "beta": beta,
            "epsilon": epsilon,
            "mu": mu,
            "epsilon_repulsion": epsilon_repulsion,
            "coupling": coupling,
            "field_pressure": field_pressure,
        }

        model = SITERModel(
            df,
            behavior_name=behavior_name,
            seed=int(seed),
            p_intra=p_intra,
            p_inter=p_inter,
            params=params,
        )

        # Persistir dataset y modelo
        S["df"] = df
        S["meta"] = meta
        S["model"] = model
        S["gps"] = FieldOperations.load_gps(gps_file) if gps_file is not None else pd.DataFrame()
        S["plan"] = FieldOperations.plan_from_territories(
            model.agent_dataframe(), n_brig=n_brig
        )
        S["brig_report"] = FieldOperations.coverage_report(
            S["plan"], S["gps"], buffer_m=gps_buffer
        )

        config = {
            "mode": mode,
            "seed": int(seed),
            "n_units": len(df),
            "behavior": behavior_name,
            "beta": beta,
            "epsilon": epsilon,
            "mu": mu,
            "epsilon_repulsion": epsilon_repulsion,
            "coupling": coupling,
            "field_pressure": field_pressure,
            "p_intra": p_intra,
            "p_inter": p_inter,
            "budget": budget,
        }
        S["experiment_id"] = "EXP-" + sha256_obj(config)[:12]
        S["hash"] = sha256_obj({
            "config": config,
            "data_meta": meta,
            "initial_head": df.head(25).to_dict("records"),
        })
        st.success(f"SETUP OK · {meta.get('mode')} · {len(df):,} unidades · {S['experiment_id']}")
    except Exception as exc:
        st.error(f"SETUP falló: {exc}")

if go_clicked:
    if S["model"] is None:
        st.warning("Primero ejecuta SETUP.")
    else:
        S["model"].step()
        S["plan"] = FieldOperations.plan_from_territories(
            S["model"].agent_dataframe(), n_brig=n_brig
        )
        S["brig_report"] = FieldOperations.coverage_report(
            S["plan"], S["gps"], buffer_m=gps_buffer
        )
        st.rerun()

if S["model"] is None:
    st.info("Ejecuta SETUP. Elige REAL para CSV+SHP; COHERENTE para sintetizar desde un CSV real; DUMMY/PURE/CALIB para pruebas.")
    st.markdown("""
### Modos de datos

| Modo | Uso |
|---|---|
| **REAL · CSV + SHP/GeoJSON** | Base real agregada + geografía real |
| **DUMMY** | Smoke tests rápidos |
| **SINTÉTICO COHERENTE** | Sintético estadísticamente derivado de un CSV real |
| **SINTÉTICO PURO** | Generación libre controlada por seed |
| **SINTÉTICO CALIBRACIÓN** | Pruebas de consenso, polarización, fragmentación, resistencia y redes |

### Qué se calibra

**Datos → indicadores → Mesa → escenarios → presupuesto → brigadas/GPS → respuestas cliente → export reproducible.**
""")
    st.stop()

model = S["model"]
adf = model.agent_dataframe()
terr = TerritorialAnalytics.table(adf)
net = TerritorialAnalytics.network(model)
budget_df = BudgetEngine.allocation(
    terr, budget, fixed_per_brigada=fixed_brigada,
    hour_cost=hour_cost, hours_per_brigada=hours_per_brigada
)

# ---------------------------------------------------------------------
# Monitores estilo NetLogo
# ---------------------------------------------------------------------
st.markdown("### 🎛️ WORLD CONTROL")

m = model.metrics()
cols = st.columns(7)
cols[0].metric("Tick", m["step"])
cols[1].metric("SIMPATIZANTE", f"{100*m['SIMPATIZANTE']:.1f}%")
cols[2].metric("OPOSITOR", f"{100*m['OPOSITOR']:.1f}%")
cols[3].metric("INDECISO", f"{100*m['INDECISO']:.1f}%")
cols[4].metric("Gini", f"{m['Gini']:.3f}")
cols[5].metric("Polarización", f"{m['Polarizacion']:.3f}")
cols[6].metric("Aristas", f"{m['edges']:,}")

tabs = st.tabs([
    "🌐 WORLD GIS / MESA",
    "📊 ESTADÍSTICA",
    "💰 PRESUPUESTO",
    "🥾 BRIGADISTAS / GPS",
    "🧪 CALIBRACIÓN",
    "🎯 CLIENTE",
    "📤 EXPORT",
])

# ---------------------------------------------------------------------
# World
# ---------------------------------------------------------------------
with tabs[0]:
    st.subheader("World View · GIS real cuando existe; red territorial como fallback")

    if S.get("gdf_joined") is not None and HAS_GIS:
        g = S["gdf_joined"].copy()
        # Convertir geometría a GeoJSON para pydeck si está disponible.
        try:
            import pydeck as pdk
            gj = json.loads(g.to_json())
            layer = pdk.Layer(
                "GeoJsonLayer",
                gj,
                pickable=True,
                stroked=True,
                filled=True,
                get_fill_color="[100, 150, 220, 90]",
                get_line_color="[50, 50, 50, 180]",
                line_width_min_pixels=1,
            )
            view = pdk.ViewState(
                latitude=float(adf["lat"].mean()),
                longitude=float(adf["lon"].mean()),
                zoom=10.5,
            )
            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=view,
                tooltip={"text": "{alcaldia} | {seccion}"},
            ), use_container_width=True)
            st.caption("Capa GIS autoritativa. Los atributos del CSV se unen por territorial_unit_id o seccion cuando existe llave común.")
        except Exception as exc:
            st.warning(f"No se pudo renderizar la capa GIS interactiva: {exc}")
    else:
        # Fallback visual: red + agentes, no se presenta como cartografía oficial.
        pos = nx.spring_layout(model.G, seed=int(seed), iterations=35)
        ex, ey = [], []
        for u, v in model.G.edges():
            ex += [pos[u][0], pos[v][0], None]
            ey += [pos[u][1], pos[v][1], None]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1), name="red"))
        fig.add_trace(go.Scatter(
            x=[pos[str(a.unique_id)][0] for a in model.agents],
            y=[pos[str(a.unique_id)][1] for a in model.agents],
            mode="markers",
            text=[f"{a.alcaldia}<br>{a.seccion}" for a in model.agents],
            marker=dict(
                size=[14 if a.es_broker else 8 for a in model.agents],
                color=[a.opinion for a in model.agents],
                colorscale="RdBu", cmin=-1, cmax=1,
                showscale=True,
            ),
            name="agentes",
        ))
        fig.update_layout(height=650, template="plotly_dark", xaxis_visible=False, yaxis_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        st.info("No hay SHP/GeoJSON unido. Esta vista es World View de red, no cartografía oficial.")

    st.markdown("#### Estado territorial")
    st.dataframe(terr, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------
with tabs[1]:
    st.subheader("Indicadores estadísticos para decisión territorial")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.bar(terr, x="alcaldia", y="simpat_pct", color="campo",
                   color_discrete_map=FIELD_COLORS,
                   title="Estado agregado por alcaldía"),
            use_container_width=True
        )
    with c2:
        st.plotly_chart(
            px.scatter(terr, x="resistencia_prom", y="prioridad_prom",
                       size="n_unidades", color="polarizacion_std",
                       hover_name="alcaldia",
                       title="Prioridad × resistencia × polarización"),
            use_container_width=True
        )

    st.markdown("#### Red")
    st.json(net)
    st.dataframe(TerritorialAnalytics.spof(model), use_container_width=True, hide_index=True)

    st.markdown("#### Series Mesa")
    hist = pd.DataFrame(model.history)
    if not hist.empty:
        st.plotly_chart(
            px.line(hist, x="step", y=["SIMPATIZANTE","OPOSITOR","INDECISO"],
                    title="Evolución de estados"),
            use_container_width=True
        )

# ---------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------
with tabs[2]:
    st.subheader("Presupuesto · cobertura · costo · sensibilidad · Monte Carlo")

    bcols = st.columns(6)
    bcols[0].metric("Presupuesto", f"${budget:,.0f}")
    bcols[1].metric("Costo asignado", f"${budget_df.costo.sum():,.0f}")
    bcols[2].metric("Brigadas", int(budget_df.brigadas.sum()))
    bcols[3].metric("Horas", int(budget_df.horas.sum()))
    bcols[4].metric("Cobertura est.", f"{100*budget_df.cobertura_est.mean():.1f}%")
    bcols[5].metric("Beneficio proxy", f"{budget_df.beneficio_operacional.sum():.2f}")

    st.dataframe(
        budget_df.sort_values("workload", ascending=False),
        use_container_width=True, hide_index=True
    )

    st.markdown("#### Sensibilidad presupuestaria")
    budgets = sorted(set([
        max(0, int(budget*.5)),
        max(0, int(budget)),
        int(budget*1.5),
        int(budget*2),
    ]))
    sens = BudgetEngine.sensitivity(terr, budgets)
    st.plotly_chart(
        px.line(sens, x="presupuesto", y="cobertura_est_pct",
                markers=True, title="Cobertura estimada vs presupuesto"),
        use_container_width=True
    )
    st.dataframe(sens, use_container_width=True, hide_index=True)

    st.markdown("#### Monte Carlo")
    reps = st.slider("Repeticiones Monte Carlo", 50, 1000, 300, 50, key="mc_reps")
    mc = BudgetEngine.monte_carlo(
        terr, budget, reps=reps,
        fixed_per_brigada=fixed_brigada,
        hour_cost=hour_cost,
        hours_per_brigada=hours_per_brigada,
        seed=int(seed)
    )
    if not mc.empty:
        p95 = np.percentile(mc["cobertura_pct"], 95)
        p50 = np.percentile(mc["cobertura_pct"], 50)
        p05 = np.percentile(mc["cobertura_pct"], 5)
        st.write({
            "P05_cobertura": round(p05, 2),
            "P50_cobertura": round(p50, 2),
            "P95_cobertura": round(p95, 2),
            "P(mejora >5pp)": round(float((mc["cobertura_pct"] > 5).mean()), 3),
        })
        st.plotly_chart(
            px.histogram(mc, x="cobertura_pct", nbins=30,
                         title="Distribución Monte Carlo de cobertura"),
            use_container_width=True
        )

# ---------------------------------------------------------------------
# Brigadistas
# ---------------------------------------------------------------------
with tabs[3]:
    st.subheader("Trabajo de brigadistas · plan vs observación GPS")

    s = FieldOperations.summary(S["plan"], S["gps"], S["brig_report"])
    c = st.columns(5)
    c[0].metric("Km plan", f"{s['km_plan_total']:.2f}")
    c[1].metric("Km GPS", f"{s['km_gps_total']:.2f}")
    c[2].metric("Cobertura ruta", f"{s['cobertura_promedio_pct']:.1f}%")
    c[3].metric("Brigadas con GPS", s["brigadas_con_gps"])
    c[4].metric("Territorios observados", s["territorios_observados"])

    st.caption(
        "La cobertura se calcula por coincidencia espacial de la traza GPS con el plan, "
        "no como GPS/plan, para evitar contar vueltas adicionales como cobertura."
    )

    if not S["brig_report"].empty:
        st.dataframe(S["brig_report"], use_container_width=True, hide_index=True)

    # Mapa de campo
    if not S["plan"].empty:
        fig = go.Figure()
        for b, p in S["plan"].groupby("brigada"):
            p = p.sort_values("orden")
            fig.add_trace(go.Scattermap(
                lat=p["lat"], lon=p["lon"], mode="lines+markers",
                name=f"Plan {b}", hovertext=p["territorial_unit_id"]
            ))
        if not S["gps"].empty:
            for b, gg in S["gps"].groupby("brigada"):
                gg = gg.sort_values("timestamp")
                fig.add_trace(go.Scattermap(
                    lat=gg["lat"], lon=gg["lon"], mode="lines",
                    name=f"GPS {b}"
                ))
        fig.update_layout(
            map=dict(style="open-street-map", zoom=10),
            height=600,
            margin=dict(l=0,r=0,t=0,b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Plan territorial")
    st.dataframe(S["plan"], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------
with tabs[4]:
    st.subheader("Calibración: pruebas antes de datos reales")

    st.markdown("""
**Secuencia recomendada**

1. DUMMY → comprobar que la aplicación responde.
2. CALIBRACIÓN → probar consenso, polarización, fragmentación, resistencia y red.
3. SINTÉTICO PURO → stress test de tamaño.
4. SINTÉTICO COHERENTE → verificar que el comportamiento conserve patrones estadísticos del CSV real.
5. REAL + SHP → producción analítica.

Esto permite separar **error de software**, **error de datos** y **error de modelo**.
""")

    cal_summary = pd.DataFrame([
        {"test": "balance", "objetivo": "distribución sin polarización extrema"},
        {"test": "polarizacion", "objetivo": "dos polos separados"},
        {"test": "fragmentacion", "objetivo": "múltiples estados"},
        {"test": "consenso", "objetivo": "convergencia"},
        {"test": "resistencia_alta", "objetivo": "inercia institucional"},
        {"test": "red_hub", "objetivo": "dependencia de hubs"},
    ])
    st.dataframe(cal_summary, use_container_width=True, hide_index=True)

    st.markdown("#### Métricas de validación")
    validation = {
        "n": len(adf),
        "opinion_mean": float(adf.opinion.mean()),
        "opinion_std": float(adf.opinion.std()),
        "gini": float(model.compute_gini()),
        "network_density": float(net.get("density", 0)),
        "network_gini": float(net.get("gini_influence", 0)),
        "seed": int(seed),
    }
    st.json(validation)

    if S.get("real_reference") is not None:
        cmp = CalibrationEngine.compare(S["real_reference"], S["df"])
        score = CalibrationEngine.score(S["real_reference"], S["df"])
        st.markdown("#### Calibración contra CSV real de referencia")
        st.metric("Score de similitud distributiva", f"{score:.1f}/100" if score is not None else "—")
        st.dataframe(cmp, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------
with tabs[5]:
    st.subheader("🎯 Question Engine · respuesta calculada al cliente")
    question = st.selectbox("Pregunta", QuestionEngine.CATALOG)
    answer = QuestionEngine.answer(
        question, adf, terr, net,
        budget_df=budget_df,
        brig_report=S["brig_report"],
        model=model,
    )
    st.success(answer)

    st.markdown("#### Indicadores de soporte")
    st.dataframe(terr, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
with tabs[6]:
    st.subheader("📤 Export reproducible")

    config = {
        "version": "SITER-CAE-v5.6-MESA-WORLD",
        "experiment_id": S["experiment_id"],
        "seed": int(seed),
        "mode": mode,
        "behavior": behavior_name,
        "n_units": len(adf),
        "budget": budget,
        "fixed_brigada": fixed_brigada,
        "hour_cost": hour_cost,
        "hours_per_brigada": hours_per_brigada,
    }

    payload = {
        "metadata": {
            **config,
            "output_hash": S["hash"],
            "data_meta": S["meta"],
            "governance": {
                "personal_data": False,
                "aggregated": True,
                "synthetic_supported": True,
                "geography": "CDMX",
            },
        },
        "model": model.metrics(),
        "territorial_fields": terr.to_dict("records"),
        "network": net,
        "budget": budget_df.to_dict("records"),
        "brigadistas": S["brig_report"].to_dict("records"),
        "history": model.history,
    }

    st.code(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))

    st.download_button(
        "⬇️ JSON completo",
        data=json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        file_name=f"{S['experiment_id']}.json",
        mime="application/json",
    )

    st.download_button(
        "⬇️ CSV territorial",
        data=terr.to_csv(index=False).encode("utf-8"),
        file_name=f"{S['experiment_id']}_territorial.csv",
        mime="text/csv",
    )

    if not S["brig_report"].empty:
        st.download_button(
            "⬇️ CSV brigadistas",
            data=S["brig_report"].to_csv(index=False).encode("utf-8"),
            file_name=f"{S['experiment_id']}_brigadistas.csv",
            mime="text/csv",
        )

    st.markdown("""
### Trazabilidad

`seed → datos → modelo → simulación → indicadores → presupuesto/brigadas → respuesta → hash`

El hash identifica la configuración/salida exportada; el seed permite repetir el
experimento cuando la misma fuente de datos y configuración estén disponibles.
""")

st.sidebar.markdown("---")
st.sidebar.caption("SITER-CAE v5.6 · Mesa · GIS · Presupuesto · Campo/GPS · Calibración · Sin PII")
