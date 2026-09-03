"""
SITER-CDMX v5.5 UNIFICADA
=========================
Todo junto:
- Motor Mesa (estilo NetLogo) + Behaviors intercambiables
- 5 modos de datos: real (CSV) | dummy | coherent | pure | calib
- Mapa + análisis estadísticos + presupuesto + brigadistas
- SPOF + funciones de respuesta al cliente + export
- Sin geopandas obligatorio (Streamlit Cloud compatible)

Ejecutar:
    pip install -r requirements.txt
    streamlit run app_siter_cdmx_v55.py
"""
from __future__ import annotations

import hashlib, json, uuid
from collections import defaultdict, Counter
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st

from mesa import Agent, Model

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
# CONSTANTES CDMX
# =============================================================================
ALCALDIAS = [
    "ALVARO OBREGON","AZCAPOTZALCO","BENITO JUAREZ","COYOACAN",
    "CUAJIMALPA DE MORELOS","CUAUHTEMOC","GUSTAVO A MADERO",
    "IZTACALCO","IZTAPALAPA","LA MAGDALENA CONTRERAS",
    "MIGUEL HIDALGO","MILPA ALTA","TLAHUAC","TLALPAN",
    "VENUSTIANO CARRANZA","XOCHIMILCO"
]
ALCALDIA_COORDS = {
    "CUAUHTEMOC":(19.4326,-99.1332),"BENITO JUAREZ":(19.3984,-99.1576),
    "MIGUEL HIDALGO":(19.4285,-99.2000),"COYOACAN":(19.3467,-99.1617),
    "IZTAPALAPA":(19.3550,-99.0620),"GUSTAVO A MADERO":(19.4900,-99.1100),
    "ALVARO OBREGON":(19.3580,-99.2270),"TLALPAN":(19.2880,-99.1670),
    "XOCHIMILCO":(19.2630,-99.1040),"VENUSTIANO CARRANZA":(19.4200,-99.1000),
    "AZCAPOTZALCO":(19.4870,-99.1860),"IZTACALCO":(19.3950,-99.0980),
    "CUAJIMALPA DE MORELOS":(19.3570,-99.2900),"LA MAGDALENA CONTRERAS":(19.3200,-99.2400),
    "TLAHUAC":(19.2700,-99.0050),"MILPA ALTA":(19.1920,-99.0230),
}
COLOR_INT = {"SIMPATIZANTE":"#2ecc71","OPOSITOR":"#e74c3c","INDECISO":"#95a5a6"}
CAMPO_COLOR = {"CONSOLIDACION":"#2ecc71","DISPUTA_ABIERTA":"#e74c3c","CONTENCION":"#3498db"}

def sha256(v):
    p = v if isinstance(v,str) else json.dumps(v,sort_keys=True,default=str)
    return hashlib.sha256(p.encode()).hexdigest()

def campo_de(simpat, indec):
    if simpat >= 0.5: return "CONSOLIDACION"
    if indec >= 0.35: return "DISPUTA_ABIERTA"
    return "CONTENCION"

# =============================================================================
# BEHAVIORS (Mesa)
# =============================================================================
class Behavior:
    name="base"
    def __init__(self, params=None): self.params=params or {}
    def step_agent(self, agent): raise NotImplementedError

class VoterBehavior(Behavior):
    name="voter"
    def step_agent(self, agent):
        ns=list(agent.get_neighbors())
        if not ns: return
        other=agent.random.choice(ns)
        prob=min(0.9, other.influencia*self.params.get("beta",1.2))
        if agent.es_broker: prob*=0.3
        if agent.spin!=other.spin and agent.random.random()<prob:
            agent.spin=other.spin
            agent.opinion=0.7 if agent.spin==1 else (-0.7 if agent.spin==-1 else 0.0)
            agent.intencion={1:"SIMPATIZANTE",-1:"OPOSITOR",0:"INDECISO"}[agent.spin]

class DeffuantBehavior(Behavior):
    name="deffuant"
    def step_agent(self, agent):
        ns=list(agent.get_neighbors())
        if not ns: return
        other=agent.random.choice(ns)
        mu,eps,eps_rep=self.params.get("mu",0.3),self.params.get("epsilon",0.4),self.params.get("epsilon_rep",0.8)
        d=abs(agent.opinion-other.opinion)
        if d<eps:
            delta=mu*(other.opinion-agent.opinion)
            agent.opinion=float(np.clip(agent.opinion+delta,-1,1))
            other.opinion=float(np.clip(other.opinion-delta,-1,1))
        elif d>eps_rep:
            delta=0.2*mu*(other.opinion-agent.opinion)
            agent.opinion=float(np.clip(agent.opinion-delta,-1,1))
            other.opinion=float(np.clip(other.opinion+delta,-1,1))
        agent.spin=1 if agent.opinion>0.25 else (-1 if agent.opinion<-0.25 else 0)
        agent.intencion={1:"SIMPATIZANTE",-1:"OPOSITOR",0:"INDECISO"}[agent.spin]

class ABMSAFBehavior(Behavior):
    name="abm_saf"
    def step_agent(self, agent):
        ns=list(agent.get_neighbors())
        if not ns: return
        if agent.fatiga>0.8 and agent.random.random()<0.7: return
        # presupuesto
        if agent.model.presupuesto_activo:
            if agent.model.dinero<=0 or agent.model.horas<=0: return
        for other in ns:
            prob=agent.influencia*0.55*self.params.get("beta",1.2)
            if agent.es_broker: prob*=1.45
            if agent.random.random()<min(prob,0.75):
                d=abs(agent.opinion-other.opinion)
                if d<0.40:
                    other.opinion=float(np.clip(other.opinion+0.3*(agent.opinion-other.opinion),-1,1))
                if abs(agent.spin-other.spin)>=1 and agent.spin!=0:
                    other.spin=agent.spin
                    other.opinion=0.7 if other.spin==1 else -0.7
                    other.intencion={1:"SIMPATIZANTE",-1:"OPOSITOR",0:"INDECISO"}[other.spin]
                    if agent.model.presupuesto_activo:
                        costo=5.0 if agent.es_broker else float(agent.random.uniform(10,40))
                        agent.model.dinero-=costo*0.1
                        agent.model.horas-=1
                agent.fatiga=min(1.0,agent.fatiga+0.05)
        agent.fatiga*=0.95

BEHAVIORS={"voter":VoterBehavior,"deffuant":DeffuantBehavior,"abm_saf":ABMSAFBehavior}

# =============================================================================
# AGENTE + MODELO MESA
# =============================================================================
class SeccionAgent(Agent):
    def __init__(self, model, alcaldia="CUAUHTEMOC", seccion="0001",
                 lat=19.43, lon=-99.13, opinion=0.0, capital_social=0.5,
                 influencia=0.5, es_broker=False, **kw):
        super().__init__(model)
        self.alcaldia=alcaldia; self.seccion=str(seccion)
        self.lat=float(lat); self.lon=float(lon)
        self.opinion=float(opinion)
        self.spin=1 if self.opinion>0.25 else (-1 if self.opinion<-0.25 else 0)
        self.intencion={1:"SIMPATIZANTE",-1:"OPOSITOR",0:"INDECISO"}[self.spin]
        self.capital_social=capital_social; self.influencia=influencia
        self.fatiga=0.0; self.es_broker=es_broker; self._neighbors=[]
    def get_neighbors(self): return self._neighbors
    def step(self): self.model.behavior.step_agent(self)

class SITERModel(Model):
    def __init__(self, df, adj, behavior="voter", seed=42, presupuesto=None, **params):
        super().__init__(seed=seed)
        self._seed=seed; self.adj=adj; self.agents_list=[]
        self.behavior=BEHAVIORS.get(behavior,VoterBehavior)(params)
        self.behavior_name=behavior
        self.presupuesto_activo=presupuesto is not None
        self.dinero=float((presupuesto or {}).get("dinero",999999))
        self.horas=float((presupuesto or {}).get("horas",999999))
        for i,row in df.reset_index(drop=True).iterrows():
            a=SeccionAgent(model=self,
                alcaldia=row.get("alcaldia","CUAUHTEMOC"),
                seccion=row.get("seccion",row.get("territorial_unit_id",f"{i:04d}")),
                lat=float(row.get("lat",19.43)), lon=float(row.get("lon",-99.13)),
                opinion=float(row.get("opinion_continua",0.0)),
                capital_social=float(row.get("capital_social",0.5)),
                influencia=float(row.get("influencia_SAF",row.get("habilidades_sociales",0.5))),
                es_broker=bool(row.get("es_broker_insertado",False)))
            a._idx=i; self.agents_list.append(a)
        for a in self.agents_list:
            a._neighbors=[self.agents_list[j] for j in adj.get(a._idx,[]) if j<len(self.agents_list)]
        self._history=[]; self._collect()

    def _collect(self):
        self._history.append({
            "step":len(self._history),
            "SIMPATIZANTE":self.count_spin(1),"OPOSITOR":self.count_spin(-1),"INDECISO":self.count_spin(0),
            "dinero":self.dinero,"horas":self.horas
        })
    def count_spin(self,v):
        return sum(1 for a in self.agents_list if a.spin==v)/max(len(self.agents_list),1)
    def step(self):
        order=list(self.agents_list); self.random.shuffle(order)
        for a in order: a.step()
        self._collect()
    def get_df(self):
        return pd.DataFrame([{
            "agent_id":f"{'BRK' if a.es_broker else 'SEC'}-{a._idx}",
            "alcaldia":a.alcaldia,"seccion":a.seccion,"lat":a.lat,"lon":a.lon,
            "opinion":a.opinion,"spin":a.spin,"intencion":a.intencion,
            "influencia":a.influencia,"es_broker":a.es_broker,"fatiga":a.fatiga
        } for a in self.agents_list])
    def insert_broker(self, lat, lon, alcaldia, capital=0.9, intencion="SIMPATIZANTE", grado=12):
        idx=len(self.agents_list)
        op=0.75 if intencion=="SIMPATIZANTE" else (-0.75 if intencion=="OPOSITOR" else 0.0)
        b=SeccionAgent(model=self,alcaldia=alcaldia,seccion=f"BRK{idx}",lat=lat,lon=lon,
                       opinion=op,capital_social=capital,influencia=0.95,es_broker=True)
        b._idx=idx
        cands=sorted(self.agents_list,key=lambda x:x.influencia,reverse=True)[:grado]
        b._neighbors=cands
        for c in cands:
            if b not in c._neighbors: c._neighbors.append(b)
        self.agents_list.append(b); return b

# =============================================================================
# DATA PROVIDER — 5 MODOS
# =============================================================================
class DataProvider:
    def __init__(self, mode="synth_pure", seed=42):
        self.mode=mode; self.seed=seed; self.rng=np.random.default_rng(seed)

    def load(self, n=300, secciones_csv=None, electoral_csv=None, socio_csv=None, **kw):
        if self.mode=="real":
            return self._real(secciones_csv, electoral_csv, socio_csv)
        if self.mode=="dummy": return self._pure(min(n,48))
        if self.mode=="synth_coherent": return self._coherent(n)
        if self.mode=="synth_calib": return self._calib(n, kw.get("scenario","polarizacion_alta"))
        return self._pure(n)

    def _real(self, sec_csv, elec_csv, socio_csv):
        if sec_csv is None:
            raise ValueError("Modo real requiere CSV de secciones")
        df=pd.read_csv(sec_csv) if not hasattr(sec_csv,"read") else pd.read_csv(sec_csv)
        df.columns=[c.strip().lower() for c in df.columns]
        df["seccion"]=df["seccion"].astype(str).str.zfill(4)
        df["alcaldia"]=df["alcaldia"].str.upper().str.strip()
        if "lat" not in df.columns:
            df["lat"]=df["alcaldia"].map(lambda a: ALCALDIA_COORDS.get(a,(19.43,-99.13))[0]+self.rng.normal(0,0.008))
            df["lon"]=df["alcaldia"].map(lambda a: ALCALDIA_COORDS.get(a,(19.43,-99.13))[1]+self.rng.normal(0,0.008))
        if elec_csv is not None:
            elec=pd.read_csv(elec_csv) if not hasattr(elec_csv,"read") else pd.read_csv(elec_csv)
            elec.columns=[c.strip().lower() for c in elec.columns]
            elec["seccion"]=elec["seccion"].astype(str).str.zfill(4)
            if "votos" in elec.columns and "partido_o_coalicion" in elec.columns:
                piv=elec.pivot_table(index="seccion",columns="partido_o_coalicion",values="votos",aggfunc="sum",fill_value=0)
                total=piv.sum(axis=1).replace(0,np.nan)
                share=(piv.max(axis=1)/total).fillna(0.33)
                df=df.merge(share.rename("share_max"),left_on="seccion",right_index=True,how="left")
                df["opinion_continua"]=(df["share_max"].fillna(0.33)*2-1).clip(-1,1)
            else:
                df["opinion_continua"]=0.0
        else:
            df["opinion_continua"]=0.0
        if socio_csv is not None:
            socio=pd.read_csv(socio_csv) if not hasattr(socio_csv,"read") else pd.read_csv(socio_csv)
            socio.columns=[c.strip().lower() for c in socio.columns]
            socio["seccion"]=socio["seccion"].astype(str).str.zfill(4)
            df=df.merge(socio,on="seccion",how="left",suffixes=("", "_s"))
        df["capital_social"]=df.get("escolaridad_prom",pd.Series([10]*len(df))).fillna(10)/15
        df["influencia_SAF"]=df["capital_social"].clip(0,1)*0.7+0.2
        df["es_broker_insertado"]=False
        meta={"mode":"real","n_nodes":len(df),"hash":sha256(str(df["seccion"].tolist()[:20]))}
        return df, self._adj(df), meta

    def _pure(self,n):
        rows=[]
        for i in range(n):
            alc=str(self.rng.choice(ALCALDIAS)); lat,lon=ALCALDIA_COORDS[alc]
            rows.append({"seccion":f"{i:04d}","alcaldia":alc,
                "lat":lat+self.rng.normal(0,0.012),"lon":lon+self.rng.normal(0,0.012),
                "opinion_continua":float(self.rng.uniform(-0.8,0.8)),
                "capital_social":float(self.rng.uniform(0.3,0.8)),
                "influencia_SAF":float(self.rng.uniform(0.2,0.9)),"es_broker_insertado":False})
        df=pd.DataFrame(rows)
        return df,self._adj(df),{"mode":"synth_pure","n_nodes":len(df),"hash":sha256(f"pure-{n}-{self.seed}")}

    def _coherent(self,n):
        rows=[]
        for i in range(n):
            alc=str(self.rng.choice(ALCALDIAS)); lat,lon=ALCALDIA_COORDS[alc]
            share=float(np.clip(self.rng.normal(0.37,0.14),0.05,0.95))
            rows.append({"seccion":f"C{i:04d}","alcaldia":alc,
                "lat":lat+self.rng.normal(0,0.01),"lon":lon+self.rng.normal(0,0.01),
                "opinion_continua":2*share-1,
                "capital_social":float(np.clip(1-self.rng.normal(0.4,0.15),0.2,0.9)),
                "influencia_SAF":float(self.rng.uniform(0.3,0.85)),"es_broker_insertado":False})
        df=pd.DataFrame(rows)
        return df,self._adj(df,0.07,0.02),{"mode":"synth_coherent","n_nodes":len(df),"hash":sha256(f"coh-{n}")}

    def _calib(self,n,scenario="polarizacion_alta"):
        half=n//2; rows=[]
        for i in range(n):
            b=0 if i<half else 1; alc=ALCALDIAS[b*4+(i%4)]; lat,lon=ALCALDIA_COORDS[alc]
            op=0.75 if b==0 else -0.75
            op=float(np.clip(op+self.rng.normal(0,0.08),-1,1))
            rows.append({"seccion":f"P{i:04d}","alcaldia":alc,
                "lat":lat+self.rng.normal(0,0.008),"lon":lon+self.rng.normal(0,0.008),
                "opinion_continua":op,"capital_social":0.6,"influencia_SAF":0.5,"es_broker_insertado":False})
        df=pd.DataFrame(rows)
        return df,self._adj(df,0.12,0.005),{"mode":"synth_calib","scenario":scenario,"n_nodes":len(df),"hash":sha256(f"cal-{scenario}-{n}")}

    def _adj(self,df,p_intra=0.08,p_inter=0.02):
        adj=defaultdict(list)
        for alc in df["alcaldia"].unique():
            idxs=df.index[df["alcaldia"]==alc].tolist()
            for a in range(len(idxs)):
                for b in range(a+1,len(idxs)):
                    if self.rng.random()<p_intra:
                        x,y=idxs[a],idxs[b]; adj[x].append(y); adj[y].append(x)
        alcs=list(df["alcaldia"].unique())
        for i,a1 in enumerate(alcs):
            for a2 in alcs[i+1:]:
                i1=df.index[df["alcaldia"]==a1].tolist()[:4]
                i2=df.index[df["alcaldia"]==a2].tolist()[:4]
                for x in i1:
                    for y in i2:
                        if self.rng.random()<p_inter:
                            adj[x].append(y); adj[y].append(x)
        return dict(adj)

# =============================================================================
# ANÁLISIS ESTADÍSTICOS + SPOF + BRIGADAS + RESPUESTA AL CLIENTE
# =============================================================================
class Analisis:
    @staticmethod
    def territoriales(df):
        res=[]
        for alc in df["alcaldia"].unique():
            sub=df[df["alcaldia"]==alc]
            counts=sub["intencion"].value_counts(normalize=True)
            simpat=(sub["intencion"]=="SIMPATIZANTE").mean()
            opos=(sub["intencion"]=="OPOSITOR").mean()
            indec=(sub["intencion"]=="INDECISO").mean()
            ent=-sum(p*np.log(p) for p in counts if p>0)
            res.append({"alcaldia":alc,"n":len(sub),"simpat_pct":round(simpat*100,1),
                        "opos_pct":round(opos*100,1),"indec_pct":round(indec*100,1),
                        "polarizacion":round(1-abs(simpat-opos),3),"entropia":round(ent,3),
                        "campo":campo_de(simpat,indec),
                        "influencia_prom":round(sub["influencia"].mean(),3)})
        return pd.DataFrame(res)

    @staticmethod
    def spof(df, top_k=8):
        # SPOF simplificado por influencia y grado aproximado
        top=df.nlargest(top_k,"influencia")
        rows=[]
        for _,r in top.iterrows():
            rows.append({"agent_id":r["agent_id"],"alcaldia":r["alcaldia"],
                         "influencia":round(r["influencia"],3),
                         "riesgo_spof":round(r["influencia"]*1.2,3)})
        return pd.DataFrame(rows).sort_values("riesgo_spof",ascending=False)

    @staticmethod
    def brigadas(n_brig=4, pasos=8, seed=42):
        rng=np.random.default_rng(seed); rutas=[]
        for b in range(n_brig):
            alc=str(rng.choice(ALCALDIAS)); lat,lon=ALCALDIA_COORDS[alc]
            for step in range(pasos+1):
                if step>0:
                    alc=str(rng.choice(ALCALDIAS)); lat,lon=ALCALDIA_COORDS[alc]
                    lat+=rng.normal(0,0.01); lon+=rng.normal(0,0.01)
                rutas.append({"brigada":f"B-{b}","step":step,"lat":lat,"lon":lon,"alcaldia":alc})
        return pd.DataFrame(rutas)

    @staticmethod
    def preguntas_cliente(df, df_terr):
        """Funciones de respuesta rápida al cliente (calculadas)."""
        filas=[]
        def add(cat,q,a):
            filas.append({"categoria":cat,"pregunta":q,"respuesta":str(a),"evidencia":"SIM-CALC"})
        add("Global","¿Cuántas secciones/agentes hay?",len(df))
        add("Global","¿Cuántas alcaldías?",df["alcaldia"].nunique())
        if not df_terr.empty:
            add("Global","¿Alcaldía con mayor simpatía?",f"{df_terr.loc[df_terr['simpat_pct'].idxmax(),'alcaldia']} ({df_terr['simpat_pct'].max()}%)")
            add("Global","¿Alcaldía más polarizada?",f"{df_terr.loc[df_terr['polarizacion'].idxmax(),'alcaldia']}")
            add("Global","¿Campo dominante?",df_terr["campo"].value_counts().idxmax())
        add("Global","¿Agente más influyente?",df.loc[df["influencia"].idxmax(),"agent_id"] if len(df) else "—")
        for _,r in df_terr.iterrows():
            add(f"Alcaldía {r['alcaldia']}",f"[{r['alcaldia']}] Campo",r["campo"])
            add(f"Alcaldía {r['alcaldia']}",f"[{r['alcaldia']}] Simpatía %",r["simpat_pct"])
        return pd.DataFrame(filas)

# =============================================================================
# UI STREAMLIT
# =============================================================================
st.set_page_config(page_title="SITER-CDMX v5.5", page_icon="🧠", layout="wide")
st.title("🧠 SITER-CDMX v5.5 — Unificada")
st.caption("Mesa · NetLogo-style · 5 modos de datos · Mapa · Análisis · Presupuesto · Brigadas · Respuesta al cliente")

if "s55" not in st.session_state:
    st.session_state.s55={"model":None,"df":pd.DataFrame(),"adj":{},"tray":[],"tick":0,
                          "meta":{},"rutas":pd.DataFrame(),"presupuesto":None}
S=st.session_state.s55

# ----- SIDEBAR -----
st.sidebar.header("⚙️ Setup")
data_mode=st.sidebar.selectbox("Modo de datos",["synth_pure","dummy","synth_coherent","synth_calib","real"])
seed=st.sidebar.number_input("Seed",1,99999,42)
n=st.sidebar.slider("N secciones",50,800,300,50)
behavior=st.sidebar.selectbox("Behavior",list(BEHAVIORS.keys()))
beta=st.sidebar.slider("Beta",0.3,2.5,1.2,0.1)
steps_go=st.sidebar.slider("Pasos por Go",1,20,5)

# Presupuesto
usar_pres=st.sidebar.checkbox("Activar presupuesto limitado")
dinero_p=st.sidebar.number_input("Dinero $",500,20000,5000,500) if usar_pres else 999999
horas_p=st.sidebar.number_input("Horas",20,500,100,10) if usar_pres else 999999

# Real files
sec_file=elec_file=socio_file=None
if data_mode=="real":
    st.sidebar.markdown("**CSV reales**")
    sec_file=st.sidebar.file_uploader("Secciones (obligatorio)",type=["csv"])
    elec_file=st.sidebar.file_uploader("Electoral (opcional)",type=["csv"])
    socio_file=st.sidebar.file_uploader("Socio (opcional)",type=["csv"])

c1,c2,c3=st.sidebar.columns(3)
with c1:
    if st.button("🔄 Setup",use_container_width=True):
        try:
            prov=DataProvider(mode=data_mode,seed=int(seed))
            df,adj,meta=prov.load(n=n,secciones_csv=sec_file,electoral_csv=elec_file,socio_csv=socio_file)
            pres={"dinero":dinero_p,"horas":horas_p} if usar_pres else None
            model=SITERModel(df,adj,behavior=behavior,seed=int(seed),presupuesto=pres,beta=beta)
            S.update({"model":model,"df":df,"adj":adj,"tray":list(model._history),
                      "tick":0,"meta":meta,"presupuesto":pres})
            st.success(f"Setup · {meta.get('mode')} · {meta.get('n_nodes')} nodos")
        except Exception as e:
            st.error(str(e))
with c2:
    if st.button("▶️ Go",use_container_width=True):
        if S["model"] is None: st.warning("Setup primero")
        else:
            for _ in range(steps_go):
                S["model"].step(); S["tick"]+=1
            S["tray"]=list(S["model"]._history); st.rerun()
with c3:
    if st.button("⏹ Reset",use_container_width=True):
        S["model"]=None; S["tray"]=[]; S["tick"]=0; st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Broker")
b_alc=st.sidebar.selectbox("Alcaldía",ALCALDIAS)
b_int=st.sidebar.selectbox("Intención",["SIMPATIZANTE","OPOSITOR","INDECISO"])
if st.sidebar.button("➕ Insertar Broker"):
    if S["model"] is None: st.sidebar.warning("Setup primero")
    else:
        lat,lon=ALCALDIA_COORDS[b_alc]
        S["model"].insert_broker(lat,lon,b_alc,intencion=b_int)
        st.sidebar.success("Broker OK"); st.rerun()

# ----- MAIN -----
if S["model"] is None:
    st.info("➡️ **Setup** para generar el universo. Usa modo `real` + CSV de plantillas para datos base.")
    st.markdown("""
    **5 modos de datos**
    - `real` → CSV secciones (+ electoral + socio)
    - `dummy` → mínimo para pruebas
    - `synth_coherent` → sintético realista CDMX
    - `synth_pure` → sintético libre
    - `synth_calib` → escenarios de calibración

    **Incluye:** mapa, análisis por alcaldía, presupuesto, brigadas, SPOF, preguntas al cliente, export.
    """)
else:
    model:SITERModel=S["model"]
    adf=model.get_df()
    terr=Analisis.territoriales(adf)

    # Monitors
    m1,m2,m3,m4,m5,m6=st.columns(6)
    m1.metric("Tick",S["tick"])
    m2.metric("Simpatizantes",f"{model.count_spin(1):.1%}")
    m3.metric("Opositores",f"{model.count_spin(-1):.1%}")
    m4.metric("Indecisos",f"{model.count_spin(0):.1%}")
    m5.metric("Brokers",sum(1 for a in model.agents_list if a.es_broker))
    if S.get("presupuesto"):
        m6.metric("Presupuesto",f"${model.dinero:.0f} / {model.horas:.0f}h")
    else:
        m6.metric("Modo",S["meta"].get("mode","—"))

    tabs=st.tabs(["🗺️ Mapa","📊 Análisis","💰 Presupuesto & Brigadas","🎯 SPOF & Cliente","📤 Export"])

    with tabs[0]:
        if HAS_PLOTLY:
            fig=px.scatter(adf,x="lon",y="lat",color="intencion",size="influencia",
                           hover_name="agent_id",color_discrete_map=COLOR_INT,
                           hover_data=["alcaldia","seccion","opinion"],
                           title=f"CDMX · {model.behavior_name} · Tick {S['tick']}",size_max=24)
            br=adf[adf["es_broker"]]
            if not br.empty:
                fig.add_trace(go.Scatter(x=br["lon"],y=br["lat"],mode="markers",
                    marker=dict(symbol="star",size=18,color="gold",line=dict(width=1,color="black")),name="Broker"))
            fig.update_layout(height=560,template="plotly_dark")
            st.plotly_chart(fig,use_container_width=True)
        st.dataframe(terr,use_container_width=True)

    with tabs[1]:
        st.subheader("Indicadores por Alcaldía")
        st.dataframe(terr,use_container_width=True)
        if HAS_PLOTLY and not terr.empty:
            st.plotly_chart(px.bar(terr,x="alcaldia",y="simpat_pct",color="campo",
                                   color_discrete_map=CAMPO_COLOR,title="Simpatía % por Alcaldía"),
                            use_container_width=True)
        if S["tray"]:
            tdf=pd.DataFrame(S["tray"])
            st.line_chart(tdf.set_index("step")[["SIMPATIZANTE","OPOSITOR","INDECISO"]])

    with tabs[2]:
        st.subheader("Presupuesto")
        if S.get("presupuesto") and S["tray"]:
            tdf=pd.DataFrame(S["tray"])
            if "dinero" in tdf.columns:
                st.line_chart(tdf.set_index("step")[["dinero","horas"]])
        else:
            st.info("Activa presupuesto en el sidebar y haz Setup.")
        st.subheader("Brigadistas")
        nb=st.slider("N brigadas",1,8,4)
        if st.button("Generar rutas de brigadas"):
            S["rutas"]=Analisis.brigadas(nb,8,int(seed))
        if not S["rutas"].empty:
            if HAS_PLOTLY:
                fig=px.line(S["rutas"],x="lon",y="lat",color="brigada",markers=True,
                            hover_data=["step","alcaldia"],title="Rutas de brigadas")
                fig.update_layout(height=450,template="plotly_dark")
                st.plotly_chart(fig,use_container_width=True)
            st.dataframe(S["rutas"],use_container_width=True)

    with tabs[3]:
        st.subheader("SPOF (puntos críticos)")
        st.dataframe(Analisis.spof(adf),use_container_width=True)
        st.subheader("Respuesta al cliente (preguntas calculadas)")
        preg=Analisis.preguntas_cliente(adf,terr)
        st.dataframe(preg,use_container_width=True)

    with tabs[4]:
        st.subheader("Export reproducible")
        payload={
            "meta":{"version":"siter-cdmx-v5.5","seed":S["model"]._seed,
                    "behavior":model.behavior_name,"data":S["meta"],
                    "tick":S["tick"],"hash":sha256(S["meta"])},
            "territorial":terr.to_dict(orient="records"),
            "trayectoria":S["tray"],
            "governance":{"public":True,"personal":False,"agregado":True}
        }
        st.json({"hash":payload["meta"]["hash"][:16]+"…","n_alcaldias":len(terr),"ticks":S["tick"]})
        st.download_button("⬇️ Descargar JSON",data=json.dumps(payload,ensure_ascii=False,indent=2,default=str),
                           file_name="siter_cdmx_export.json",mime="application/json")

st.sidebar.markdown("---")
st.sidebar.caption("SITER-CDMX v5.5 unificada · Mesa · Sin PII")
