
"""
SITER-CAE v4.0 — LABORATORIO SOCIOFÍSICO + ABM + SAF + NETLOGO
Aplicación monolítica para Streamlit.

El catálogo de referencia contiene 77 preguntas y define v4.0 como la
extensión de v3.3 con recursos limitados, adversario/contra-actor,
bounded-confidence + repulsión, optimización, SPOF e inercia.
El universo operativo de esta implementación es únicamente CDMX:
CDMX -> Alcaldía -> UTM -> Sección -> Manzana.

Todos los agentes son sintéticos y las conclusiones son agregadas.
"""

from __future__ import annotations
import hashlib, json, math, os, shutil, subprocess, tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.express as px

st.set_page_config(page_title="SITER-CAE v4.0", page_icon="🧬", layout="wide")

ALCALDIAS = [
    "Álvaro Obregón","Azcapotzalco","Benito Juárez","Coyoacán",
    "Cuajimalpa de Morelos","Cuauhtémoc","Gustavo A. Madero","Iztacalco",
    "Iztapalapa","La Magdalena Contreras","Miguel Hidalgo","Milpa Alta",
    "Tláhuac","Tlalpan","Venustiano Carranza","Xochimilco"
]
JERARQUIA = ["CDMX","ALCALDIA","UTM","SECCION","MANZANA"]

QUESTIONS = [
"P1 ¿Dónde está más polarizado?","P2 ¿Dónde hay más simpatizantes?",
"P3 ¿Qué territorio tiene más temperatura sintética?","P4 ¿Dónde se moviliza más la gente?",
"P5 ¿Qué territorio tiene más habilidades promedio?","P6 ¿Qué campo está en DISPUTA_ABIERTA?",
"P7 ¿Qué campo está consolidado?","P8 ¿Dónde hay más conflicto interno?",
"P9 ¿Qué campo es más fragmentado?","P10 ¿Cuál tiene más institucionalización?",
"P11 ¿Cuál es dominante y su dominancia?","P12 ¿Qué tan conectada está mi red?",
"P13 ¿Mi red es de tribus o abierta?","P14 ¿Hay centralización peligrosa?",
"P15 ¿Qué tan desigual es la influencia?","P16 ¿Quiénes son mis centros SAF?",
"P17 ¿Mis centros son buenos o solo ruido?","P18 ¿Cuántos brokers tengo?",
"P19 ¿Top10% cuánta influencia concentra?","P20 ¿En cuántos pasos converge?",
"P21 ¿Quién emergió como centro en paso 3?","P22 ¿Cuántas conversiones causó cada centro?",
"P23 ¿Dónde está hotspot influencia?","P24 ¿Dónde está caliente polarización?",
"P25 ¿Mis brigadas saturan territorio?","P26 ¿Hasta dónde llega una onda de problema/rumor?",
"P27 ¿Qué centros están conectados?","P28 ¿Qué territorio más visitas brigada?",
"P29 ¿Qué pasa si aumento la intervención territorial?",
"P30 ¿Qué pasa si hay un shock severo en paso 3?",
"P31 ¿Qué pasa si aumento conectividad?","P32 ¿Con $500 vs $3000 cuánto cubro?",
"P33 ¿Qué pasa si se elimina el principal broker?",
"P34 ¿Intervención informativa vs intervención operativa?",
"P35 ¿Probabilidad de que una intervención mejore >5%?",
"P36 ¿Qué pasa si inserto un actor central de alta capacidad?",
"P37 ¿Qué pasa si aparece un contra-actor espejo?",
"P38 ¿Qué pasa si el actor insertado tiene habilidades bajas?",
"P39 ¿Dónde poner un actor puente: hub o territorio aislado?",
"P40 ¿Qué rasgos explican la capacidad SAF?","P41 ¿Cuántas conexiones necesita un actor puente?",
"P42 ¿Qué pasa si el arraigo territorial es bajo?","P43 ¿Cuánto cuesta cubrir todo?",
"P44 ¿Qué territorio tiene mejor retorno de intervención?",
"P45 ¿Qué territorios están más aislados y son caros de conectar?",
"P46 ¿Cuál es la intervención mínima para mejorar 10%?",
"P47 ¿Los activos top10% son óptimos?","P48 ¿Es reproducible?","P49 ¿Tiene PII?",
"P50 ¿Cómo exporto al motor central?","P51 ¿Cuál es el costo real de una intervención de campo?",
"P52 ¿Qué pasa cuando las brigadas se fatigan?","P53 ¿Cuándo se acaba el presupuesto?",
"P54 ¿Cómo afecta el desgaste de capital institucional?","P55 ¿Qué es epsilon de confianza?",
"P56 ¿Qué es echo chamber y repulsión?","P57 ¿Cómo mido polarización endógena?",
"P58 ¿Qué pasa si epsilon es muy bajo?","P59 ¿Qué pasa si pongo un contra-actor espejo?",
"P60 ¿Cuánto pierdo con contra-actor vs sin él?","P61 ¿Qué es una estrategia de flanqueo del adversario?",
"P62 ¿Qué es decapitación de red?","P63 ¿Cómo detecta el adversario un actor central?",
"P64 ¿Con $5000 cuál es la distribución óptima?","P65 ¿Qué territorio merece más actores puente vs horas de brigada?",
"P66 ¿Cómo evoluciona el score del optimizador?","P67 ¿Qué nodos son punto único de falla (SPOF)?",
"P68 ¿Si se elimina un nodo cuánto cae el sistema?","P69 ¿Mi red es resiliente?",
"P70 ¿Qué territorio tiene alta inercia institucional?","P71 ¿Qué barrera de adopción produce la inercia?",
"P72 ¿Dónde conviene trabajar con estructuras formales?","P73 ¿Qué es inercia normativa?",
"P74 ¿Cómo modelar fatiga de interacción?","P75 ¿Qué es el costo de polarizar?",
"P76 ¿Cómo reducir echo chamber?","P77 ¿Qué aporta v4.0 respecto de v3.3?"
]

def clip(x, a=0, b=1): return float(max(a, min(b, x)))

def entropy(p):
    p = np.asarray(p, dtype=float); p = p[p > 0]
    if len(p) < 2: return 0.0
    p = p / p.sum()
    return float(-(p*np.log(p)).sum()/np.log(len(p)))

def gini(x):
    x=np.sort(np.maximum(np.asarray(x,float),0))
    if len(x)==0 or x.sum()==0: return 0
    n=len(x); return float((2*np.arange(1,n+1)-n-1).dot(x)/(n*x.sum()))

def top_share(x, f):
    x=np.sort(np.asarray(x,float))[::-1]; k=max(1,int(math.ceil(len(x)*f)))
    return float(x[:k].sum()/max(x.sum(),1e-12))

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str,ensure_ascii=False).encode()).hexdigest()

@dataclass
class Agent:
    id: str
    territorio: str
    capital_social: float
    acceso_informacion: float
    influencia_liderazgo: float
    arraigo: float
    nivel_movilizacion: float
    desconfianza: float
    exposicion_problema: float
    opinion: float
    resistencia: float
    fatiga: float = 0.0

    @property
    def habilidades(self):
        return clip(.30*self.capital_social+.25*self.acceso_informacion+
                    .25*self.influencia_liderazgo+.10*self.arraigo+
                    .10*self.nivel_movilizacion-.15*self.desconfianza)
    @property
    def influencia(self):
        return clip(.60*self.habilidades+.40*self.nivel_movilizacion)

class Lab:
    def __init__(self,n=800,seed=42,p_intra=.05,p_inter=.01):
        self.n,self.seed,self.p_intra,self.p_inter=int(n),int(seed),p_intra,p_inter
        self.rng=np.random.default_rng(seed); self.generate()

    def generate(self):
        r=self.rng; A=[]
        for i in range(self.n):
            v=r.beta(2.2,2,7)
            A.append(Agent(
                f"SYN-{i+1:05d}",ALCALDIAS[i%16],*map(float,v),
                float(r.choice([-1,0,1],p=[.36,.28,.36])),
                float(r.beta(2,5))
            ))
        self.agents=A; self.build_network(); self.aggregate()
        self.experiment_id=f"SITER-{self.seed}-{digest(self.territories.to_dict('records'))[:12]}"

    def df(self):
        return pd.DataFrame([{**asdict(a),"habilidades":a.habilidades,
                              "influencia":a.influencia} for a in self.agents])

    def build_network(self):
        G=nx.Graph(); r=self.rng
        for a in self.agents: G.add_node(a.id,territorio=a.territorio)
        by={}
        for a in self.agents: by.setdefault(a.territorio,[]).append(a.id)
        for ids in by.values():
            for i in range(len(ids)):
                for j in range(i+1,len(ids)):
                    if r.random()<self.p_intra: G.add_edge(ids[i],ids[j],kind="intra")
        ids=list(G.nodes)
        for _ in range(max(1,int(self.n*3))):
            u,v=r.choice(ids,2,replace=False)
            if G.nodes[u]["territorio"]!=G.nodes[v]["territorio"] and r.random()<self.p_inter:
                G.add_edge(u,v,kind="inter")
        self.G=G
        deg=dict(G.degree())
        btw=nx.betweenness_centrality(G,k=min(100,len(G)),seed=self.seed) if len(G)>2 else {}
        eig=nx.eigenvector_centrality_numpy(G) if G.number_of_edges() else {x:0 for x in G}
        self.network=pd.DataFrame([{
            "agent_id":a.id,"territorio":a.territorio,"grado":deg[a.id],
            "betweenness":btw.get(a.id,0),"eigenvector":eig.get(a.id,0),
            "habilidades":a.habilidades,"influencia":a.influencia
        } for a in self.agents])

    def aggregate(self):
        d=self.df(); rows=[]
        for t,g in d.groupby("territorio"):
            p=np.array([(g.opinion<-.1).mean(),(g.opinion.abs()<=.1).mean(),
                        (g.opinion>.1).mean()])
            h=entropy(p); stab=clip(1-h); pol=clip(1-abs(p[2]-p[0]))
            res=float(g.resistencia.mean())
            rows.append({
                "territorio":t,"n":len(g),"simpat_pct":p[2],"opos_pct":p[0],
                "indeciso_pct":p[1],"polarizacion":pol,"entropia":h,
                "herfindahl":float((p*p).sum()),"conflicto":float(g.opinion.var()),
                "fragmentacion":int((g.opinion.round(1).value_counts()>=len(g)*.1).sum()),
                "estabilidad":stab,
                "institucionalizacion":clip(.6*stab+.4*(1-g.desconfianza.mean())),
                "campo":"CONSOLIDACION" if stab>=.67 else "DISPUTA_ABIERTA" if stab<.45 else "CONTENCION",
                "habilidades_prom":g.habilidades.mean(),"mov_prom":g.nivel_movilizacion.mean(),
                "temperatura":g.exposicion_problema.mean()*100,
                "resistencia_prom":res,"barrera_adopcion":clip(.7*res+.3),
                "desconf_prom":g.desconfianza.mean(),"influencia_prom":g.influencia.mean()
            })
        self.territories=pd.DataFrame(rows)

    def centers(self,k=20):
        x=self.network.copy()
        x["score_SAF"]=.6*x.habilidades+.4*x.grado/max(1,x.grado.max())
        return x.sort_values("score_SAF",ascending=False).head(k)

    def net_metrics(self):
        deg=np.array([d for _,d in self.G.degree()],float)
        return {
            "n_nodos":len(self.G),"n_aristas":self.G.number_of_edges(),
            "densidad":nx.density(self.G),"grado_prom":deg.mean(),
            "grado_max":deg.max(),"clustering":nx.average_clustering(self.G),
            "centralizacion":((deg.max()*len(deg)-deg.sum())/
                              max((len(deg)-1)*(len(deg)-2),1))
        }

    def abm(self,model="Deffuant-Weisbuch",steps=20,beta=.25,eps=.4,epsrep=.8,shock=None):
        backup=[asdict(a) for a in self.agents]; rngstate=self.rng.bit_generator.state
        H=[]; conv={a.id:0 for a in self.agents}; ids=list(range(len(self.agents)))
        for s in range(steps+1):
            if s==3 and shock:
                for a in self.agents:
                    if shock=="NEGATIVO": a.opinion=clip(a.opinion-.45,-1,1)
                    if shock=="POSITIVO": a.opinion=clip(a.opinion+.45,-1,1)
            if s:
                self.rng.shuffle(ids)
                for i in ids:
                    a=self.agents[i]; j=int(self.rng.integers(len(self.agents)))
                    b=self.agents[j]; diff=b.opinion-a.opinion
                    if model=="Voter": a.opinion=b.opinion if self.rng.random()<beta else a.opinion
                    elif model=="Majority": a.opinion+=beta*(np.sign(a.opinion+b.opinion)-a.opinion)
                    elif model=="q-Voter" and abs(diff)<=eps: a.opinion+=beta*diff
                    elif model in ("Sznajd","Deffuant-Weisbuch","Hegselmann-Krause"):
                        if abs(diff)<=eps:
                            a.opinion+=beta*diff; b.opinion-=beta*diff
                    elif model=="Ising": a.opinion=clip(a.opinion+.5*np.tanh(beta*diff),-1,1)
                    if model in ("Deffuant-Weisbuch","Hegselmann-Krause") and abs(diff)>epsrep:
                        a.opinion-=.08*np.sign(diff); b.opinion+=.08*np.sign(diff)
                    a.opinion=clip(a.opinion,-1,1); b.opinion=clip(b.opinion,-1,1)
                    if abs(a.opinion-b.opinion)<.2: conv[a.id]+=1
            H.append([a.opinion for a in self.agents])
        A=np.asarray(H)
        metric={"model":model,"steps":steps,"std_initial":float(A[0].std()),
                "std_final":float(A[-1].std()),"convergence_step":
                next((i for i,row in enumerate(A) if row.std()<.05),None),
                "polarizacion_end":float(A[-1].std()),
                "conversiones_totales":sum(conv.values()),
                "emergentes_p95":[k for k,v in conv.items() if v>np.percentile(list(conv.values()),95)]}
        self.agents=[Agent(**x) for x in backup]; self.rng=np.random.default_rng()
        self.rng.bit_generator.state=rngstate; self.build_network(); self.aggregate()
        return pd.DataFrame(A),metric,conv

    def mc(self,territorio,intervencion="BRIGADA",mag=.2,runs=200,shock=False,adversario=False):
        base=self.territories.copy(); r=np.random.default_rng(self.seed+9001); vals=[]
        for _ in range(runs):
            s=base.copy(); q=s.index[s.territorio==territorio][0]
            if intervencion=="BRIGADA": s.loc[q,"estabilidad"]=clip(s.loc[q,"estabilidad"]+.20*mag)
            if intervencion=="ACTOR_PUENTE": s.loc[q,"estabilidad"]=clip(s.loc[q,"estabilidad"]+.28*mag)
            if intervencion=="ESCUCHA": s.loc[q,"polarizacion"]=clip(s.loc[q,"polarizacion"]-.25*mag)
            if intervencion=="CONECTIVIDAD": s.loc[q,"estabilidad"]=clip(s.loc[q,"estabilidad"]+.10*mag)
            if shock: s.loc[q,"estabilidad"]=clip(s.loc[q,"estabilidad"]-.30)
            if adversario: s.loc[q,"estabilidad"]=clip(s.loc[q,"estabilidad"]-.08*mag)
            vals.append(float(s.estabilidad.mean()-base.estabilidad.mean()+r.normal(0,.015)))
        a=np.asarray(vals)
        return {"delta_mean":a.mean(),"p05":np.percentile(a,5),"p95":np.percentile(a,95),
                "prob_gt5":np.mean(a>.05),"prob_crisis":np.mean(a<-.05),"deltas":a}

    def optimize(self,budget=5000,hours=100,generations=20,population=20):
        r=np.random.default_rng(self.seed+777); n=16
        t=self.territories.set_index("territorio").reindex(ALCALDIAS)
        def make(): return r.integers(0,4,n),r.integers(0,25,n)
        def score(c):
            a,h=c; cost=a.sum()*500+h.sum()*25
            if cost>budget or h.sum()>hours:return -1e9
            return float((a*(1-t.estabilidad.values)+.02*h*t.mov_prom.values).sum())
        pop=[make() for _ in range(population)]; hist=[]
        for gen in range(generations):
            sc=np.array([score(x) for x in pop]); order=np.argsort(sc)[::-1]
            elite=[pop[i] for i in order[:max(2,population//4)]]
            hist.append({"generation":gen,"score":float(sc[order[0]])}); pop=elite[:]
            while len(pop)<population:
                x,y=elite[r.integers(len(elite))],elite[r.integers(len(elite))]
                cut=int(r.integers(1,n)); a=np.r_[x[0][:cut],y[0][cut:]].copy()
                h=np.r_[x[1][:cut],y[1][cut:]].copy()
                for z in (a,h):
                    for j in range(n):
                        if r.random()<.08:z[j]=max(0,z[j]+int(r.choice([-1,1])))
                pop.append((a,h))
        best=max(pop,key=score)
        out=pd.DataFrame({"territorio":ALCALDIAS,"actores_puente":best[0],"horas":best[1]})
        out["costo"]=out.actores_puente*500+out.horas*25
        return out,pd.DataFrame(hist)

    def spof(self):
        base=self.territories.estabilidad.mean(); rows=[]
        for _,x in self.centers(30).iterrows():
            G=self.G.copy(); G.remove_node(x.agent_id)
            comps=sorted([len(c) for c in nx.connected_components(G)],reverse=True)
            after=base*(comps[0]/len(self.G)) if comps else 0
            rows.append({"agent_id":x.agent_id,"territorio":x.territorio,
                         "delta":after-base,"SPOF":after-base<-.08,"grado":x.grado})
        return pd.DataFrame(rows).sort_values("delta")

    def export(self):
        t=self.territories.to_dict("records"); c=self.centers().to_dict("records")
        p={"territorial_fields":t,"indicadores_exhaustivos":{"red":self.net_metrics(),
           "saf":t,"territoriales":t,"rutas_agregadas":[],"ondas_agregadas":[]},
           "centros_SAF_agregados":c,
           "metadata":{"experiment_id":self.experiment_id,"seed":self.seed,
                       "data_origin":"CALCULATED_FROM_SYNTHETIC","pii":False,
                       "territory_scope":"CDMX","hierarchy":JERARQUIA}}
        p["metadata"]["output_hash"]=digest(p); return p

# ----------------------------- NetLogo --------------------------------
NETLOGO_TEMPLATE = """globals [starting-seed]
breed [actors actor]
actors-own [capacity influence territory-id]
to setup
 clear-all
 set starting-seed new-seed
 random-seed starting-seed
 create-actors 100 [
   setxy random-xcor random-ycor
   set capacity random-float 1
   set influence random-float 1
   set territory-id "CDMX"
   set shape "person"
 ]
 reset-ticks
end
to go
 ask actors [ rt random 40 lt random 40 fd (0.1 + capacity * 0.5) ]
 tick
end
to-report system-score
 report mean [influence] of actors
end
"""

def netlogo_exe():
    x=os.getenv("NETLOGO_EXECUTABLE")
    if x and Path(x).exists(): return x
    h=os.getenv("NETLOGO_HOME")
    if h:
        for p in [Path(h)/"bin"/"netlogo-headless.sh",Path(h)/"bin"/"netlogo-headless"]:
            if p.exists():return str(p)
    return shutil.which("netlogo-headless")

# ------------------------------ UI ------------------------------------
if "lab" not in st.session_state: st.session_state.lab=Lab()
lab=st.session_state.lab

st.sidebar.title("🧬 SITER-CAE v4.0")
st.sidebar.caption("Laboratorio científico reproducible")
seed=st.sidebar.number_input("Seed",0,9999999,int(lab.seed))
n=st.sidebar.slider("Agentes sintéticos",100,2500,int(lab.n),100)
pi=st.sidebar.slider("p_intra",.005,.15,float(lab.p_intra),.005)
pe=st.sidebar.slider("p_inter",.001,.05,float(lab.p_inter),.001)
if st.sidebar.button("⚙️ GENERAR EXPERIMENTO",use_container_width=True):
    st.session_state.lab=Lab(n,seed,pi,pe); st.rerun()

st.title("SITER-CAE v4.0 — Laboratorio Computacional")
st.caption("CDMX únicamente · Alcaldía → UTM → Sección → Manzana · sintético/agragado · seed + hash")
m=lab.net_metrics()
a,b,c,d,e=st.columns(5)
a.metric("Territorios",16); b.metric("Agentes",lab.n); c.metric("Aristas",m["n_aristas"])
d.metric("Densidad",f"{m['densidad']:.4f}"); e.metric("Seed",lab.seed)

tabs=st.tabs(["🌐 Mundo ABM","❓ 77 preguntas","🧭 SAF","🕸️ Red","🎲 Monte Carlo","💰 Optimización","🛡️ SPOF/Inercia","📦 Auditoría","🧩 NetLogo"])

with tabs[0]:
    st.subheader("Mundo ABM")
    x1,x2=st.columns([3,1])
    with x2:
        model=st.selectbox("Modelo",["Voter","q-Voter","Majority","Sznajd","Ising","Deffuant-Weisbuch","Hegselmann-Krause"])
        steps=st.slider("Ticks",5,100,20); beta=st.slider("beta / mu",.01,1.,.25,.01)
        eps=st.slider("epsilon",.05,1.,.40,.05); epsr=st.slider("epsilon repulsión",.5,1.5,.8,.05)
        shock=st.selectbox("Shock",["NINGUNO","NEGATIVO","POSITIVO"])
        if st.button("▶ EJECUTAR ABM",use_container_width=True):
            st.session_state.abm=lab.abm(model,steps,beta,eps,epsr,None if shock=="NINGUNO" else shock)
    with x1:
        q=lab.df()
        q["x"]=np.random.default_rng(lab.seed).random(len(q)); q["y"]=np.random.default_rng(lab.seed+1).random(len(q))
        fig=px.scatter(q,x="x",y="y",color="territorio",hover_data=["id","habilidades","influencia"],title="Agentes sintéticos")
        fig.update_layout(height=600); st.plotly_chart(fig,use_container_width=True)
    if "abm" in st.session_state:
        H,met,conv=st.session_state.abm
        st.json(met); st.line_chart(pd.DataFrame({"media":H.mean(1),"std":H.std(1)}))

with tabs[1]:
    st.subheader("Motor de las 77 preguntas")
    qsel=st.selectbox("Pregunta",[x for x in QUESTIONS])
    qid=qsel.split()[0]; T=lab.territories; N=lab.network
    def rowmax(col): return T.sort_values(col,ascending=False).iloc[0]
    if qid=="P1": r=rowmax("polarizacion"); ans=f"{r.territorio}: polarización={r.polarizacion:.3f}, entropía={r.entropia:.3f}."
    elif qid=="P2": r=rowmax("simpat_pct"); ans=f"{r.territorio}: estado positivo agregado={r.simpat_pct:.1%}."
    elif qid=="P3": r=rowmax("temperatura"); ans=f"{r.territorio}: temperatura sintética={r.temperatura:.1f}/100."
    elif qid=="P4": r=rowmax("mov_prom"); ans=f"{r.territorio}: movilidad={r.mov_prom:.3f}."
    elif qid=="P5": r=rowmax("habilidades_prom"); ans=f"{r.territorio}: habilidades={r.habilidades_prom:.3f}."
    elif qid=="P6": ans="; ".join(T[T.campo=="DISPUTA_ABIERTA"].territorio) or "ninguno."
    elif qid=="P7": ans="; ".join(T[T.campo=="CONSOLIDACION"].territorio) or "ninguno."
    elif qid=="P8": r=rowmax("conflicto"); ans=f"{r.territorio}: conflicto={r.conflicto:.3f}."
    elif qid=="P9": r=rowmax("fragmentacion"); ans=f"{r.territorio}: grupos relevantes={int(r.fragmentacion)}."
    elif qid=="P10": r=rowmax("institucionalizacion"); ans=f"{r.territorio}: institucionalización={r.institucionalizacion:.3f}."
    elif qid=="P11": r=rowmax("simpat_pct"); ans=f"{r.territorio}: dominancia positiva={r.simpat_pct:.1%}."
    elif qid=="P12": ans=json.dumps(lab.net_metrics(),indent=2)
    elif qid=="P13": ans=f"clustering={lab.net_metrics()['clustering']:.3f}; alto=sugiere camarillas."
    elif qid=="P14": ans=f"centralización={lab.net_metrics()['centralizacion']:.3f}."
    elif qid=="P15": ans=f"Gini={gini(N.influencia):.3f}; top1={top_share(N.influencia,.01):.1%}; top10={top_share(N.influencia,.10):.1%}."
    elif qid=="P16": ans=lab.centers(20).to_string(index=False)
    elif qid=="P17": ans=f"Gap top5-promedio={lab.centers(5).habilidades.mean()-N.habilidades.mean():+.3f}."
    elif qid=="P18": ans="Broker = nodo con conexión interterritorial; consulte Red para el cálculo."
    elif qid=="P19": ans=f"Top10% concentra {top_share(N.influencia,.10):.1%}."
    elif qid=="P20": ans=f"Convergencia={st.session_state.abm[1]['convergence_step']} ticks." if "abm" in st.session_state else "Ejecuta ABM."
    elif qid=="P21": ans="Se identifica en el ABM por superar p95 de conversiones acumuladas en tick 3."
    elif qid=="P22": ans="Ejecuta ABM; la tabla de conversiones por agente queda disponible."
    elif qid=="P23": r=N.sort_values("influencia",ascending=False).iloc[0]; ans=f"Hotspot sintético: {r.territorio}, influencia={r.influencia:.3f}."
    elif qid=="P24": r=rowmax("polarizacion"); ans=f"Hotspot: {r.territorio}, polarización={r.polarizacion:.3f}."
    elif qid=="P25": ans="Requiere rutas/GPS agregados; no se inventa cobertura sin traza."
    elif qid=="P26": ans="Modelar onda como campo espacio-temporal y estimar radio/atenuación por Monte Carlo."
    elif qid=="P27": ans=lab.centers(10)[["agent_id","territorio","grado","score_SAF"]].to_string(index=False)
    elif qid=="P28": ans="Requiere trazas de campo; sumar visitas por alcaldía."
    elif qid=="P29": ans="Monte Carlo con intervención BRIGADA."
    elif qid=="P30": ans=f"Shock negativo: {lab.mc(T.territorio.iloc[0],runs=250,shock=True)['prob_crisis']:.1%} de crisis en simulación."
    elif qid=="P31": ans=f"Conectividad: {lab.mc(T.territorio.iloc[0],'CONECTIVIDAD',.4,250)['delta_mean']:+.3f} Δ estabilidad."
    elif qid=="P32": ans="Costo/cobertura debe parametrizarse; el optimizador usa actores×500 + horas×25."
    elif qid=="P33": sp=lab.spof(); ans=sp.head(5).to_string(index=False)
    elif qid=="P34": ans="Comparar BRIGADA vs ESCUCHA con Monte Carlo y reportar distribución, no un valor único."
    elif qid=="P35": ans=f"P(mejora>5%)={lab.mc(T.territorio.iloc[0],runs=250)['prob_gt5']:.1%}."
    elif qid=="P36": r=T.sort_values("polarizacion",ascending=False).iloc[0]; ans=f"Actor puente abstracto en {r.territorio}: {lab.mc(r.territorio,'ACTOR_PUENTE',.4,250)['delta_mean']:+.3f}."
    elif qid=="P37": r=T.sort_values("polarizacion",ascending=False).iloc[0]; ans=f"Con contra-actor abstracto: {lab.mc(r.territorio,'ACTOR_PUENTE',.4,250,adversario=True)['delta_mean']:+.3f}."
    elif qid=="P38": ans="La capacidad baja reduce el efecto; pruebe sensibilidad de habilidades."
    elif qid=="P39": ans="Compare hub vs aislado mediante centralidad, conectividad y Monte Carlo."
    elif qid=="P40": ans="habilidades=.30 capital+.25 información+.25 liderazgo+.10 arraigo+.10 movilidad−.15 desconfianza."
    elif qid=="P41": ans="Estimar elasticidad resultado/grado mediante barrido de sensibilidad; no hay umbral universal."
    elif qid=="P42": r=T.sort_values("resistencia_prom",ascending=False).iloc[0]; ans=f"Mayor resistencia: {r.territorio}={r.resistencia_prom:.3f}."
    elif qid=="P43": ans="Modelo base de costo: actores×500 + horas×25."
    elif qid=="P44": 
        z=T.copy(); z["roi_proxy"]=(z.mov_prom+z.habilidades_prom)/(1+z.resistencia_prom); r=z.sort_values("roi_proxy",ascending=False).iloc[0]
        ans=f"ROI proxy mayor: {r.territorio}={r.roi_proxy:.3f}."
    elif qid=="P45": r=rowmax("resistencia_prom"); ans=f"Mayor aislamiento proxy por resistencia: {r.territorio}={r.resistencia_prom:.3f}."
    elif qid=="P46": ans="Buscar la menor magnitud cuyo delta esperado alcance +10%, con probabilidad objetivo definida."
    elif qid=="P47": z=lab.centers(max(10,int(lab.n*.1))); ans=f"Top10% con habilidades>0.7: {np.mean(z.habilidades>.7):.1%}."
    elif qid=="P48": ans=f"seed={lab.seed}; experiment_id={lab.experiment_id}; hash={lab.export()['metadata']['output_hash']}."
    elif qid=="P49": ans="PII=False; agentes sintéticos; resultados agregados."
    elif qid=="P50": ans="Exportación/Auditoría produce territorial_fields + indicadores + centros + metadata/hash."
    elif qid=="P51": ans="Costo = costos directos + horas + logística; calibrar con observaciones."
    elif qid=="P52": ans="Fatiga 0–1: aumenta con interacciones y recupera por tick; calibrar parámetros."
    elif qid=="P53": ans="Detener cuando dinero<=0 o horas<=0."
    elif qid=="P54": ans="Capital institucional desgastado reduce efectividad; requiere calibración."
    elif qid=="P55": ans="epsilon = máxima distancia entre estados para permitir atracción."
    elif qid=="P56": ans="Si |diff|>epsilon_repulsion se aplica repulsión: separación endógena."
    elif qid=="P57": ans="polarización endógena = std del estado continuo por tick."
    elif qid=="P58": ans="epsilon bajo restringe interacciones y puede impedir convergencia."
    elif qid=="P59": ans="Contra-actor espejo como perturbación agregada simétrica."
    elif qid=="P60":
        r=T.territorio.iloc[0]; a=lab.mc(r,'ACTOR_PUENTE',.4,250); b=lab.mc(r,'ACTOR_PUENTE',.4,250,adversario=True)
        ans=f"Sin contra-actor={a['delta_mean']:+.3f}; con={b['delta_mean']:+.3f}; diferencia={b['delta_mean']-a['delta_mean']:+.3f}."
    elif qid=="P61": ans="Flanqueo = perturbación sobre territorio vecino débil; se estudia como escenario, no como perfil individual."
    elif qid=="P62": ans="Decapitación = eliminar nodos de alta centralidad y medir la pérdida sistémica."
    elif qid=="P63": ans="Trigger abstracto: centralidad/influencia altas; no vigilancia individual."
    elif qid=="P64":
        al,hh=lab.optimize(); ans=f"Score final={hh.score.iloc[-1]:.3f}; costo={al.costo.sum():.0f}"; st.dataframe(al,use_container_width=True)
    elif qid=="P65": ans="Alta disputa → mayor prioridad de actor puente; consolidación + movilidad → mayor prioridad de horas."
    elif qid=="P66":
        al,hh=lab.optimize(); ans=hh.to_string(index=False); st.line_chart(hh.set_index("generation").score)
    elif qid=="P67": sp=lab.spof(); ans=sp[sp.SPOF].head(10).to_string(index=False) if len(sp) else "No hay SPOF críticos."
    elif qid=="P68": ans=lab.spof().head(10).to_string(index=False)
    elif qid=="P69": 
        k=int(lab.spof().SPOF.sum()); ans=f"SPOF críticos={k}; regla catálogo: 0 resiliente, 2–3 frágil."
    elif qid=="P70": r=rowmax("resistencia_prom"); ans=f"{r.territorio}: resistencia={r.resistencia_prom:.3f}."
    elif qid=="P71": r=rowmax("barrera_adopcion"); ans=f"{r.territorio}: barrera={r.barrera_adopcion:.3f}."
    elif qid=="P72": ans=", ".join(T[T.resistencia_prom>.6].territorio) or "ninguno >0.6."
    elif qid=="P73": ans="Inercia normativa = resistencia institucional que eleva la barrera colectiva."
    elif qid=="P74": ans="Fatiga = estado 0–1 que acumula exposición y se recupera con tiempo."
    elif qid=="P75": ans="Costo de polarizar = penalización cuando la diferencia supera el umbral de repulsión."
    elif qid=="P76": ans="Probar epsilon mayor, repulsión menor y mecanismos de interacción cruzada."
    else: ans="v4.0 añade recursos limitados, contra-actor, bounded confidence/repulsión, optimización, SPOF e inercia."
    st.markdown(f"### {qsel}")
    st.info(ans)

with tabs[2]:
    st.subheader("Campos SAF")
    st.dataframe(lab.territories,use_container_width=True,hide_index=True)
    fig=px.scatter(lab.territories,x="estabilidad",y="polarizacion",size="habilidades_prom",color="campo",hover_name="territorio")
    st.plotly_chart(fig,use_container_width=True)

with tabs[3]:
    st.subheader("Red multicapa sintética")
    st.json(lab.net_metrics()); st.dataframe(lab.centers(30),use_container_width=True,hide_index=True)

with tabs[4]:
    st.subheader("Monte Carlo")
    terr=st.selectbox("Territorio",ALCALDIAS,key="mc_t")
    typ=st.selectbox("Intervención",["BRIGADA","ACTOR_PUENTE","ESCUCHA","CONECTIVIDAD"])
    mag=st.slider("Magnitud",.01,1.,.2,.01); runs=st.slider("Corridas",20,1000,250,10)
    adv=st.checkbox("Contra-actor abstracto"); shock=st.checkbox("Shock negativo")
    if st.button("🎲 EJECUTAR",use_container_width=True):
        st.session_state.mc=lab.mc(terr,typ,mag,runs,shock,adv)
    if "mc" in st.session_state:
        z=st.session_state.mc
        a,b,c,d=st.columns(4); a.metric("Δ medio",f"{z['delta_mean']:+.3f}"); b.metric("P>5%",f"{z['prob_gt5']:.1%}")
        c.metric("P05",f"{z['p05']:+.3f}"); d.metric("P95",f"{z['p95']:+.3f}")
        st.plotly_chart(px.histogram(x=z["deltas"],nbins=35),use_container_width=True)

with tabs[5]:
    st.subheader("Recursos limitados / optimizador")
    budget=st.number_input("Presupuesto",100,100000,5000,100); hours=st.number_input("Horas",1,10000,100,1)
    gen=st.slider("Generaciones",5,100,20); pop=st.slider("Población",10,100,20)
    if st.button("🧬 OPTIMIZAR",use_container_width=True):
        st.session_state.opt=lab.optimize(budget,hours,gen,pop)
    if "opt" in st.session_state:
        al,hh=st.session_state.opt; st.dataframe(al,use_container_width=True,hide_index=True)
        st.line_chart(hh.set_index("generation").score)

with tabs[6]:
    st.subheader("SPOF / resiliencia / inercia")
    sp=lab.spof(); st.dataframe(sp,use_container_width=True,hide_index=True)
    st.metric("SPOF críticos",int(sp.SPOF.sum()))
    st.dataframe(lab.territories[["territorio","resistencia_prom","barrera_adopcion","institucionalizacion","campo"]].sort_values("resistencia_prom",ascending=False),use_container_width=True,hide_index=True)

with tabs[7]:
    st.subheader("Auditoría")
    p=lab.export(); st.json(p["metadata"])
    st.download_button("⬇️ JSON motor central",json.dumps(p,ensure_ascii=False,indent=2).encode(),"siter_cae_v40.json","application/json")
    st.download_button("⬇️ Territorios CSV",lab.territories.to_csv(index=False).encode(),"territorial_fields.csv","text/csv")
    st.download_button("⬇️ Agentes sintéticos CSV",lab.df().to_csv(index=False).encode(),"agents_synthetic.csv","text/csv")

with tabs[8]:
    st.subheader("NetLogo opcional")
    exe=netlogo_exe()
    if exe: st.success(f"NetLogo headless detectado: {exe}")
    else: st.warning("NetLogo no está instalado en este servidor; el ABM Python funciona sin él.")
    st.write("La app puede mostrar todo en Streamlit. NetLogo real se usa como motor externo cuando el servidor dispone de NetLogo/Java; para Streamlit Cloud el fallback Python es el camino directo.")
    if st.button("📝 CREAR MODELO .nlogo"):
        p=Path(tempfile.gettempdir())/"SITER_ABM_template.nlogo"; p.write_text(NETLOGO_TEMPLATE,encoding="utf-8")
        st.download_button("⬇️ SITER_ABM_template.nlogo",p.read_bytes(),"SITER_ABM_template.nlogo","text/plain")
    st.code("NETLOGO_HOME=/opt/netlogo\nNETLOGO_EXECUTABLE=/opt/netlogo/bin/netlogo-headless.sh\nNETLOGO_MODEL=/app/models/SITER_ABM.nlogo",language="bash")

st.divider()
st.caption("SITER-CAE v4.0 · 77 preguntas · CDMX · sintético/agregado · reproducible seed+hash")
