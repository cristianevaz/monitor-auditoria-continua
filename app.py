"""
app.py - Interface Streamlit do Monitor de Riscos Contínuos & Fraudes.
Execute com:  streamlit run app.py
"""
import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

import gerador
import motor
from motor import ROTULOS, Parametros, brl

st.set_page_config(page_title="Monitor de Riscos Contínuos & Fraudes", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------------------
# Funções com cache
# ---------------------------------------------------------------------------
@st.cache_data
def base_demo():
    df, gab = gerador.gerar_base()
    return gerador.para_csv_bytes(df), gab


@st.cache_data(show_spinner="Analisando 100% dos lançamentos...")
def rodar(raw: bytes, nome: str, params_json: str) -> dict:
    p = Parametros(**json.loads(params_json))
    return motor.analisar_arquivo(raw, nome, p)


# ---------------------------------------------------------------------------
# Cabeçalho: o que é, em 10 segundos
# ---------------------------------------------------------------------------
st.title("🛡️ Monitor de Riscos Contínuos & Fraudes")
st.markdown(
    "Testa 100% dos seus lançamentos contábeis (e não uma amostra) com regras de auditoria + Machine Learning, "
    "aponta *o que* é suspeito e *por quê*, e entrega o papel de trabalho com evidência pronta para a auditoria."
)
c1, c2, c3 = st.columns(3)
with c1.container(border=True):
    st.markdown("1️⃣ Carregue os dados")
    st.caption("CSV do razão/diário (ou use a base de demonstração ao lado, com 17 fraudes plantadas).")
with c2.container(border=True):
    st.markdown("2️⃣ A análise testa tudo")
    st.caption("SoD, duplicidade de NF, fracionamento, sub-alçada, feriados/horários, valores atípicos, Benford e IA.")
with c3.container(border=True):
    st.markdown("3️⃣ Triagem e evidência")
    st.caption("Score 0-100 com motivo em português, triagem de casos e Excel com hash, parâmetros e parecer.")

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuração")
modo = st.sidebar.radio("Dados", ["🧪 Base de demonstração", "📁 Enviar meu CSV"])

raw_demo, gabarito = base_demo()
with st.sidebar.expander("📎 Arquivos de teste"):
    st.caption("Baixe, abra no Excel, altere à vontade e envie de volta pelo modo 'Enviar meu CSV'.")
    st.download_button("Base de exemplo (1.517 lançamentos)", raw_demo, "lancamentos_exemplo.csv", "text/csv")
    st.download_button("Gabarito das fraudes plantadas",
                       gabarito.to_csv(index=False, sep=";").encode("utf-8-sig"), "gabarito_exemplo.csv", "text/csv")
    modelo = "\n".join(raw_demo.decode("utf-8-sig").splitlines()[:4]).encode("utf-8-sig")
    st.download_button("Modelo (só cabeçalho + 3 linhas)", modelo, "modelo_vazio.csv", "text/csv")

with st.sidebar.expander("Regras e limites"):
    alcada = st.number_input("Alçada de aprovação (R$)", min_value=1.0, value=10000.0, step=500.0)
    margem = st.slider("'Logo abaixo da alçada' (% abaixo)", 1, 20, 5)
    jan_dup = st.slider("Janela de duplicidade (dias)", 0, 30, 5)
    tol = st.number_input("Tolerância de valor na duplicidade (R$)", min_value=0.0, value=0.05, step=0.05)
    jan_frac = st.slider("Janela de fracionamento (dias)", 1, 30, 5)
    parcela = st.slider("Parcela mínima p/ fracionamento (% da alçada)", 10, 90, 35)
    z = st.slider("Corte de valor atípico (z-score)", 2.0, 6.0, 3.0, 0.5)
    h1, h2 = st.slider("Expediente (horas)", 0, 24, (7, 20))
    contam = st.slider("Anomalias esperadas pela IA (%)", 1, 10, 3)

pesos = dict(motor.PESOS_PADRAO)
with st.sidebar.expander("Pesos do score (pontos)"):
    for chave, rotulo in ROTULOS.items():
        pesos[chave] = st.slider(rotulo, 0, 100, int(motor.PESOS_PADRAO[chave]))
    peso_ia = st.slider("IA (máximo de pontos)", 0, 60, 30)

params = Parametros(alcada=alcada, margem_pct=margem, janela_dup_dias=jan_dup, tol_valor=tol,
                    janela_frac_dias=jan_frac, parcela_min_pct=parcela, z_corte=z, hora_ini=h1, hora_fim=h2,
                    contaminacao=contam / 100, peso_ia=peso_ia, pesos=pesos)

# ---------------------------------------------------------------------------
# Entrada de dados
# ---------------------------------------------------------------------------
demo = modo.startswith("🧪")
if demo:
    raw, nome = raw_demo, "lancamentos_exemplo.csv (demonstração)"
else:
    up = st.sidebar.file_uploader("CSV de lançamentos", type=["csv"])
    if up is None:
        st.info("Envie um CSV pela barra lateral ou volte para a base de demonstração.")
        st.markdown("Colunas obrigatórias: `ID_Transacao`, `Data`, `Usuario`, `Conta_Contabil`, `Valor`  \n"
                    "Opcionais (ativam mais testes): `Fornecedor`, `Numero_NF` (duplicidade e fracionamento), "
                    "`Hora` e `Tipo_Lancamento` (fora do expediente), `Aprovador` (segregação de funções).  \n"
                    "Aceita separador `;` `,` ou tab, datas `dd/mm/aaaa` ou `aaaa-mm-dd` e valores `1.234,56` ou `1234.56`.")
        st.stop()
    raw, nome = up.getvalue(), up.name

try:
    res = rodar(raw, nome, json.dumps(asdict(params), sort_keys=True))
except ValueError as erro:
    st.error(str(erro))
    st.stop()

df, evid = res["df"], res["evidencia"]
for aviso in res["avisos"]:
    st.warning(aviso)

if st.session_state.get("hash_dados") != evid["SHA-256 do arquivo"]:
    st.session_state["hash_dados"] = evid["SHA-256 do arquivo"]
    st.session_state["triagem"] = {}
triagem = st.session_state["triagem"]

NIVEIS_ORD = ["Crítico", "Alto", "Médio", "Baixo"]
tab_resumo, tab_alertas, tab_triagem, tab_benford, tab_evid = st.tabs(
    ["📊 Resumo", "🚨 Alertas", "🗂️ Triagem de casos", "📈 Benford", "🧾 Evidência & Metodologia"])

# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------
with tab_resumo:
    ca = df["Nivel_Risco"].isin(["Crítico", "Alto"])
    k = st.columns(5)
    k[0].metric("Transações analisadas", f"{len(df):,}".replace(",", "."))
    k[1].metric("Valor analisado", brl(df["Valor"].sum()))
    k[2].metric("🔴 Críticos", int((df["Nivel_Risco"] == "Crítico").sum()))
    k[3].metric("🟠 Altos", int((df["Nivel_Risco"] == "Alto").sum()))
    k[4].metric("Valor sob suspeita", brl(df.loc[ca, "Valor"].sum()))

    if demo:
        with st.container(border=True):
            st.markdown("#### 🎯 Teste de eficácia (a base de demonstração tem fraudes plantadas)")
            m = gabarito.merge(df[["ID_Transacao", "Score_Risco", "Nivel_Risco"]], on="ID_Transacao")
            achou = m["Nivel_Risco"].isin(["Crítico", "Alto"])
            fp = int((ca & ~df["ID_Transacao"].isin(gabarito["ID_Transacao"])).sum())
            a, b, c = st.columns(3)
            a.metric("Fraudes plantadas", len(gabarito))
            b.metric("Encontradas (Alto/Crítico)", f"{int(achou.sum())} de {len(gabarito)}")
            c.metric("Outros alertas Alto/Crítico", fp, help="Alertas fora do gabarito: precisam de validação do auditor.")
            por_cenario = m.groupby("Cenario").agg(Lançamentos=("ID_Transacao", "size"),
                                                   Score_máx=("Score_Risco", "max"), Score_mín=("Score_Risco", "min"))
            st.dataframe(por_cenario, width="stretch")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("Transações por nível de risco")
        st.bar_chart(df["Nivel_Risco"].value_counts().reindex(NIVEIS_ORD).fillna(0))
    with g2:
        st.markdown("Alertas disparados por tipo de teste")
        st.bar_chart(pd.Series({ROTULOS[c]: int(df[c].sum()) for c in ROTULOS}).sort_values(ascending=False))

    st.markdown("Perfil de risco por usuário")
    st.dataframe(motor.perfil_usuarios(df), width="stretch", hide_index=True)
    st.markdown("Mapa das transações (valor x data)")
    st.scatter_chart(df[["Data", "Valor", "Nivel_Risco"]], x="Data", y="Valor", color="Nivel_Risco")

# ---------------------------------------------------------------------------
# Alertas
# ---------------------------------------------------------------------------
cols_tab = ["ID_Transacao", "Data", "Dia_Semana", "Hora", "Usuario", "Aprovador", "Conta_Contabil", "Fornecedor",
            "Numero_NF", "Valor", "Score_Risco", "Nivel_Risco", "Motivos", "Acao_Sugerida"]
cols_tab = [c for c in cols_tab if c in df.columns]
cfg = {
    "Score_Risco": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
    "Valor": st.column_config.NumberColumn("Valor (R$)", format="R$ %.2f"),
    "Data": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
}
with tab_alertas:
    f1, f2, f3 = st.columns(3)
    niveis = f1.multiselect("Níveis", NIVEIS_ORD, default=["Crítico", "Alto"])
    tipos = f2.multiselect("Só quem disparou o teste", list(ROTULOS), format_func=lambda c: ROTULOS[c])
    usuario = f3.selectbox("Usuário", ["Todos"] + sorted(df["Usuario"].unique().tolist()))
    vis = df[df["Nivel_Risco"].isin(niveis)]
    for t in tipos:
        vis = vis[vis[t]]
    if usuario != "Todos":
        vis = vis[vis["Usuario"] == usuario]
    vis = vis.sort_values("Score_Risco", ascending=False)
    st.write(f"{len(vis)} transação(ões) · valor total {brl(vis['Valor'].sum())}")
    st.dataframe(vis[cols_tab], width="stretch", hide_index=True, column_config=cfg)

# ---------------------------------------------------------------------------
# Triagem
# ---------------------------------------------------------------------------
with tab_triagem:
    st.caption("Registre o andamento de cada caso. Os dados ficam nesta sessão e saem no Excel (aba 'Triagem').")
    lim = st.multiselect("Níveis na fila", NIVEIS_ORD, default=["Crítico", "Alto"], key="niveis_triagem")
    fila = df[df["Nivel_Risco"].isin(lim)].sort_values("Score_Risco", ascending=False).head(300)
    base = fila[["ID_Transacao", "Nivel_Risco", "Score_Risco", "Usuario", "Valor", "Motivos"]].copy()
    base["Status"] = base["ID_Transacao"].map(lambda i: triagem.get(i, {}).get("Status", "Novo"))
    base["Responsavel"] = base["ID_Transacao"].map(lambda i: triagem.get(i, {}).get("Responsavel", ""))
    base["Parecer"] = base["ID_Transacao"].map(lambda i: triagem.get(i, {}).get("Parecer", ""))
    editado = st.data_editor(
        base, hide_index=True, width="stretch", key=f"editor_{hash(tuple(base['ID_Transacao']))}",
        disabled=["ID_Transacao", "Nivel_Risco", "Score_Risco", "Usuario", "Valor", "Motivos"],
        column_config={"Status": st.column_config.SelectboxColumn("Status", options=motor.STATUS, required=True),
                       "Valor": st.column_config.NumberColumn("Valor (R$)", format="R$ %.2f")})
    for _, r in editado.iterrows():
        triagem[r["ID_Transacao"]] = {"Status": r["Status"], "Responsavel": r["Responsavel"], "Parecer": r["Parecer"]}
    st.write("Andamento: " + " · ".join(f"{s}: {sum(1 for v in triagem.values() if v['Status'] == s)}"
                                            for s in motor.STATUS))

# ---------------------------------------------------------------------------
# Benford
# ---------------------------------------------------------------------------
with tab_benford:
    st.markdown("A Lei de Benford diz que, em dados contábeis naturais, o primeiro dígito dos valores segue uma "
                "distribuição conhecida (o dígito 1 aparece ~30% das vezes). Desvios grandes sugerem valores "
                "inventados, arredondados ou lançados para ficar abaixo de limites.")
    b1, b2, b3 = st.columns(3)
    b1.metric("Resultado (base toda)", res["benford_classe"])
    b2.metric("MAD", f"{res['benford_mad']:.4f}", help="Desvio absoluto médio. < 0,006 = conformidade alta.")
    b3.metric("Valores testados", res["benford_n"])
    st.line_chart(res["benford"].set_index("Digito")[["Observado (%)", "Esperado (%)"]])
    st.markdown("Por conta contábil (só conclusivo com 300+ lançamentos)")
    st.dataframe(res["benford_contas"], width="stretch", hide_index=True)
    st.caption("Benford é um indício estatístico no nível da população; ele não aponta transações individuais.")

# ---------------------------------------------------------------------------
# Evidência & metodologia
# ---------------------------------------------------------------------------
with tab_evid:
    st.markdown("#### Trilha de evidência desta execução")
    st.table(pd.DataFrame({"Item": [k for k in evid if k != "Parâmetros"],
                           "Valor": [str(v) for k, v in evid.items() if k != "Parâmetros"]}).set_index("Item"))
    st.download_button("⬇️ Baixar papel de trabalho (Excel)",
                       motor.exportar_excel(df, evid, res["benford"], triagem),
                       f"papel_de_trabalho_{evid['ID da execução']}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with st.expander("Parâmetros usados"):
        st.json(json.loads(evid["Parâmetros"]))

    st.markdown("#### Como o score é calculado")
    st.markdown("Score (0-100) = pontos dos testes disparados + pontos da IA (limitado a 100). "
                "Níveis: Crítico ≥ 70 · Alto 50-69 · Médio 30-49 · Baixo < 30.")
    st.table(pd.DataFrame({"Teste": [ROTULOS[c] for c in ROTULOS],
                           "Pontos": [params.pesos[c] for c in ROTULOS]}).set_index("Teste"))
    st.caption(f"IA (Isolation Forest): pontua só o top 20% mais anômalo, até {params.peso_ia:.0f} pontos. "
               "Considera valor, dia, horário, tipo de lançamento, raridade usuário × conta e fornecedor.")

    with st.expander("Por que isto não é 'só um agente de IA'?"):
        st.markdown(
            "- Reprodutível: mesmo arquivo + mesmos parâmetros = mesmo resultado, comprovado por hash e ID de execução. "
            "Um chat não garante isso.\n"
            "- População completa: testa todas as linhas, sem limite de contexto e sem 'alucinar' contas.\n"
            "- Explicável e calibrável: cada ponto do score é rastreável e ajustável ao apetite de risco.\n"
            "- Dados ficam com você: roda local, sem enviar o razão para terceiros.\n"
            "- Usável por agentes: a análise funciona como ferramenta, por linha de comando ou como biblioteca Python. "
            "O agente conversa; a ferramenta testa e gera a evidência.")
    with st.expander("Limitações"):
        st.markdown("A IA é não supervisionada e relativa à própria base: aponta o que foge do padrão, não comprova "
                    "fraude. Todo alerta exige validação do auditor. A análise é em lote sobre o arquivo carregado; "
                    "tempo real exigiria integração com o ERP/banco. Feriados: nacionais e pontos facultativos usuais "
                    "(não inclui feriados estaduais/municipais).")