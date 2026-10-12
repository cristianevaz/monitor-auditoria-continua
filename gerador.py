"""
gerador.py - Gera a base de teste (lançamentos fictícios + fraudes plantadas) e o gabarito.
Uso:  python gerador.py   ->  cria dados_exemplo/lancamentos_exemplo.csv e gabarito_exemplo.csv
"""
import os

import numpy as np
import pandas as pd

from motor import feriados

PERFIL = {  # usuário -> contas que costuma usar
    "Ana Silva": ["60205", "20105", "60420"],
    "Carlos Souza": ["60101", "60310", "60205"],
    "Beatriz Lima": ["20105", "60420", "60205"],
    "Roberto Alves": ["60310", "60420", "60101"],
    "Fernanda Costa": ["60101", "60205", "20105"],
    "Marcos Pereira": ["60205", "60420", "60310"],
    "Juliana Rocha": ["10102", "10102", "20105"],  # Tesouraria
}
CONTAS = {
    "60101": ("60101 - Despesas com Viagens", 0.6),
    "60205": ("60205 - Serviços de Terceiros", 1.4),
    "60310": ("60310 - Material de Escritório", 0.3),
    "60420": ("60420 - Manutenção e Reparos", 1.0),
    "20105": ("20105 - Fornecedores Nacionais", 1.8),
    "10102": ("10102 - Bancos Conta Movimento", 2.5),
}
FORNECEDORES = ["Alfa Serviços", "Beta Tecnologia", "Gama Viagens", "Delta Consultoria", "Ômega Transportes",
                "Sigma Papelaria", "Theta Manutenção", "Lambda Software", "Kappa Engenharia", "Épsilon Logística",
                "Iota Facilities", "Zeta Suprimentos", "Pi Treinamentos", "Rho Seguros"]
APROVADORES = ["Paulo Nunes", "Luciana Barros", "Helena Prado"]


def gerar_base(seed: int = 42, n: int = 1500):
    rng = np.random.default_rng(seed)

    # --- população "normal" ---
    cal = pd.date_range("2026-01-02", "2026-04-30", freq="D")
    fer = feriados([2026])
    peso = np.array([0.03 if (d.dayofweek >= 5 or d.date() in fer) else 1.0 for d in cal])
    datas = rng.choice(cal, size=n, p=peso / peso.sum())

    usuarios = rng.choice(list(PERFIL), size=n)
    contas = []
    for u in usuarios:
        cod = rng.choice(PERFIL[u]) if rng.random() < 0.96 else rng.choice(list(CONTAS))
        contas.append(cod)
    mult = np.array([CONTAS[c][1] for c in contas])
    valores = np.round(np.exp(np.log(1200 * mult) + 0.9 * rng.standard_normal(n)), 2)

    manual = rng.random(n) < 0.25
    hora = np.where(
        manual,
        np.where(rng.random(n) < 0.03, rng.choice([5, 6, 19, 21, 22], size=n), rng.integers(8, 19, size=n)),
        rng.integers(1, 4, size=n),  # lançamentos automáticos rodam de madrugada
    )
    minuto = rng.integers(0, 60, size=n)
    nfs = rng.choice(np.arange(100000, 999999), size=n, replace=False)

    df = pd.DataFrame({
        "Data": pd.to_datetime(datas),
        "Hora": [f"{h:02d}:{m:02d}" for h, m in zip(hora, minuto)],
        "Usuario": usuarios,
        "Aprovador": np.where(manual, rng.choice(APROVADORES, size=n), "Sistema"),
        "Conta_Contabil": [CONTAS[c][0] for c in contas],
        "Fornecedor": rng.choice(FORNECEDORES, size=n),
        "Numero_NF": [f"NF-{x}" for x in nfs],
        "Tipo_Lancamento": np.where(manual, "Manual", "Automático"),
        "Valor": valores,
        "Tag": "",
    })

    # --- fraudes plantadas ---
    def plantar(tag, data, hora_, usuario, aprov, conta, forn, nf, valor):
        return {"Data": pd.Timestamp(data), "Hora": hora_, "Usuario": usuario, "Aprovador": aprov,
                "Conta_Contabil": CONTAS[conta][0], "Fornecedor": forn, "Numero_NF": nf,
                "Tipo_Lancamento": "Manual", "Valor": valor, "Tag": tag}

    P = []
    # 1) Valores logo abaixo da alçada de R$ 10 mil, repetidos pelo mesmo usuário
    for d, v, nf in [("2026-01-14", 9870.00, "NF-310011"), ("2026-02-03", 9920.50, "NF-310012"),
                     ("2026-03-12", 9950.00, "NF-310013")]:
        P.append(plantar("1|Sub-alçada", d, "15:10", "Carlos Souza", "Paulo Nunes", "60205", "Delta Consultoria", nf, v))
    # 2) Lançamento manual de alto valor em domingo / feriado, fora do expediente
    P.append(plantar("2|Fim de semana/feriado", "2026-02-22", "23:40", "Ana Silva", "Luciana Barros", "60205",
                     "Ômega Transportes", "NF-320021", 48000.00))
    P.append(plantar("2|Fim de semana/feriado", "2026-04-21", "22:10", "Marcos Pereira", "Helena Prado", "60420",
                     "Beta Tecnologia", "NF-320022", 37500.00))
    # 3) Fraude de voucher: mesma nota com dígitos trocados
    P.append(plantar("3|Duplicidade (NF com digitação trocada)", "2026-03-03", "10:30", "Beatriz Lima", "Paulo Nunes",
                     "20105", "Alfa Serviços", "NF-482130", 12480.00))
    P.append(plantar("3|Duplicidade (NF com digitação trocada)", "2026-03-05", "11:15", "Beatriz Lima", "Paulo Nunes",
                     "20105", "Alfa Serviços", "NF-482310", 12480.00))
    # 4) Mesma NF lançada duas vezes, com 13 dias de intervalo
    P.append(plantar("4|Mesma NF lançada 2x", "2026-01-20", "09:45", "Roberto Alves", "Helena Prado", "60420",
                     "Beta Tecnologia", "NF-770412", 6725.50))
    P.append(plantar("4|Mesma NF lançada 2x", "2026-02-02", "14:20", "Roberto Alves", "Helena Prado", "60420",
                     "Beta Tecnologia", "NF-770412", 6725.50))
    # 5) Fracionamento: 4 parcelas em 4 dias que somam R$ 16 mil
    for d, v, nf, h in [("2026-03-16", 4200.00, "NF-550101", "16:05"), ("2026-03-17", 3900.00, "NF-550102", "16:20"),
                        ("2026-03-18", 4100.00, "NF-550103", "16:10"), ("2026-03-19", 3800.00, "NF-550104", "16:30")]:
        P.append(plantar("5|Fracionamento", d, h, "Fernanda Costa", "Luciana Barros", "60101", "Gama Viagens", nf, v))
    # 6) Segregação de funções: quem lança também aprova
    for d, v, nf in [("2026-02-09", 18940.30, "NF-660031"), ("2026-03-23", 7315.75, "NF-660032"),
                     ("2026-04-08", 26180.90, "NF-660033")]:
        P.append(plantar("6|Segregação de funções", d, "11:00", "Marcos Pereira", "Marcos Pereira", "60205",
                         "Kappa Engenharia", nf, v))
    # 7) Usuário que nunca lança em Bancos, valor redondo, fornecedor inédito, à noite
    P.append(plantar("7|Conta atípica + valor redondo", "2026-04-14", "21:50", "Roberto Alves", "Paulo Nunes", "10102",
                     "Zeta Holding Ltda", "NF-880001", 50000.00))

    df = pd.concat([df, pd.DataFrame(P)], ignore_index=True).sort_values(["Data", "Hora"]).reset_index(drop=True)
    df.insert(0, "ID_Transacao", [str(100001 + i) for i in range(len(df))])

    gab = df[df["Tag"] != ""][["ID_Transacao", "Tag"]].copy()
    gab["Cenario"] = gab["Tag"].str.split("|").str[1]
    gab = gab.drop(columns="Tag").reset_index(drop=True)
    return df.drop(columns="Tag"), gab


def para_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV no padrão brasileiro (; e vírgula decimal, data dd/mm/aaaa)."""
    out = df.copy()
    if pd.api.types.is_datetime64_any_dtype(out["Data"]):
        out["Data"] = out["Data"].dt.strftime("%d/%m/%Y")
    return out.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


if __name__ == "__main__":
    base, gabarito = gerar_base()
    os.makedirs("dados_exemplo", exist_ok=True)
    open("dados_exemplo/lancamentos_exemplo.csv", "wb").write(para_csv_bytes(base))
    open("dados_exemplo/gabarito_exemplo.csv", "wb").write(gabarito.to_csv(index=False, sep=";").encode("utf-8-sig"))
    open("dados_exemplo/modelo_vazio.csv", "wb").write(para_csv_bytes(base.head(3)))
    print(f"{len(base)} lançamentos gerados, {len(gabarito)} fraudes plantadas.")