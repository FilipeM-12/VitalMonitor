"""
Tempo de cálculo por janela: cadeia própria (a de produção) contra NeuroKit.

PORQUE É QUE ISTO EXISTE
  A comparação em janelas emparelhadas mostra que a vantagem do NeuroKit não
  vem da extração. Falta saber o que custaria usá-lo em produção, porque o
  servidor extrai características de uma janela de 15 s por pulseira a cada
  5 s, e o custo por janela é o que limita o número de pulseiras por processo.

O QUE SE MEDE, E PORQUE É ESTA A COMPARAÇÃO JUSTA
  Mede-se o tempo de extrair as características de UMA janela de 15 s a 100 Hz,
  que é o que o worker faz.

  A cadeia própria corre `comum.extrair_features_janela`, exatamente a função de
  produção, sem cópia nem reimplementação.

  Do lado do NeuroKit corre-se a sequência que a extração offline usa
  (`ecg_clean`, `ecg_peaks`, `ecg_delineate` com wavelet, e `ppg_process`) sobre
  a mesma janela. A extração offline consegue amortizar a delineação uma vez por
  segmento inteiro, mas em produção não existe segmento, existe uma janela que
  chega a cada 5 s, por isso o custo por janela é o que conta.

  Só se mede a extração. Não se mede a inferência do modelo, que é a mesma nas
  duas cadeias e não depende desta escolha.

  NÃO COMPARAR com o teste de carga (1200 pulseiras por processo). Esse mede o
  serviço de tendência inteiro, com `extrair_features_janela_ppg` (só PPG,
  2,8 ms por janela) e um pool de 8 threads. Aqui mede-se
  `extrair_features_janela` (ECG e PPG, que é a cadeia da formulação absoluta,
  a única onde o NeuroKit foi ensaiado) e um só núcleo.

DADOS
  Sinais reais do dispositivo, em `arduino/Dados/ECG+PPG`, a 100 Hz, que é a
  cadência de produção. Usam-se os ficheiros `*_bruto_*` dos dois voluntários,
  porque a filtragem faz parte do que se está a medir. Não se usa o MIMIC porque
  está a 125 Hz e passaria pela reamostragem, que não faz parte do caminho de
  produção.
"""
import glob
import os
import sys
import time

import pandas as pd

SERVIDOR = "/home/filipe/Desktop/Tese/server"
SINAIS = ("/home/filipe/Desktop/Tese/arduino/Dados/ECG+PPG/Configuração 1/"
          "Todos_*_bruto_*.csv")
RAIZ = os.path.dirname(os.path.abspath(__file__))

REPETICOES = 5
# Cadência com que cada pulseira entrega uma janela nova ao servidor.
PERIODO_JANELA_S = 5

sys.path.insert(0, SERVIDOR)
import comum  # noqa: E402  (precisa do sys.path acima)
import neurokit2 as nk  # noqa: E402

FS = comum.FS
N = comum.FEATURE_WINDOW


def janelas_do_sinal():
    """Divide cada ficheiro em janelas de 15 s sem sobreposição."""
    for caminho in sorted(glob.glob(SINAIS)):
        df = pd.read_csv(caminho)
        nome = os.path.basename(caminho)
        for i in range(len(df) // N):
            fatia = df.iloc[i * N:(i + 1) * N]
            snap = [{"ecg": e, "ir": ir, "red": r}
                    for e, ir, r in zip(fatia.ecg, fatia.ir, fatia.red)]
            yield (f"{nome}#{i}", snap,
                   fatia.ecg.to_numpy(float), fatia.ir.to_numpy(float))


def custo_propria(snap):
    t = time.perf_counter()
    saida = comum.extrair_features_janela(snap, fs=FS)
    return time.perf_counter() - t, saida is not None


def custo_neurokit(ecg, ppg):
    t = time.perf_counter()
    ok = True
    try:
        limpo = nk.ecg_clean(ecg, sampling_rate=FS, method="neurokit")
        _, info = nk.ecg_peaks(limpo, sampling_rate=FS, method="neurokit")
        picos = info["ECG_R_Peaks"]
        if len(picos) >= 3:
            nk.ecg_delineate(limpo, picos, sampling_rate=FS, method="dwt")
        else:
            ok = False
        # O PPG do dispositivo é um sinal de absorção, sobe quando o volume de
        # sangue desce, e a extração própria inverte-o antes de o usar.
        nk.ppg_process(-ppg, sampling_rate=FS, method="elgendi")
    except Exception:
        ok = False
    return time.perf_counter() - t, ok


def main():
    linhas = []
    for janela, snap, ecg, ppg in janelas_do_sinal():
        for _ in range(REPETICOES):
            tp, ok_p = custo_propria(snap)
            tn, ok_n = custo_neurokit(ecg, ppg)
            linhas.append({"janela": janela, "propria_s": tp, "neurokit_s": tn,
                           "propria_extraiu": ok_p, "neurokit_extraiu": ok_n})

    d = pd.DataFrame(linhas)
    d.to_csv(f"{RAIZ}/custo_extracao.csv", index=False)

    mp, mn = d.propria_s.median(), d.neurokit_s.median()
    print(f"janelas de 15 s a {FS} Hz: {d.janela.nunique()}, "
          f"{REPETICOES} repetições cada")
    for nome, col in [("cadeia própria", "propria_s"), ("NeuroKit", "neurokit_s")]:
        print(f"{nome:16s} mediana {d[col].median() * 1000:7.2f} ms   "
              f"min {d[col].min() * 1000:7.2f}   max {d[col].max() * 1000:7.2f}")
    print(f"razão NeuroKit / própria: {mn / mp:.1f}x")
    print(f"janelas em que cada uma produziu características: "
          f"própria {d.propria_extraiu.mean():.0%}, NeuroKit {d.neurokit_extraiu.mean():.0%}")
    # Teto SÓ da extração, num núcleo. Não é a capacidade do serviço, que tem
    # ainda a receção por MQTT, a derivação, a inferência e a escrita na BD.
    print(f"teto de extração num núcleo, com uma janela a cada {PERIODO_JANELA_S} s: "
          f"própria {PERIODO_JANELA_S / mp:.0f} pulseiras, "
          f"NeuroKit {PERIODO_JANELA_S / mn:.0f}")
    print(f"\nEscrito: {RAIZ}/custo_extracao.csv")


if __name__ == "__main__":
    main()
