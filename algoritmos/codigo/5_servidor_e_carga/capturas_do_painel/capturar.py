#!/usr/bin/env python3
"""Capturas do painel de controlo para o anexo, com o Brave sem janela (Playwright, sem browsers próprios).
Espera até cada doente DEMO ter pelo menos MIN_TENDENCIA previsões de tendência, e tira:
  1 visao_geral.png       separador Visão Geral
  2 monitorizacao.png     Monitorização com um doente selecionado (PPG ao vivo, sinais vitais, tendência)
  3 historico.png         Histórico de um doente
  4 historico_graficos.png  gráficos do histórico (janela que o botão abre)
  5 configuracao.png      separador Configuração
  6 atribuicao.png        formulário de atribuição, preenchido com um doente fictício (não é submetido)
As listas são filtradas pela caixa de pesquisa do próprio painel ("demonstração"), para não mostrar outros doentes.
Na Visão Geral (que não tem pesquisa) a resposta de /api/dashboard_resumo é filtrada aos doentes DEMO, com os contadores
recalculados pela regra do servidor; o doente de teste antigo (TESTE001, parado) não aparece."""
from playwright.sync_api import sync_playwright

SAIDA = "/home/filipe/Desktop/Tese/algoritmo-ml/Tentativa_9_10/capturas_painel"
URL = "http://127.0.0.1:5000/painel"
# a espera pelas previsões de tendência está em esperar.py (corre com o Python do servidor, que tem acesso à base)

def so_demo(route):
    """Visão Geral: só os doentes DEMO, com os contadores recalculados pela regra do servidor (server.py, dashboard_resumo)."""
    r = route.fetch(); d = r.json()
    d["pacientes"] = [x for x in d["pacientes"] if x["id"].startswith("PAC_00")]
    d["total_pacientes_ativos"] = len(d["pacientes"])
    d["alertas_criticos"] = sum(1 for x in d["pacientes"] if x["estado"] == "danger")
    d["observacoes_pendentes"] = sum(1 for x in d["pacientes"] if x["estado"] == "warning")
    route.fulfill(response=r, json=d)


with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/brave.com/brave/brave", headless=True)
    pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
    pg.route("**/api/dashboard_resumo", so_demo)
    pg.goto(URL); pg.wait_for_timeout(4000)
    pg.screenshot(path=f"{SAIDA}/visao_geral.png")

    pg.click('.tab-btn[data-tab="monitor"]'); pg.wait_for_timeout(1500)
    pg.fill("#pesquisaMonitor", "Paciente "); pg.wait_for_timeout(800)
    pg.click('#lista-pacientes-monitor .patient-item[data-id="PAC_001"]'); pg.wait_for_timeout(8000)
    pg.screenshot(path=f"{SAIDA}/monitorizacao.png")

    pg.click('.tab-btn[data-tab="historico"]'); pg.wait_for_timeout(1500)
    pg.fill("#pesquisaHistorico", "Paciente "); pg.wait_for_timeout(800)
    pg.click('#lista-todos-pacientes .historico-paciente-item[data-id="PAC_001"]'); pg.wait_for_timeout(4000)
    pg.screenshot(path=f"{SAIDA}/historico.png")
    if pg.is_visible("#btnVerGraficosHistorico"):
        pg.set_viewport_size({"width": 1440, "height": 1650})            # a janela dos gráficos inteira, com o detetor de tendência
        pg.click("#btnVerGraficosHistorico"); pg.wait_for_timeout(4000)
        pg.screenshot(path=f"{SAIDA}/historico_graficos.png")
        pg.keyboard.press("Escape"); pg.evaluate("document.getElementById('modalGraficosHistorico').style.display='none'")

    pg.set_viewport_size({"width": 1440, "height": 900})
    pg.click('.tab-btn[data-tab="config"]'); pg.wait_for_timeout(1500)
    pg.screenshot(path=f"{SAIDA}/configuracao.png")

    pg.set_viewport_size({"width": 1440, "height": 1300})                # o formulário inteiro, com o botão
    pg.click("#btnAdicionarPaciente"); pg.wait_for_timeout(1000)
    for campo, valor in (("#pulseira_input", "PULSEIRA_P4"), ("#paciente_id", "PAC_004"), ("#nome", "Paciente 4"),
                         ("#idade", "61"), ("#pressao_sistolica", "134"), ("#pressao_diastolica", "81")):
        pg.fill(campo, valor)
    pg.select_option("#sexo", index=1)
    pg.wait_for_timeout(500)
    pg.screenshot(path=f"{SAIDA}/atribuicao.png")
    b.close()
print("capturas feitas")
