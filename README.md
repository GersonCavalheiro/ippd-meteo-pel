# MeteoPel - Previsão Meteorológica para Pelotas, RS

Este projeto contém dois componentes principais:

1. **Script Python** (`get_meteo_data.py`) - Coleta dados meteorológicos da API Open-Meteo e gera um CSV
2. **Programa C** (`minha_previsao.c`) - Implementa um modelo Random Forest com OpenMP para prever chuva

---

## 1. Script Python: `get_meteo_data.py`

### Descrição
Obtém dados meteorológicos históricos (últimos 14 dias) e previsão (próximo 1 dia) para Pelotas, RS (lat: -31.765, lon: -52.337) via API Open-Meteo. Calcula a variação de pressão em 3 horas (`delta_pressao`) e salva em `dados_pelotas.csv` no formato esperado pelo programa C.

### Dependências Python
| Biblioteca | Instalação |
|------------|------------|
| `pandas` | `pip install pandas` |
| `requests` | `pip install requests` |

**Instalação rápida:**
```bash
pip install pandas requests
```

Ou usando `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Execução
```bash
python get_meteo_data.py
```

**Saída esperada:**
```
Arquivo 'dados_pelotas.csv' gerado com sucesso contendo N linhas!
```

O arquivo `dados_pelotas.csv` terá as colunas:
- `temperature_2m` - Temperatura a 2m (°C)
- `relative_humidity_2m` - Umidade relativa a 2m (%)
- `surface_pressure` - Pressão superficial (hPa)
- `delta_pressao` - Variação de pressão em 3h (hPa)
- `wind_speed_10m` - Velocidade do vento a 10m (km/h)

---

## 2. Programa C: `minha_previsao.c`

### Descrição
Implementa um Random Forest sintético (100 árvores, 15 nós cada) para prever precipitação horária. Usa OpenMP para paralelizar a inferência entre amostras. Classifica o risco de chuva em 5 categorias.

### Dependências
| Biblioteca | Descrição |
|------------|-----------|
| `libm` (math.h) | Funções matemáticas (FLT_MAX, etc.) - linkada automaticamente com `-lm` |
| `OpenMP` (omp.h) | Paralelismo - requer suporte do compilador (`-fopenmp`) |

### Compilação

**GCC (Linux/macOS):**
```bash
gcc -O3 -fopenmp -o minha_previsao minha_previsao.c -lm
```

**Clang:**
```bash
clang -O3 -fopenmp -o minha_previsao minha_previsao.c -lm
```

> **Nota:** Em macOS com Homebrew, instale `libomp` se necessário: `brew install libomp`

### Execução
```bash
# Usa o arquivo padrão dados_pelotas.csv
./minha_previsao

# Ou especifica outro CSV
./minha_previsao outro_arquivo.csv
```

**Saída esperada:**
```
=== RANDOM FOREST - INFERENCIA OPENMP ===

Arquivo: dados_pelotas.csv
Amostras carregadas: N

Faixas observadas:
Feature 0: XX.XX a XX.XX
Feature 1: XX.XX a XX.XX
Feature 2: XX.XX a XX.XX
Feature 3: XX.XX a XX.XX
Feature 4: XX.XX a XX.XX

Threads OpenMP: 8
Tempo de inferencia: 0.00XXXX segundos

--- PREVISÃO DE CHUVA PARA A PRÓXIMA HORA (PELOTAS) ---
Hora [0] -> Estimativa:  XX.XX mm/h | Status: Chuva Leve / Garoa
Hora [1] -> Estimativa:  XX.XX mm/h | Status: Tempo Seco / Estável
...
```

---

## Fluxo Completo

```bash
# 1. Gerar dados meteorológicos
python get_meteo_data.py

# 2. Compilar o modelo C
gcc -O3 -fopenmp -o minha_previsao minha_previsao.c -lm

# 3. Executar a previsão
./minha_previsao
```

---

## Estrutura do CSV de Entrada (esperado pelo C)

O programa C espera um CSV com **cabeçalho** e **5 colunas numéricas** na ordem:
1. temperature_2m
2. relative_humidity_2m
3. surface_pressure
4. delta_pressao
5. wind_speed_10m

Linhas com dados inválidos são ignoradas.

---

## Classificação de Risco de Chuva

| Precipitação (mm/h) | Classificação |
|---------------------|---------------|
| < 1.0 | Tempo Seco / Estável |
| 1.0 - 5.0 | Chuva Leve / Garoa |
| 5.0 - 15.0 | Chuva Moderada |
| 15.0 - 30.0 | ALERTA: Chuva Forte |
| ≥ 30.0 | PERIGO: Tempestade Severa |

---

## Requisitos do Sistema

- **Python 3.7+** com `pip`
- **Compilador C** com suporte a OpenMP (GCC 4.2+, Clang 3.7+)
- Conexão com internet (para acessar a API Open-Meteo)

---

## Licença

Projeto educacional/experimental para fins de demonstração de Random Forest paralelo em C.