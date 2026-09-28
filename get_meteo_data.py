import pandas as pd
import requests

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": -31.765,
    "longitude": -52.337,
    "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
    "past_days": 14,
    "forecast_days": 1,
    "timezone": "America/Sao_Paulo",
}

# 1. Faz o download dos dados
response = requests.get(url, params=params)
data = response.json()["hourly"]

# 2. Converte para DataFrame e calcula a Variação de Pressão em 3h (delta_pressao)
df = pd.DataFrame(data)
df["delta_pressao"] = df["surface_pressure"].diff(3)

# 3. Formata as colunas exatamente como o programa C espera
df_c = df[
    [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "delta_pressao",
        "wind_speed_10m",
    ]
].dropna()

# 4. Salva no formato CSV pronto para o C
df_c.to_csv("dados_pelotas.csv", index=False, header=True)
print(
    f"Arquivo 'dados_pelotas.csv' gerado com sucesso contendo {len(df_c)} linhas!"
)
