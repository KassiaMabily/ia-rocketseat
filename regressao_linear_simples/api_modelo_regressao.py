import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Criar uma instância do fastapi
app = FastAPI()

class request_body(BaseModel):
    horas_estudo: float

# Carregar modelo para realizar a predição
modelo_pontuacao = joblib.load('./modelo_regressao.pkl')

@app.post('/predict')
def predict(data: request_body):
    # Prepara os dados para predição
    input_feature = [[data.horas_estudo]]

    # Realizar a predição
    y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)

    return {'pontuacao_teste': y_pred.tolist()}


# uvicorn api_modelo_salario:app --reload
