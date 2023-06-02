import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_excel('./base/dataset.xlsx')

# Testando comando acima
# print(data)

# ALterando object types
# print(data.dtypes)

# Criando dict passando coluna Portátil com chave -> dicionário novo abaixo
alterando = {'Portatil': {'Smartphone': 1, 'Tablet': 2}}

# Inserir as alterações no data
data.replace(alterando, inplace=True) #adicionando 'alterando' dentro do 'data'
print(data) # agora, tudo que é smartphone é 1 e tablet é 2

y = data.Portatil
x = data.drop(columns= ['Portatil'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
# o computador vai aprender a diferenciar smartphone de tablet de acordo com a nossa base
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train) # são os 60% da base

# prediter ( o computador será testado para saber se de fato sabe diferenciar)
resp_pc = clf.predict(x_test) # indicando a base de teste
gabarito = y_test # será o nosso gabarito

# print(f'Resultados obtidos: \n{resp_pc}') # resposta do computador
# print(f'Gabarito ---------: \n{gabarito.values}') # .values() ->pegar valor dentro do test
# acima, resposta correta

print(f'Precisão: {str(metrics.precision_score(gabarito, resp_pc))}')

# git status -> verifica situação
# git add . ->
# git commit -m "Mensagem"
# git push origin main




