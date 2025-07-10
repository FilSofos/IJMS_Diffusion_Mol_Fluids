import pandas as pd
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split

#Read data
df = pd.DataFrame(pd.read_csv("Bulk.csv"))

X = pd.DataFrame(df, columns=["Rho_reduced","Temp_reduced"])
X = X.rename(columns={"Rho_reduced": "x1", "Temp_reduced": "x2"})
y = pd.DataFrame(df, columns=["D_reduced"])
y = y.rename(columns={"D_reduced": "y"})

#Run SR
no_of_reruns = 41

Train_percentage = 0.80
Test_percentage = 0.20

for i in range(1,no_of_reruns):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=Train_percentage, test_size =Test_percentage, random_state = i)
    
    model = PySRRegressor(
        procs=14,
        populations=30,
        population_size=1000,
        ncyclesperiteration=200,
        niterations=100,
        maxsize=30,
        maxdepth=10,
        binary_operators=["*", "^", "+", "-", "/"], 
        unary_operators=["exp","log", "sqrt", "square"],    # "sqrt" and "square" can be excluded since "^" is used
        progress=True,
        nested_constraints={"exp": {"log": 0, "exp": 0},
        "log": {"log": 0, "exp": 0}},
        weight_randomize=0.1,
        cluster_manager=None,
        precision=64,
        turbo=True,
        julia_project=None,
        update=False,
        complexity_of_variables=2,
        constraints={"^": (2, 1)})
    model.fit(X_train,y_train)
    equations_df = pd.DataFrame(model.equations_)
    equations_df.to_csv(f"Results\\Confined_{i}.csv", index=False)