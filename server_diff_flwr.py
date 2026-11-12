import flwr as fl


strategy = fl.server.strategy.FedAvg(

    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,

)

fl.server.start_server(

    server_address="localhost:8080",

    config=fl.server.ServerConfig(num_rounds=3),

    strategy=strategy,
)