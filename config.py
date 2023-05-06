corpus = "Twitter"
stego_method = "VLC"
dataset = ["1bpw"]

tuning_param = ["lambda_loss", "main_learning_rate", "batch_size", "nepoch", "temperature", "SEED", "dataset"]
lambda_loss = [0.5]
temperature = [0.3]
batch_size = [8]
decay = 1e-02
main_learning_rate = [2e-05]

hidden_size = 768
nepoch = [10]

loss_type = "scl"  # ['ce', 'scl']
model_type = "electra"  # ['electra', 'bert']

is_waug = True  # if loss_type = "ce", please set is_waug = Flase
label_list = [None]
SEED = [0]

param = {"temperature": temperature, "corpus": corpus, "stego_method": stego_method, "dataset": dataset,
         "main_learning_rate": main_learning_rate, "batch_size": batch_size, "hidden_size": hidden_size,
         "nepoch": nepoch, "dataset": dataset, "lambda_loss": lambda_loss, "loss_type": loss_type,
         "decay": decay, "SEED": SEED, "model_type": model_type, "is_waug": is_waug}


