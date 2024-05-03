from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.agent.predict_verify13 import DeepLearningAgent
from keras.models import load_model
from dlgo import httpfrontend


#这里是通过sl训练的模型
#latest_model_path = "../../checkpoints/alphago_con_200.keras"
#print(latest_model_path)
#alphago_sl_policy = load_model(latest_model_path, compile=False)  # 加载模型权重

#这里又通过了强化训练。不过这个和上面这个sl，是同一个模型，所以可以用同一个agent。
latest_model_path = "../checkpoints/alphago_rl_1.keras"
print(latest_model_path)
alphago_sl_policy = load_model(latest_model_path, compile=False)  # 加载模型权重

encoder = AlphaGoEncoder()
print(encoder.num_planes)

alphago_sl_agent = DeepLearningAgent(alphago_sl_policy, encoder)

# model_file = h5py.File("../agents/deep_bot.h5", "r")
# bot_from_file = load_prediction_agent(model_file)
web_app = httpfrontend.get_web_app({'predict': alphago_sl_agent})
web_app.run()