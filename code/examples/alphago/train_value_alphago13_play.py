from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.agent.predict_verify13 import DeepLearningAgent
from dlgo.rl.q_verify13 import QAgent
from dlgo.rl.value_verify13 import ValueAgent #, load_experience

from keras.models import load_model
from dlgo import httpfrontend


latest_model_path = "../checkpoints/alphago_value_10.keras"
print(latest_model_path)
alphago_value = load_model(latest_model_path, compile=False)  # 加载模型权重


# rows, cols = 19, 19
# encoder = AlphaGoEncoder()
# input_shape = (encoder.num_planes, rows, cols)
# alphago_value_network = alphago_model(input_shape)

# alphago_value = ValueAgent(alphago_value_network, encoder)



encoder = AlphaGoEncoder()
print(encoder.num_planes)

#alphago_sl_agent = QAgent(alphago_sl_policy, encoder)
alphago_sl_agent = ValueAgent(alphago_value, encoder)

# model_file = h5py.File("../agents/deep_bot.h5", "r")
# bot_from_file = load_prediction_agent(model_file)
web_app = httpfrontend.get_web_app({'predict': alphago_sl_agent})
web_app.run()