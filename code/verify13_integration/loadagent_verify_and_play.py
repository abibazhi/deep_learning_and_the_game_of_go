# verify13,
# 这是因为simple encoder的代码已经改了，变成了BHWC
# 但是alphago的encoder代码没有改，还是BCHW
# 但是alphago的NETWORK model已经改了，不得不改，变成了BHWC

# 就是alphagoModel+policyAgent时，policyAgent需要把BCHW转换成BHWC
# 但是当simpleModel+policyAgent时，不需要转换，因为encoder已经转好了。
# 所以，就需要两个policyAgent

from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.encoders.simple_fastpolicy import SimpleEncoder as BWHCSimpleEncoder
from dlgo.agent.predict_verify13 import DeepLearningAgent
from dlgo.rl.value_verify13 import ValueAgent #, load_experience
from dlgo.agent import AlphaGoMCTS

from keras.models import load_model
from dlgo import httpfrontend
from dlgo.agent.pg import PolicyAgent
from dlgo.agent.pg_verify13 import PolicyAgent as BWHCPolicyAgent


def load_fastpolicy_agent():
    # 9.3.3 Initializing an agent
    latest_model_path = "small_model_epoch_5.keras"
    print(latest_model_path)
    SimpleEncoder_policy_model = load_model(latest_model_path, compile=False)  # 加载模型权重


    encoder = BWHCSimpleEncoder((19, 19))
    print(f"BWHCSimple-encoder.name={encoder.name()}")
    policy_fast_agent = BWHCPolicyAgent(SimpleEncoder_policy_model, encoder)
    return policy_fast_agent

def load_prediction_agent():
    #这里是通过sl训练的模型
    latest_model_path = "alphago_con_200.keras"
    print(latest_model_path)
    alphago_policy_model = load_model(latest_model_path, compile=False)  # 加载模型权重

    encoder = AlphaGoEncoder()
    print(encoder.num_planes)

    alphago_deepleanring_agent = DeepLearningAgent(alphago_policy_model, encoder)
    return alphago_deepleanring_agent

    # model_file = h5py.File("../agents/deep_bot.h5", "r")
    # bot_from_file = load_prediction_agent(model_file)    
    
    

def load_policy_agent():
    #这里又通过了强化训练。不过这个和上面这个sl，是同一个模型，所以可以用同一个agent。
    latest_model_path = "alphago_rl_1.keras"
    alphago_rl_model = load_model(latest_model_path, compile=False)  # 加载模型权重

    encoder = AlphaGoEncoder()
    print(encoder.num_planes)

    #alphago_sl_agent = DeepLearningAgent(alphago_sl_policy, encoder)
    alphago_policy_agent = PolicyAgent(alphago_rl_model, encoder)
    return alphago_policy_agent


def load_value_agent():
    latest_model_path = "alphago_value_10.keras"
    alphago_value_model = load_model(latest_model_path, compile=False)  # 加载模型权重
    encoder = AlphaGoEncoder()
    print(encoder.num_planes)

    alphago_value_agent = ValueAgent(alphago_value_model, encoder)
    return alphago_value_agent

def load_alphago_agent():
    fast_policy = load_prediction_agent()
    strong_policy = load_policy_agent()
    value = load_value_agent()
    alphago = AlphaGoMCTS(strong_policy, fast_policy, value)
    return alphago 


def load_alphago_agent_new():
    fast_policy = load_fastpolicy_agent()
    strong_policy = load_policy_agent()
    value = load_value_agent()
    alphago = AlphaGoMCTS(strong_policy, fast_policy, value)
    return alphago 


#agent = load_policy_agent()
#agent = load_prediction_agent()
#agent = load_value_agent()
#agent = load_alphago_agent()
#agent = load_fastpolicy_agent()
agent = load_alphago_agent_new()
web_app = httpfrontend.get_web_app({'predict': agent})
web_app.run()