# 这两个policy程序，对弈后的经验数据，也顺便用来被自己训练。
# 这是个示例程序，实际要循环训练后，然后确认训练后的agent比之前的强。
# 这个确认过程和训练过程是非常耗时的。之前只是验证了其可行性，现在没有资源来做这个工作！
# tag::load_opponents[]
from dlgo.agent.pg import PolicyAgent
from dlgo.agent.predict import load_prediction_agent
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.rl.simulate import experience_simulation
import h5py
from keras.models import load_model


encoder = AlphaGoEncoder()
print(encoder.num_planes)

#sl_agent_model_path = "../../checkpoints/alphago_con_200.keras" #最原始的sl训练的模型，
rl_agent_model_path = "../checkpoints/alphago_rl_1.keras" # rl训练后得到的模型
opponent_model_path = "../../checkpoints/alphago_con_200.keras" #和上面这个一样，但是要被打败的模型。

# 这里的模型还是深度学习模型
rl_agent_model = load_model(rl_agent_model_path, compile=False)  # 加载模型权重
opponent_model = load_model(opponent_model_path, compile=False)  # 加载模型权重



#alphago_sl_agent = DeepLearningAgent(alphago_sl_policy, encoder)
#sl_agent = load_prediction_agent(h5py.File('alphago_sl_policy.h5'))
#sl_opponent = load_prediction_agent(h5py.File('alphago_sl_policy.h5'))

# 这两个变成agent模型了，有了select_move
alphago_rl_agent = PolicyAgent(rl_agent_model, encoder)
opponent = PolicyAgent(opponent_model, encoder)
# end::load_opponents[]

# tag::run_simulation[]
num_games = 80
experience = experience_simulation(num_games, alphago_rl_agent, opponent)

alphago_rl_agent.train(experience)

with h5py.File('alphago_rl_policy.h5', 'w') as rl_agent_out:
    alphago_rl_agent.serialize(rl_agent_out)

with h5py.File('alphago_rl_experience.h5', 'w') as exp_out:
    experience.serialize(exp_out)
# end::run_simulation[]
