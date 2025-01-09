import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import gymnasium as gym
from gym_custom_envs.o2_env import AntEnv
from stable_baselines3.common.env_util import make_vec_env
# from gymnasium.wrappers import TimeLimit


env = AntEnv(render_mode='human')
# env = TimeLimit(AntEnv(render_mode="human"), max_episode_steps=100)
# env = make_vec_env(lambda: TimeLimit(AntEnv(render_mode=None), max_episode_steps=100), n_envs=1)
# env = make_vec_env(lambda: AntEnv(render_mode="human"), n_envs=1) 

env.reset()

# env.model: accedo a dati del modello statici (numero giunti, DoF, ..)
# env.data: accedo a dati dinamici durante la simulazione (forces, positions, ..)

# Test the environment
obs = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated:
        obs = env.reset()
env.close()



# Struttura che descrive i dettagli di ciascun contatto attualmente attivo nella simulazione.

# Esamina i dettagli del primo contatto
if env.data.ncon > 0:
    contact = env.data.contact[0]
    print(f"Posizione contatto: {contact.pos}")
    print(f"Normale: {contact.frame[:3]}")
    #print(f"Forza applicata (da cfrc_ext): {contact.force}")

# Forze esterne (contatti o forze esterne esplicite) applicate su ciascun corpo rigido nello spazio globale.

# Forza esterna totale (lineare e angolare) su un corpo specifico
body_id = 2
force = env.data.cfrc_ext[body_id, :3]  # Forze lineari
torque = env.data.cfrc_ext[body_id, 3:]  # Momenti (torque)

# Forze esercitate specificamente dai contatti nella simulazione.

# Itera su tutti i contatti per ottenere le forze
for i in range(env.data.ncon):  # Numero di contatti
    force = env.contact_forces[i]
    print(f"Contatto {i}: Forza = {force}")


# Coppie/forze generate dagli attuatori per muovere i giunti.
import mujoco

# Controllo delle forze generate dagli attuatori
for i in range(env.model.nv):  # nv = number of DoF
    joint_id = env.model.dof_jntid[i]  # Trova il giunto associato al DoF
    joint_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    print(f"DoF {i}: Forza generata = {env.data.qfrc_actuator[i]} (Giunto: {joint_name})")
