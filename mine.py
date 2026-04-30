from collections import deque
import numpy as np
import pong_rl_environment as envi
import deepQ_agent as ag
import random

# ===== SPIELMODUS AUSWÄHLEN =====
# Ändern Sie diese Variable um zwischen Spielmodi zu wechseln:
# True  -> Mensch spielt links gegen KI rechts
# False -> KI vs. KI (Training)
left_human = False
# ================================

def play_human_vs_ai():
    """Mensch spielt gegen die KI (Mensch links, KI rechts)"""
    history_len = 10
    env = envi.pong_environment(render=True)
    agent_right = ag.my_agent(8 * history_len, 3, loadmodel=True, trainme=True)

    positiondata = env.give_start_state()
    history = deque([positiondata] * history_len, maxlen=history_len)

    print("\n=== Mensch vs. KI ===")
    print("Steuerung: W = Oben, S = Unten")
    print("Drücke ESC oder schließe das Fenster zum Beenden\n")

    while True:
        state = np.concatenate(history)
        actionrightpaddle = agent_right.get_action(state)

        next_positiondata, reward, rewardleft, done, running = env.one_step(
            actionrightpaddle,
            human=True,
            actionleftpaddle=2
        )

        if not running:
            break

        history.append(next_positiondata)

def play_ai_vs_ai():
    """Zwei KIs spielen gegeneinander und trainieren dabei"""
    history_len = 10
    env = envi.pong_environment(render=True)
    agent_left = ag.my_agent(8 * history_len, 3, loadmodel=True, trainme=True, filename="pong_left.keras")
    agent_right = ag.my_agent(8 * history_len, 3, loadmodel=True, trainme=True, filename="pong_right.keras")

    positiondata = env.give_start_state()
    history_left = deque([positiondata] * history_len, maxlen=history_len)
    history_right = deque([positiondata] * history_len, maxlen=history_len)

    print("\n=== KI vs. KI (Training) ===")
    print("Beobachte die beiden KIs beim Spielen und Trainieren!")
    print("Linke KI speichert in: pong_left.keras")
    print("Rechte KI speichert in: pong_right.keras")
    print("Schließe das Fenster zum Beenden\n")

    while True:
        state_left = np.concatenate(history_left)
        state_right = np.concatenate(history_right)

        actionleftpaddle = agent_left.get_action(state_left)
        actionrightpaddle = agent_right.get_action(state_right)

        next_positiondata, reward, rewardleft, done, running = env.one_step(
            actionrightpaddle,
            human=False,
            actionleftpaddle=actionleftpaddle
        )

        if not running:
            break

        history_left.append(next_positiondata)
        history_right.append(next_positiondata)
        next_state_left = np.concatenate(history_left)
        next_state_right = np.concatenate(history_right)

        # Speichere Experience für beide Agenten
        agent_left.memory.append((state_left, actionleftpaddle, rewardleft, next_state_left, done))
        agent_right.memory.append((state_right, actionrightpaddle, reward, next_state_right, done))

        # Trainiere beide Agenten
        agent_left.train()
        agent_right.train()

def main():
    if left_human:
        play_human_vs_ai()
    else:
        play_ai_vs_ai()

if __name__ == "__main__":
    main()
