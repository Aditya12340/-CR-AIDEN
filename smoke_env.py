"""Exercise each action on one training episode; this is not a policy study."""

from data import load_split
from env import Action, SellExecutionEnv


def main():
    episode = next(load_split("training"))
    # This small fixed Q is a software smoke-test input, not a calibrated order size.
    quantity_btc = 0.01
    print(f"Training episode: {episode[0]['decision_time'].isoformat()}; Q={quantity_btc} BTC")
    for action in Action:
        env = SellExecutionEnv(episode, quantity_btc)
        while not env.done:
            env.step_action(action)
        result = env.summary()
        print(f"{action.name:16} steps={len(env.history):2} "
              f"completion={result.completion_fraction:.6f} "
              f"net_usdt={result.net_received_usdt:.6f} "
              f"score_bps={result.completion_score_bps:.6f}")


if __name__ == "__main__":
    main()
