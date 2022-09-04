import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.encoder_hidden_dims = (256,)
    config.dynamic_hidden_dims = (128, 128)
    config.embedding_dim = 256 # output layer unit for encoder and dynamic
    config.hidden_dims = () # actor, critic

    config.discount = 0.99

    config.expectile = 0.9  # The actual tau for expectiles.
    config.temperature = 10.0
    config.dropout_rate = None

    config.tau = 0.005  # For soft target updates.

    config.base_prob = 0.2
    config.last_layer_norm = False
    config.batch_norm = False

    return config
